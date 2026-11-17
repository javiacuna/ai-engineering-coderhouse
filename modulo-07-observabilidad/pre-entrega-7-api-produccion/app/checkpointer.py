"""LangGraph checkpoints in Redis: the graph's state survives restarts and waits for the human reviewer.

Two implementations, chosen with CHECKPOINTER:

- "redis" (default) → AsyncRedisCheckpointer, below. Uses only basic Redis commands (hashes, sets, expire),
  so it works with ANY Redis: Homebrew, Docker, a managed service or fakeredis in the tests.
- "langgraph-redis" → AsyncRedisSaver from the official `langgraph-checkpoint-redis` package. It needs Redis
  with the Search and JSON modules (Redis 8 in Docker, Redis Stack or Redis Cloud); a plain Redis answers
  "unknown command 'FT.INFO'".

AsyncRedisCheckpointer mirrors LangGraph's InMemorySaver (the reference implementation), with Redis instead of
Python dicts. Keys, for a thread (= one job) and a namespace (`""` for the main graph, one per subgraph):

    lg:ns:{thread}                          SET   namespaces used by the thread (to list and delete it)
    lg:cp:{thread}:{ns}                     HASH  checkpoint_id -> checkpoint + metadata + parent id
    lg:blobs:{thread}:{ns}                  HASH  [channel, version] -> channel value
    lg:writes:{thread}:{ns}:{checkpoint}    HASH  [task_id, idx] -> pending write of a task

Channel values are stored once per version ("blobs"), like InMemorySaver: a checkpoint that does not change a
channel points to the previous version instead of copying it again. Everything is serialized with the graph's
serializer (JsonPlusSerializer: messages, Pydantic models, dates...) and base64-encoded inside JSON.
"""

import base64
import json
import logging
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from redis import asyncio as aioredis

logger = logging.getLogger("app.checkpointer")

PREFIX = "lg"


def _encode(typed: tuple[str, bytes]) -> str:
    return json.dumps({"t": typed[0], "d": base64.b64encode(typed[1]).decode("ascii")})


def _decode(raw: str | bytes) -> tuple[str, bytes]:
    data = json.loads(raw)
    return data["t"], base64.b64decode(data["d"])


class AsyncRedisCheckpointer(BaseCheckpointSaver[str]):
    """Async LangGraph checkpointer on plain Redis. Use it with `ainvoke` / `astream` / `aget_state`."""

    def __init__(self, client: aioredis.Redis, ttl_seconds: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = client
        self.ttl_seconds = ttl_seconds   # None = keep forever; jobs waiting for approval need their state

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _ns_key(thread_id: str) -> str:
        return f"{PREFIX}:ns:{thread_id}"

    @staticmethod
    def _cp_key(thread_id: str, ns: str) -> str:
        return f"{PREFIX}:cp:{thread_id}:{ns}"

    @staticmethod
    def _blobs_key(thread_id: str, ns: str) -> str:
        return f"{PREFIX}:blobs:{thread_id}:{ns}"

    @staticmethod
    def _writes_key(thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{PREFIX}:writes:{thread_id}:{ns}:{checkpoint_id}"

    async def _touch(self, *keys: str) -> None:
        if self.ttl_seconds:
            async with self.client.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.expire(key, self.ttl_seconds)
                await pipe.execute()

    # ------------------------------------------------------------------ reading
    async def _load_blobs(self, thread_id: str, ns: str, versions: ChannelVersions) -> dict[str, Any]:
        if not versions:
            return {}
        fields = [json.dumps([channel, version]) for channel, version in versions.items()]
        raw_values = await self.client.hmget(self._blobs_key(thread_id, ns), fields)
        values: dict[str, Any] = {}
        for channel, raw in zip(versions, raw_values):
            if raw is None:
                continue
            typed = _decode(raw)
            if typed[0] != "empty":
                values[channel] = self.serde.loads_typed(typed)
        return values

    async def _load_writes(self, thread_id: str, ns: str, checkpoint_id: str) -> list[tuple[str, str, Any]]:
        stored = await self.client.hgetall(self._writes_key(thread_id, ns, checkpoint_id))
        entries = [json.loads(v) for v in stored.values()]
        # Same order as live execution: by task path, task id and index (writes_sort_key)
        entries.sort(key=lambda w: (w["task_path"], w["task_id"], w["idx"]))
        return [(w["task_id"], w["channel"], self.serde.loads_typed((w["t"], base64.b64decode(w["d"]))))
                for w in entries]

    async def _tuple(self, thread_id: str, ns: str, checkpoint_id: str, raw: str | bytes) -> CheckpointTuple:
        record = json.loads(raw)
        checkpoint: Checkpoint = self.serde.loads_typed(_decode(record["checkpoint"]))
        parent_id = record.get("parent_id")
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}},
            checkpoint={**checkpoint,
                        "channel_values": await self._load_blobs(thread_id, ns, checkpoint["channel_versions"])},
            metadata=self.serde.loads_typed(_decode(record["metadata"])),
            pending_writes=await self._load_writes(thread_id, ns, checkpoint_id),
            parent_config=({"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": parent_id}}
                           if parent_id else None),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """The checkpoint named in the config, or the latest one of the thread and namespace."""
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        key = self._cp_key(thread_id, ns)
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            ids = await self.client.hkeys(key)
            if not ids:
                return None
            checkpoint_id = max(i.decode() if isinstance(i, bytes) else i for i in ids)   # ids sort by time
        raw = await self.client.hget(key, checkpoint_id)
        return await self._tuple(thread_id, ns, checkpoint_id, raw) if raw is not None else None

    async def alist(self, config: RunnableConfig | None, *, filter: dict[str, Any] | None = None,
                    before: RunnableConfig | None = None, limit: int | None = None) -> AsyncIterator[CheckpointTuple]:
        """Checkpoints of a thread (or of every thread if config is None), newest first."""
        if config:
            thread_ids = [config["configurable"]["thread_id"]]
        else:
            thread_ids = [k.split(":", 2)[2] async for k in self._scan(f"{PREFIX}:ns:*")]
        wanted_ns = config["configurable"].get("checkpoint_ns") if config else None
        wanted_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        for thread_id in thread_ids:
            namespaces = [self._str(n) for n in await self.client.smembers(self._ns_key(thread_id))]
            for ns in sorted(namespaces):
                if wanted_ns is not None and ns != wanted_ns:
                    continue
                stored = await self.client.hgetall(self._cp_key(thread_id, ns))
                for checkpoint_id in sorted((self._str(i) for i in stored), reverse=True):
                    if (wanted_id and checkpoint_id != wanted_id) or (before_id and checkpoint_id >= before_id):
                        continue
                    raw = stored.get(checkpoint_id, stored.get(checkpoint_id.encode()))
                    item = await self._tuple(thread_id, ns, checkpoint_id, raw)
                    if filter and not all(item.metadata.get(k) == v for k, v in filter.items()):
                        continue
                    if limit is not None:
                        if limit <= 0:
                            return
                        limit -= 1
                    yield item

    # ------------------------------------------------------------------ writing
    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata,
                   new_versions: ChannelVersions) -> RunnableConfig:
        """Saves a checkpoint: its new channel values as blobs, the rest as one record."""
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        to_save = checkpoint.copy()
        values: dict[str, Any] = to_save.pop("channel_values")   # type: ignore[misc]
        blobs = {json.dumps([channel, version]): _encode(self.serde.dumps_typed(values[channel])
                                                         if channel in values else ("empty", b""))
                 for channel, version in new_versions.items()}
        record = json.dumps({
            "checkpoint": _encode(self.serde.dumps_typed(to_save)),
            "metadata": _encode(self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))),
            "parent_id": config["configurable"].get("checkpoint_id"),
        })
        async with self.client.pipeline(transaction=True) as pipe:
            if blobs:
                pipe.hset(self._blobs_key(thread_id, ns), mapping=blobs)
            pipe.hset(self._cp_key(thread_id, ns), checkpoint["id"], record)
            pipe.sadd(self._ns_key(thread_id), ns)
            await pipe.execute()
        await self._touch(self._ns_key(thread_id), self._cp_key(thread_id, ns), self._blobs_key(thread_id, ns))
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}}

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str,
                          task_path: str = "") -> None:
        """Saves the writes of a task (results, errors, interrupts) linked to the current checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        key = self._writes_key(thread_id, ns, config["configurable"]["checkpoint_id"])
        async with self.client.pipeline(transaction=True) as pipe:
            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                typed = self.serde.dumps_typed(value)
                entry = json.dumps({"task_id": task_id, "channel": channel, "idx": idx, "task_path": task_path,
                                    "t": typed[0], "d": base64.b64encode(typed[1]).decode("ascii")})
                field = json.dumps([task_id, idx])
                # Regular writes are written once (a retried task must not duplicate them); special writes
                # (errors, interrupts, resume values) have negative indexes and are overwritten
                if idx >= 0:
                    pipe.hsetnx(key, field, entry)
                else:
                    pipe.hset(key, field, entry)
            await pipe.execute()
        await self._touch(key)

    async def adelete_thread(self, thread_id: str) -> None:
        """Deletes every checkpoint, blob and write of a thread."""
        keys = [self._ns_key(thread_id)]
        for ns in [self._str(n) for n in await self.client.smembers(self._ns_key(thread_id))]:
            keys += [self._cp_key(thread_id, ns), self._blobs_key(thread_id, ns)]
        keys += [k async for k in self._scan(f"{PREFIX}:writes:{thread_id}:*")]
        if keys:
            await self.client.delete(*keys)

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        """Same versions as InMemorySaver: a zero-padded counter plus a random part (sortable as text)."""
        if current is None:
            number = 0
        elif isinstance(current, int):
            number = current
        else:
            number = int(current.split(".")[0])
        return f"{number + 1:032}.{random.random():016}"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _str(value: str | bytes) -> str:
        return value.decode() if isinstance(value, bytes) else value

    async def _scan(self, pattern: str) -> AsyncIterator[str]:
        async for key in self.client.scan_iter(match=pattern, count=200):
            yield self._str(key)


async def open_checkpointer(kind: str, redis_url: str, client: aioredis.Redis, stack: AsyncExitStack,
                            ttl_seconds: int | None = None) -> BaseCheckpointSaver:
    """Creates the checkpointer chosen in CHECKPOINTER. The stack closes it when the app or worker stops."""
    if kind == "langgraph-redis":
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver   # pip install langgraph-checkpoint-redis

        saver = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(redis_url))
        await saver.asetup()   # creates the search indexes the first time (needs the Search module)
        logger.info("checkpointer: AsyncRedisSaver (langgraph-checkpoint-redis)")
        return saver
    if kind != "redis":
        raise ValueError(f"CHECKPOINTER must be 'redis' or 'langgraph-redis', not {kind!r}")
    logger.info("checkpointer: AsyncRedisCheckpointer (plain Redis)")
    return AsyncRedisCheckpointer(client, ttl_seconds=ttl_seconds)
