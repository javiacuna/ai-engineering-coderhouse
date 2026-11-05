"""Creates the Pinecone Serverless index if it does not exist, and checks it matches the configuration.

Usage:
    python setup_index.py
"""

import logging
import sys
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors import PineconeError

from config import Settings

logger = logging.getLogger("rag.setup")


class IndexConfigError(Exception):
    """The existing index does not match the configured dimension or metric."""


def ensure_index(pc: Pinecone, settings: Settings, timeout: float = 120):
    """Returns the index, creating it (serverless) if needed and waiting until it is ready."""
    if settings.index_name not in pc.list_indexes().names():
        logger.info("Creating serverless index '%s' (dimension=%d, metric=%s, %s/%s)", settings.index_name,
                    settings.dimension, settings.metric, settings.cloud, settings.region)
        pc.create_index(
            name=settings.index_name,
            dimension=settings.dimension,
            metric=settings.metric,
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
    else:
        logger.info("Index '%s' already exists", settings.index_name)

    deadline = time.monotonic() + timeout
    description = pc.describe_index(settings.index_name)
    while not description.status.ready:
        if time.monotonic() > deadline:
            raise TimeoutError(f"index '{settings.index_name}' was not ready after {timeout:.0f} s")
        time.sleep(2)
        description = pc.describe_index(settings.index_name)

    # Dimension mismatch is the most common error: check it before uploading anything
    metric = str(getattr(description.metric, "value", description.metric)).lower()
    if description.dimension != settings.dimension or metric != settings.metric:
        raise IndexConfigError(
            f"index '{settings.index_name}' has dimension={description.dimension}, metric={metric}; "
            f"the configuration expects dimension={settings.dimension}, metric={settings.metric}. "
            "Use another INDEX_NAME or delete the index in the Pinecone console."
        )
    return pc.Index(settings.index_name)


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        settings = Settings.from_env()
        index = ensure_index(Pinecone(api_key=settings.pinecone_api_key.get_secret_value()), settings)
        stats = index.describe_index_stats()
    except (ValueError, IndexConfigError, TimeoutError, PineconeError) as e:
        logger.error("%s", e)
        return 1
    print(f"\nIndex '{settings.index_name}' ready: dimension={stats.dimension}, metric={settings.metric}, "
          f"vectors={stats.total_vector_count}")
    for name, summary in stats.namespaces.items():
        print(f"  namespace '{name}': {summary.vector_count} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
