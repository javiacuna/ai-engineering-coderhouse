"""Settings from environment variables (.env). Every value has a default that works on a laptop."""

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "si", "sí")


@dataclass(frozen=True)
class Settings:
    # Redis: job records, the queue and LangGraph checkpoints
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    checkpointer: str = field(default_factory=lambda: os.getenv("CHECKPOINTER", "redis"))
    checkpoint_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CHECKPOINT_TTL_SECONDS", 7 * 24 * 3600)))
    job_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("JOB_TTL_SECONDS", 7 * 24 * 3600)))

    # Worker
    run_worker_in_api: bool = field(default_factory=lambda: _bool("RUN_WORKER_IN_API", True))
    worker_concurrency: int = field(default_factory=lambda: int(os.getenv("WORKER_CONCURRENCY", 5)))
    job_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("JOB_TIMEOUT_SECONDS", 600)))

    # Observability (Arize Phoenix)
    phoenix_enabled: bool = field(default_factory=lambda: _bool("PHOENIX_ENABLED", True))
    phoenix_endpoint: str = field(default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    phoenix_project: str = field(default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "pe7-api-agente"))

    # Human-in-the-loop: a task is critical when its estimated cost reaches the threshold.
    # Cost model calibrated with the module 7 traces (gemini, paid-plan prices): ~US$ 0.0022 fixed per task
    # (supervisor + synthesis) plus ~US$ 0.0017 per product (research + analysis). 3 products ≈ US$ 0.0073.
    hitl_cost_threshold_usd: float = field(default_factory=lambda: float(os.getenv("HITL_COST_THRESHOLD_USD", 0.007)))
    cost_base_usd: float = field(default_factory=lambda: float(os.getenv("COST_BASE_USD", 0.0022)))
    cost_per_product_usd: float = field(default_factory=lambda: float(os.getenv("COST_PER_PRODUCT_USD", 0.0017)))


def load_settings() -> Settings:
    load_dotenv(find_dotenv(usecwd=True))   # the project's .env, or the repo's one (looked up in parent folders)
    return Settings()
