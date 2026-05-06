"""Task id mapping for dbtune surrogate-service tasks and canonical surrogate ids."""

from __future__ import annotations

from typing import Final

from .catalog import SURROGATE_BENCHMARKS

# --- Environment (host; mirror database HTTP naming) ---
_ENV_SURROGATE_BASE = "AGENTBBO_HTTP_SURROGATE_BASE_URL"
_ENV_SURROGATE_TIMEOUT = "AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC"
_DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8090"
_DEFAULT_TIMEOUT: Final[float] = 120.0

_HTTP_PREFIX: Final[str] = "knob_http_"


def canonical_id_from_http_task_id(http_task_id: str) -> str:
    """``knob_http_surrogate_sysbench_5`` -> ``knob_surrogate_sysbench_5``."""
    if not http_task_id.startswith(_HTTP_PREFIX):
        raise ValueError(f"Expected task id to start with {_HTTP_PREFIX!r}, got {http_task_id!r}")
    return f"knob_{http_task_id[len(_HTTP_PREFIX) :]}"


def http_task_id_from_canonical(canonical: str) -> str:
    """``knob_surrogate_sysbench_5`` -> ``knob_http_surrogate_sysbench_5``."""
    if not canonical.startswith("knob_"):
        raise ValueError(f"Expected canonical id to start with 'knob_', got {canonical!r}")
    return f"knob_http_{canonical[len('knob_') :]}"


def is_http_surrogate_task_id(task_id: str) -> bool:
    if not task_id.startswith("knob_http_surrogate_"):
        return False
    try:
        c = canonical_id_from_http_task_id(task_id)
    except ValueError:
        return False
    return c in SURROGATE_BENCHMARKS


# All registered HTTP-wrapped surrogate tasks (one per :data:`SURROGATE_BENCHMARKS` entry)
HTTP_SURROGATE_TASK_IDS: tuple[str, ...] = tuple(
    sorted(http_task_id_from_canonical(k) for k in SURROGATE_BENCHMARKS)
)
DBTUNE_SURROGATE_SERVICE_TASK_IDS: tuple[str, ...] = HTTP_SURROGATE_TASK_IDS


def is_dbtune_surrogate_service_task_id(task_id: str) -> bool:
    return is_http_surrogate_task_id(task_id)

__all__ = [
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "_DEFAULT_BASE_URL",
    "_DEFAULT_TIMEOUT",
    "_ENV_SURROGATE_BASE",
    "_ENV_SURROGATE_TIMEOUT",
    "canonical_id_from_http_task_id",
    "http_task_id_from_canonical",
    "is_dbtune_surrogate_service_task_id",
    "is_http_surrogate_task_id",
]
