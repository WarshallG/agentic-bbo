"""Resolve bundled surrogate assets (``bbo/tasks/dbtune/assets/``)."""

from __future__ import annotations

from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"

# Top-5 SYSBENCH iteration order; matches `knobs_SYSBENCH_top5.json` in assets.
SYSBENCH_5_FEATURE_ORDER: tuple[str, ...] = (
    "tmp_table_size",
    "max_heap_table_size",
    "query_prealloc_size",
    "innodb_thread_concurrency",
    "innodb_doublewrite",
)


def bundled_knobs_top5_path() -> Path:
    """Default knobs JSON (top-5 Sysbench knobs)."""
    return _ASSETS / "knobs_SYSBENCH_top5.json"


def bundled_surrogate_sysbench5_path() -> Path:
    """Default joblib for Sysbench 5-knob (delegates to ``catalog.resolve_bundled_joblib_path``)."""
    from .catalog import SURROGATE_BENCHMARKS, resolve_bundled_joblib_path

    return resolve_bundled_joblib_path(SURROGATE_BENCHMARKS["knob_surrogate_sysbench_5"])


__all__ = [
    "SYSBENCH_5_FEATURE_ORDER",
    "bundled_knobs_top5_path",
    "bundled_surrogate_sysbench5_path",
]
