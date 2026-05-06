"""Unit tests for HTTP surrogate id mapping (no Docker)."""

from __future__ import annotations

from bbo.tasks.dbtune.catalog import SURROGATE_BENCHMARKS
from bbo.tasks.dbtune.http_surrogate_specs import (
    HTTP_SURROGATE_TASK_IDS,
    canonical_id_from_http_task_id,
    http_task_id_from_canonical,
    is_http_surrogate_task_id,
)


def test_http_surrogate_ids_cover_all_canonical() -> None:
    assert len(HTTP_SURROGATE_TASK_IDS) == len(SURROGATE_BENCHMARKS)
    for canonical in SURROGATE_BENCHMARKS:
        h = http_task_id_from_canonical(canonical)
        assert h.startswith("knob_http_surrogate_")
        assert is_http_surrogate_task_id(h)
        assert canonical_id_from_http_task_id(h) == canonical


def test_round_trip() -> None:
    ex = "knob_surrogate_sysbench_5"
    assert canonical_id_from_http_task_id("knob_http_surrogate_sysbench_5") == ex
