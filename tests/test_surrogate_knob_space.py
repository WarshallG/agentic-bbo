"""Tests for knob JSON decoding (no sklearn)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bbo.tasks.dbtune.knob_space import KnobSpaceFromJson

ASSETS = Path(__file__).resolve().parents[1] / "bbo" / "tasks" / "surrogate" / "assets"
KNOBS = ASSETS / "knobs_SYSBENCH_top5.json"
FEATURES = [
    "tmp_table_size",
    "max_heap_table_size",
    "query_prealloc_size",
    "innodb_thread_concurrency",
    "innodb_doublewrite",
]


@pytest.mark.unit
def test_decode_endpoints() -> None:
    if not KNOBS.is_file():
        pytest.skip("bundled knobs JSON missing")
    space = KnobSpaceFromJson(KNOBS, FEATURES)
    low = space.decode(np.zeros(5, dtype=np.float64))
    high = space.decode(np.ones(5, dtype=np.float64))
    assert low[0] == pytest.approx(1024.0)
    assert high[0] == pytest.approx(1073741824.0)
    assert low[4] in (0.0, 1.0)
    assert high[4] in (0.0, 1.0)
