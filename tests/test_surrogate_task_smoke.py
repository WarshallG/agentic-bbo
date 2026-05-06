"""Smoke test for surrogate task (optional sklearn/joblib)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")


@pytest.mark.integration
def test_sysbench5_surrogate_evaluate() -> None:
    from bbo.core import TrialSuggestion
    from bbo.tasks import create_surrogate_task
    from bbo.tasks.dbtune.paths import bundled_surrogate_sysbench5_path

    p = bundled_surrogate_sysbench5_path()
    if not p.is_file():
        pytest.skip(
            "No surrogate .joblib found. Download RF_SYSBENCH_5knob.joblib "
            "(see bbo/tasks/dbtune/assets/README.md) into bbo/tasks/dbtune/assets/, "
            "or set AGENTIC_BBO_SYSBENCH5_SURROGATE."
        )

    task = create_surrogate_task("knob_surrogate_sysbench_5", max_evaluations=3, seed=0)
    report = task.sanity_check()
    if not report.ok:
        pytest.skip(
            "sanity check failed: " + "; ".join(e.message for e in report.errors)
        )
    cfg = task.spec.search_space.defaults()
    res = task.evaluate(TrialSuggestion(config=cfg, trial_id=0))
    assert res.success
    assert "throughput" in res.objectives
    y0 = float(res.objectives["throughput"])
    assert math.isfinite(y0)


@pytest.mark.integration
def test_sysbench5_two_distinct_configs_finite() -> None:
    """Evaluate two corner-like configs; real RF should return finite throughput."""
    from bbo.core import TrialSuggestion
    from bbo.tasks import create_surrogate_task
    from bbo.tasks.dbtune.paths import bundled_surrogate_sysbench5_path

    if not bundled_surrogate_sysbench5_path().is_file():
        pytest.skip("surrogate .joblib missing — see bbo/tasks/dbtune/assets/README.md")

    task = create_surrogate_task("knob_surrogate_sysbench_5", max_evaluations=5, seed=0)
    report = task.sanity_check()
    if not report.ok:
        pytest.skip("; ".join(e.message for e in report.errors))

    names = task.spec.search_space.names()
    a = {k: 0.1 for k in names}
    b = {k: 0.9 for k in names}
    ya = float(task.evaluate(TrialSuggestion(config=a, trial_id=0)).objectives["throughput"])
    yb = float(task.evaluate(TrialSuggestion(config=b, trial_id=1)).objectives["throughput"])
    assert math.isfinite(ya) and math.isfinite(yb)
