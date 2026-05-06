from __future__ import annotations

from pathlib import Path

from bbo.core import TrialSuggestion
from bbo.run import run_single_experiment
from bbo.tasks.synthetic import BUDGETED_SPHERE_TASK_KEY, create_budgeted_sphere_task


def test_budgeted_sphere_consumes_suggestion_budget() -> None:
    task = create_budgeted_sphere_task()
    config = {"x1": 1.0, "x2": -0.5}

    low = task.evaluate(TrialSuggestion(config=config, budget=0.25))
    high = task.evaluate(TrialSuggestion(config=config, budget=1.0))

    assert task.spec.supports_budget is True
    assert low.metrics["evaluation_budget"] == 0.25
    assert high.metrics["evaluation_budget"] == 1.0
    assert float(low.objectives["loss"]) > float(high.objectives["loss"])
    assert float(high.metrics["fidelity_gap"]) == 0.0


def test_budgeted_sphere_runs_through_cli_helpers(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name=BUDGETED_SPHERE_TASK_KEY,
        algorithm_name="random_search",
        seed=7,
        max_evaluations=4,
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )

    assert summary["task_name"] == BUDGETED_SPHERE_TASK_KEY
    assert summary["trial_count"] == 4
    assert Path(summary["results_jsonl"]).exists()
