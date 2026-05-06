"""Unit tests for SkyDiscover meta evaluator (combined_score semantics)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from bbo.algorithms.llm_based.skydiscover_bbo.meta_evaluator import evaluate


def _write_candidate(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "strategy.py"
    path.write_text(source, encoding="utf-8")
    return path


def _branin_parameter_specs() -> list[dict]:
    return [
        {"name": "x1", "type": "float", "low": -5.0, "high": 10.0, "log": False},
        {"name": "x2", "type": "float", "low": 0.0, "high": 15.0, "log": False},
    ]


@pytest.mark.unit
def test_meta_evaluator_contract_mode_default(tmp_path: Path) -> None:
    """Missing meta_combined_score_mode keeps legacy contract scoring."""
    ctx = tmp_path / "ctx.json"
    ctx.write_text(
        json.dumps(
            {
                "parameter_specs": [
                    {
                        "name": "x",
                        "type": "float",
                        "low": -1.0,
                        "high": 1.0,
                        "log": False,
                    }
                ],
                "objective_direction": "minimize",
                "recent_history": [],
                "seed": 0,
                "task_name": "minimal",
                "problem_key": "minimal",
                "description_fingerprint": "fp",
            }
        ),
        encoding="utf-8",
    )
    cand = _write_candidate(
        tmp_path,
        """
def suggest_next_config(**kwargs):
    return {"x": 0.0}
""",
    )
    os.environ["BBO_SKYDISCOVER_META_CONTEXT"] = str(ctx)
    try:
        out = evaluate(str(cand))
    finally:
        os.environ.pop("BBO_SKYDISCOVER_META_CONTEXT", None)

    assert out["combined_score"] == 10.0
    assert out.get("meta_trials_ok") == 3


@pytest.mark.unit
def test_meta_evaluator_distance_mode_near_branin_optimum(tmp_path: Path) -> None:
    cand = _write_candidate(
        tmp_path,
        """
def suggest_next_config(*, history, parameter_specs, objective_direction, seed, trial_index):
    return {"x1": -3.141592653589793, "x2": 12.275}
""",
    )
    ctx = tmp_path / "ctx.json"
    ctx.write_text(
        json.dumps(
            {
                "parameter_specs": _branin_parameter_specs(),
                "objective_direction": "minimize",
                "recent_history": [],
                "seed": 1,
                "task_name": "branin_demo",
                "problem_key": "branin_demo",
                "description_fingerprint": "fp",
                "meta_combined_score_mode": "distance_to_known_optimum",
                "known_optima": [[-3.141592653589793, 12.275], [3.141592653589793, 2.275]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.environ["BBO_SKYDISCOVER_META_CONTEXT"] = str(ctx)
    try:
        out = evaluate(str(cand))
    finally:
        os.environ.pop("BBO_SKYDISCOVER_META_CONTEXT", None)

    assert out["meta_mode"] == "distance_to_known_optimum"
    mean_d = float(out["mean_distance_to_known_optimum"])
    assert mean_d <= 1e-6
    assert abs(float(out["combined_score"]) + mean_d) <= 1e-9


@pytest.mark.unit
def test_meta_evaluator_distance_mode_far_point_scores_lower_than_optimum(tmp_path: Path) -> None:
    far = _write_candidate(
        tmp_path,
        """
def suggest_next_config(*, history, parameter_specs, objective_direction, seed, trial_index):
    return {"x1": -5.0, "x2": 0.0}
""",
    )
    ctx = tmp_path / "ctx.json"
    ctx.write_text(
        json.dumps(
            {
                "parameter_specs": _branin_parameter_specs(),
                "objective_direction": "minimize",
                "recent_history": [],
                "seed": 2,
                "task_name": "branin_demo",
                "problem_key": "branin_demo",
                "description_fingerprint": "fp",
                "meta_combined_score_mode": "distance_to_known_optimum",
                "known_optima": [[-math.pi, 12.275]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.environ["BBO_SKYDISCOVER_META_CONTEXT"] = str(ctx)
    try:
        far_out = evaluate(str(far))
    finally:
        os.environ.pop("BBO_SKYDISCOVER_META_CONTEXT", None)

    opt = tmp_path / "opt_strategy.py"
    opt.write_text(
        """import math


def suggest_next_config(*, history, parameter_specs, objective_direction, seed, trial_index):
    return {"x1": -math.pi, "x2": 12.275}
""",
        encoding="utf-8",
    )
    ctx2 = tmp_path / "ctx2.json"
    ctx2.write_text(ctx.read_text(encoding="utf-8"), encoding="utf-8")

    os.environ["BBO_SKYDISCOVER_META_CONTEXT"] = str(ctx2)
    try:
        opt_out = evaluate(str(opt))
    finally:
        os.environ.pop("BBO_SKYDISCOVER_META_CONTEXT", None)

    assert float(far_out["combined_score"]) < float(opt_out["combined_score"])


@pytest.mark.unit
def test_skydiscover_interleaved_writes_distance_meta_context(tmp_path: Path) -> None:
    """Synthetic tasks with known_optima should emit distance scoring mode for the meta evaluator."""
    from bbo.algorithms import SkydiscoverInterleavedAlgorithm
    from bbo.tasks import create_task

    task = create_task("branin_demo", max_evaluations=4, seed=0)
    algo = SkydiscoverInterleavedAlgorithm(
        run_dir=tmp_path,
        interleave_every=99,
        skydiscover_runner_enabled=False,
    )
    algo.setup(task.spec, seed=0, run_dir=tmp_path)
    algo._write_meta_context()
    data = json.loads((tmp_path / "generated" / "meta_context.json").read_text(encoding="utf-8"))
    assert data["meta_combined_score_mode"] == "distance_to_known_optimum"
    assert data["problem_key"] == "branin_demo"
    assert isinstance(data["known_optima"], list)


@pytest.mark.unit
def test_meta_evaluator_distance_invalid_trials_yield_strong_negative_score(tmp_path: Path) -> None:
    bad = _write_candidate(
        tmp_path,
        """
def suggest_next_config(*, history, parameter_specs, objective_direction, seed, trial_index):
    return {"wrong": 0}
""",
    )
    ctx = tmp_path / "ctx.json"
    ctx.write_text(
        json.dumps(
            {
                "parameter_specs": _branin_parameter_specs(),
                "objective_direction": "minimize",
                "recent_history": [],
                "seed": 3,
                "task_name": "branin_demo",
                "problem_key": "branin_demo",
                "description_fingerprint": "fp",
                "meta_combined_score_mode": "distance_to_known_optimum",
                "known_optima": [[0.0, 0.0]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.environ["BBO_SKYDISCOVER_META_CONTEXT"] = str(ctx)
    try:
        out = evaluate(str(bad))
    finally:
        os.environ.pop("BBO_SKYDISCOVER_META_CONTEXT", None)

    assert float(out["combined_score"]) < -1e8
    assert out.get("error") == "distance_mode_no_valid_trials"
