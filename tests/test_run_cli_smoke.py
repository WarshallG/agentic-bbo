"""Smoke tests for ``bbo.run`` CLI helpers (no plotting)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bbo.run as run_module
from bbo.run import run_single_experiment


@pytest.mark.unit
def test_run_single_experiment_writes_jsonl_and_summary(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="random_search",
        seed=3,
        max_evaluations=10,
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
    )
    results_jsonl = Path(summary["results_jsonl"])
    assert results_jsonl.exists()
    assert results_jsonl.stat().st_size > 0
    summary_path = results_jsonl.parent / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["trial_count"] == 10
    assert "plot_paths" not in data and "plot_paths" not in summary


@pytest.mark.unit
def test_run_single_experiment_forwards_task_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = object()
    surrogate_path = tmp_path / "surrogate.joblib"
    knobs_json_path = tmp_path / "knobs.json"
    captured: dict[str, object] = {}
    fake_task = SimpleNamespace(
        spec=SimpleNamespace(
            name="bboplace_bench",
            search_space=SimpleNamespace(numeric_bounds=lambda: None),
        )
    )

    def fake_create_task(task_name: str, **kwargs: object) -> object:
        captured["task_name"] = task_name
        captured["task_kwargs"] = kwargs
        return fake_task

    def fake_create_algorithm(name: str, **kwargs: object) -> object:
        captured["algorithm_name"] = name
        captured["algorithm_kwargs"] = kwargs
        return SimpleNamespace(name=name)

    class FakeLogger:
        def __init__(self, path: Path) -> None:
            self.path = path

        def load_records(self) -> list[dict[str, int]]:
            return [{"trial_id": 0}]

    class FakeExperimenter:
        def __init__(
            self,
            *,
            task: object,
            algorithm: object,
            logger_backend: object,
            config: object,
        ) -> None:
            self.logger_backend = logger_backend

        def run(self) -> object:
            return SimpleNamespace(
                task_name="bboplace_bench",
                algorithm_name="random_search",
                seed=5,
                n_completed=1,
                total_eval_time=0.1,
                best_primary_objective=1.0,
                stop_reason="synthetic_stop",
                description_fingerprint="fake-fingerprint",
                incumbents=[],
                logger_summary={"records_written": 1},
            )

    monkeypatch.setattr(run_module, "create_task", fake_create_task)
    monkeypatch.setattr(run_module, "create_algorithm", fake_create_algorithm)
    monkeypatch.setattr(run_module, "JsonlMetricLogger", FakeLogger)
    monkeypatch.setattr(run_module, "Experimenter", FakeExperimenter)

    summary = run_single_experiment(
        task_name="bboplace_bench",
        algorithm_name="random_search",
        seed=5,
        max_evaluations=2,
        results_root=tmp_path,
        generate_plots=False,
        surrogate_path=surrogate_path,
        knobs_json_path=knobs_json_path,
        task_kwargs={"definition": definition},
    )

    assert captured["task_name"] == "bboplace_bench"
    assert captured["task_kwargs"] == {
        "max_evaluations": 2,
        "seed": 5,
        "noise_std": 0.0,
        "definition": definition,
        "surrogate_path": surrogate_path,
        "knobs_json_path": knobs_json_path,
    }
    assert summary["trial_count"] == 1
