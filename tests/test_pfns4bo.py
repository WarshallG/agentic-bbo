from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import contextlib
import pytest

import bbo.run as run_module
import bbo.algorithms.model_based.pfns4bo as pfns4bo_module
from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.model_based.pfns4bo import Pfns4BoAlgorithm
from bbo.core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.run import build_arg_parser, run_single_experiment


@pytest.mark.unit
def test_pfns4bo_is_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "pfns4bo" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["pfns4bo"].family == "model_based"
    assert ALGORITHM_REGISTRY["pfns4bo"].numeric_only is False
    assert "pfns4bo" in algorithm_action.choices

    args = parser.parse_args(
        [
            "--algorithm",
            "pfns4bo",
            "--pfns-device",
            "cpu:0",
            "--pfns-pool-size",
            "128",
            "--pfns-model",
            "bnn",
        ]
    )
    assert args.algorithm == "pfns4bo"
    assert args.pfns_device == "cpu:0"
    assert args.pfns_pool_size == 128
    assert args.pfns_model == "bnn"


@pytest.mark.unit
def test_run_single_experiment_forwards_pfns_kwargs_without_runtime_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate_path = tmp_path / "surrogate.joblib"
    knobs_json_path = tmp_path / "knobs.json"
    captured: dict[str, object] = {}

    fake_task = SimpleNamespace(
        spec=SimpleNamespace(
            name="branin_demo",
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
        def __init__(self, *, task: object, algorithm: object, logger_backend: object, config: object) -> None:
            captured["experiment_task"] = task
            captured["experiment_algorithm"] = algorithm
            captured["experiment_config"] = config
            self.logger_backend = logger_backend

        def run(self) -> object:
            return SimpleNamespace(
                task_name="branin_demo",
                algorithm_name="pfns4bo",
                seed=11,
                n_completed=1,
                total_eval_time=0.25,
                best_primary_objective=1.23,
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
        task_name="branin_demo",
        algorithm_name="pfns4bo",
        seed=11,
        max_evaluations=5,
        results_root=tmp_path,
        noise_std=0.3,
        surrogate_path=surrogate_path,
        knobs_json_path=knobs_json_path,
        pfns_device="cpu:0",
        pfns_pool_size=64,
        pfns_model="bnn",
        generate_plots=False,
    )

    assert captured["task_name"] == "branin_demo"
    assert captured["task_kwargs"] == {
        "max_evaluations": 5,
        "seed": 11,
        "noise_std": 0.3,
        "surrogate_path": surrogate_path,
        "knobs_json_path": knobs_json_path,
    }
    assert captured["algorithm_name"] == "pfns4bo"
    assert captured["algorithm_kwargs"] == {
        "device": "cpu:0",
        "pool_size": 64,
        "model_name": "bnn",
    }
    assert summary["trial_count"] == 1
    assert "plot_paths" not in summary
    assert Path(summary["results_jsonl"]).with_name("summary.json").exists()


@pytest.mark.unit
def test_pfns4bo_continuous_backend_uses_onehot_converter_for_mixed_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def __init__(self, api_config: dict[str, object], model: object, **_: object) -> None:
            captured["api_config"] = api_config

        def observe(self, configs: list[dict[str, float]], values) -> None:
            captured.setdefault("observations", []).append((configs, values.tolist()))

        def suggest(self, count: int) -> list[dict[str, float]]:
            assert count == 1
            return [
                {
                    "lr": 0.02,
                    "depth": 5.0,
                    "activation::relu": 0.1,
                    "activation::gelu": 0.9,
                    "activation::tanh": 0.0,
                }
            ]

    task_spec = TaskSpec(
        name="mixed_pfns_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=6,
    )

    monkeypatch.setattr(pfns4bo_module, "select_pfns_device", lambda requested: "cpu:0")
    monkeypatch.setattr(
        pfns4bo_module,
        "resolve_pfns_model",
        lambda model_name: SimpleNamespace(
            model_name=model_name,
            model_path=Path("/tmp/fake_pfns_model.pt"),
            download_status="model_already_present",
            existed_before=True,
        ),
    )
    monkeypatch.setattr(pfns4bo_module, "load_torch_model", lambda path: object())
    monkeypatch.setattr(pfns4bo_module, "model_feature_capacity", lambda model: 32)
    monkeypatch.setattr(pfns4bo_module, "ContinuousPfnsOptimizer", FakeOptimizer)
    monkeypatch.setattr(pfns4bo_module, "deterministic_seed", lambda seed: contextlib.nullcontext())

    algorithm = Pfns4BoAlgorithm(device="cpu:0")
    algorithm.setup(task_spec, seed=11)
    algorithm.tell(
        TrialObservation.from_evaluation(
            TrialSuggestion(config={"lr": 0.01, "depth": 4, "activation": "relu"}, trial_id=0),
            EvaluationResult(objectives={"loss": 4.25}),
        )
    )

    suggestion = algorithm.ask()

    assert suggestion.config == {"lr": 0.02, "depth": 5, "activation": "gelu"}
    assert suggestion.metadata["pfns_categorical_to_continuous"] == "onehot"
    assert "activation::gelu" in captured["api_config"]
    observations = captured["observations"]
    observed_config = observations[0][0][0]
    assert observed_config["activation::relu"] == pytest.approx(1.0)
    assert observed_config["activation::gelu"] == pytest.approx(0.0)
