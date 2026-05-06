from __future__ import annotations

import json
import sys
import contextlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import bbo.run as run_module
import bbo.algorithms.model_based.pfns4bo_training as training_module
import bbo.algorithms.model_based.pfns4bo_variants as variants_module
from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.model_based.pfns4bo_encoding import EncodedCandidatePool
from bbo.algorithms.model_based.pfns4bo_variants import CustomPfnsBoAlgorithm, TabPfnV2BoAlgorithm
from bbo.core import (
    CategoricalParam,
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.run import build_arg_parser, run_single_experiment


class _FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


@pytest.mark.unit
def test_pfns_variants_are_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "pfns4bo_tabpfn_v2" in ALGORITHM_REGISTRY
    assert "pfns4bo_custom" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["pfns4bo_tabpfn_v2"].family == "model_based"
    assert ALGORITHM_REGISTRY["pfns4bo_custom"].family == "model_based"
    assert "pfns4bo_tabpfn_v2" in algorithm_action.choices
    assert "pfns4bo_custom" in algorithm_action.choices

    args = parser.parse_args(
        [
            "--algorithm",
            "pfns4bo_custom",
            "--pfns-custom-model-path",
            "/tmp/custom.pt",
            "--pfns-acquisition",
            "ucb",
            "--pfns-tabpfn-n-estimators",
            "4",
        ]
    )
    assert args.algorithm == "pfns4bo_custom"
    assert args.pfns_custom_model_path == "/tmp/custom.pt"
    assert args.pfns_acquisition == "ucb"
    assert args.pfns_tabpfn_n_estimators == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("algorithm_name", "extra_kwargs", "expected_kwargs"),
    [
        (
            "pfns4bo_tabpfn_v2",
            {
                "pfns_device": "cpu:0",
                "pfns_pool_size": 64,
                "pfns_acquisition": "ei",
                "pfns_tabpfn_n_estimators": 3,
                "pfns_tabpfn_ignore_pretraining_limits": False,
                "pfns_tabpfn_fit_mode": "fit_preprocessors",
            },
            {
                "device": "cpu:0",
                "pool_size": 64,
                "acquisition": "ei",
                "n_estimators": 3,
                "ignore_pretraining_limits": False,
                "fit_mode": "fit_preprocessors",
            },
        ),
        (
            "pfns4bo_custom",
            {
                "pfns_device": "cpu:0",
                "pfns_pool_size": 32,
                "pfns_acquisition": "mean",
                "pfns_custom_model_path": "/tmp/custom.pt",
            },
            {
                "device": "cpu:0",
                "pool_size": 32,
                "acquisition": "mean",
                "model_path": "/tmp/custom.pt",
            },
        ),
    ],
)
def test_run_single_experiment_forwards_pfns_variant_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm_name: str,
    extra_kwargs: dict[str, object],
    expected_kwargs: dict[str, object],
) -> None:
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
            self.logger_backend = logger_backend

        def run(self) -> object:
            return SimpleNamespace(
                task_name="branin_demo",
                algorithm_name=algorithm_name,
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
        algorithm_name=algorithm_name,
        seed=11,
        max_evaluations=5,
        results_root=tmp_path,
        noise_std=0.3,
        generate_plots=False,
        **extra_kwargs,
    )

    assert captured["algorithm_name"] == algorithm_name
    assert captured["algorithm_kwargs"] == expected_kwargs
    assert summary["trial_count"] == 1


@pytest.mark.unit
def test_tabpfn_variant_scores_candidate_pool_with_full_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    pool = EncodedCandidatePool(
        task_name="tabpfn_pool_demo",
        configs=(
            {"x": 0.1, "kind": "a"},
            {"x": 0.2, "kind": "b"},
            {"x": 0.3, "kind": "c"},
            {"x": 0.4, "kind": "d"},
        ),
        features=np.asarray(
            [
                [0.1, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 1.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 1.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        feature_names=("x", "kind::a", "kind::b", "kind::c", "kind::d"),
        candidate_metadata=({}, {}, {}, {}),
        full_candidate_count=4,
    )

    class FakeCriterion:
        def ei(self, logits: object, best_f: float, *, maximize: bool) -> _FakeTensor:
            captured["best_f"] = best_f
            captured["maximize"] = maximize
            return _FakeTensor([0.15, 0.9])

    class FakeRegressor:
        @classmethod
        def create_default_for_version(cls, version: object, **kwargs: object) -> FakeRegressor:
            captured["version"] = version
            captured["regressor_kwargs"] = kwargs
            return cls()

        def fit(self, X: object, y: np.ndarray) -> None:
            captured["fit_X"] = X
            captured["fit_y"] = y.tolist()

        def predict(self, X: object, *, output_type: str) -> dict[str, object]:
            captured["predict_X"] = X
            captured["output_type"] = output_type
            return {
                "mean": np.asarray([0.2, 0.8], dtype=float),
                "criterion": FakeCriterion(),
                "logits": object(),
            }

    fake_tabpfn = ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeRegressor
    fake_constants = ModuleType("tabpfn.constants")
    fake_constants.ModelVersion = SimpleNamespace(V2="v2")

    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", fake_constants)
    monkeypatch.setattr(variants_module, "build_pool_candidates", lambda task_spec, seed, pool_size: pool)
    monkeypatch.setattr(variants_module, "_configs_to_dataframe", lambda configs, search_space: list(configs))
    monkeypatch.setattr(variants_module, "select_pfns_device", lambda requested: "cpu:0")

    task_spec = TaskSpec(
        name="tabpfn_pool_demo",
        search_space=SearchSpace(
            [
                FloatParam("x", low=0.0, high=1.0, default=0.2),
                CategoricalParam("kind", choices=("a", "b", "c", "d"), default="a"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=6,
    )

    algorithm = TabPfnV2BoAlgorithm(device="cpu:0", pool_size=4, acquisition="ei", n_estimators=2)
    algorithm.setup(task_spec, seed=3)
    algorithm.tell(
        TrialObservation.from_evaluation(
            TrialSuggestion(config={"x": 0.1, "kind": "a"}, metadata={"pfns_pool_index": 0}, trial_id=0),
            EvaluationResult(objectives={"loss": 5.0}),
        )
    )
    algorithm.tell(
        TrialObservation.from_evaluation(
            TrialSuggestion(config={"x": 0.2, "kind": "b"}, metadata={"pfns_pool_index": 1}, trial_id=1),
            EvaluationResult(objectives={"loss": 2.0}),
        )
    )

    suggestion = algorithm.ask()

    assert suggestion.config == {"x": 0.4, "kind": "d"}
    assert suggestion.metadata["pfns_variant"] == "tabpfn_v2"
    assert captured["output_type"] == "full"
    assert captured["fit_y"] == [0.0, 1.0]
    assert captured["best_f"] == pytest.approx(1.0)


@pytest.mark.unit
def test_custom_pfns_variant_uses_transformer_pool_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    model_path = tmp_path / "custom.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({"prior": "hebo", "max_features": 16}), encoding="utf-8")

    pool = EncodedCandidatePool(
        task_name="custom_pool_demo",
        configs=(
            {"x": 0.1},
            {"x": 0.2},
            {"x": 0.3},
            {"x": 0.4},
        ),
        features=np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=float),
        feature_names=("x",),
        candidate_metadata=({}, {}, {}, {}),
        full_candidate_count=4,
    )

    class FakeTransformerBOMethod:
        def __init__(self, model: object, *, device: str) -> None:
            captured["selector_model"] = model
            captured["selector_device"] = device

        def observe_and_suggest(self, observed_matrix: np.ndarray, utilities: np.ndarray, pending_matrix: np.ndarray) -> int:
            captured["observed_matrix"] = observed_matrix.tolist()
            captured["utilities"] = utilities.tolist()
            captured["pending_matrix"] = pending_matrix.tolist()
            return 1

    fake_pfns4bo = ModuleType("pfns4bo")
    fake_scripts = ModuleType("pfns4bo.scripts")
    fake_acq = ModuleType("pfns4bo.scripts.acquisition_functions")
    fake_acq.TransformerBOMethod = FakeTransformerBOMethod

    monkeypatch.setitem(sys.modules, "pfns4bo", fake_pfns4bo)
    monkeypatch.setitem(sys.modules, "pfns4bo.scripts", fake_scripts)
    monkeypatch.setitem(sys.modules, "pfns4bo.scripts.acquisition_functions", fake_acq)
    monkeypatch.setattr(variants_module, "build_pool_candidates", lambda task_spec, seed, pool_size: pool)
    monkeypatch.setattr(variants_module, "select_pfns_device", lambda requested: "cpu:0")
    monkeypatch.setattr(variants_module, "load_torch_model", lambda path: {"loaded_from": str(path)})
    monkeypatch.setattr(variants_module, "model_feature_capacity", lambda model: 8)
    monkeypatch.setattr(variants_module, "require_pfns4bo", lambda: fake_pfns4bo)
    monkeypatch.setattr(variants_module, "deterministic_seed", lambda seed: contextlib.nullcontext())

    task_spec = TaskSpec(
        name="custom_pool_demo",
        search_space=SearchSpace([FloatParam("x", low=0.0, high=1.0, default=0.2)]),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=6,
    )

    algorithm = CustomPfnsBoAlgorithm(model_path=model_path, device="cpu:0", pool_size=4, acquisition="ei")
    algorithm.setup(task_spec, seed=5)
    algorithm.tell(
        TrialObservation.from_evaluation(
            TrialSuggestion(config={"x": 0.1}, metadata={"pfns_pool_index": 0}, trial_id=0),
            EvaluationResult(objectives={"loss": 5.0}),
        )
    )
    algorithm.tell(
        TrialObservation.from_evaluation(
            TrialSuggestion(config={"x": 0.2}, metadata={"pfns_pool_index": 1}, trial_id=1),
            EvaluationResult(objectives={"loss": 2.0}),
        )
    )

    suggestion = algorithm.ask()

    assert suggestion.config == {"x": 0.4}
    assert suggestion.metadata["pfns_variant"] == "custom_pfn"
    assert suggestion.metadata["pfns_training_prior"] == "hebo"
    assert captured["utilities"] == [0.0, 1.0]


@pytest.mark.unit
def test_build_custom_pfn_training_recipe_routes_by_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_module, "_load_pfns4bo_training_modules", lambda: {"fake": True})
    monkeypatch.setattr(training_module, "_build_hebo_recipe", lambda **kwargs: {"recipe": "hebo", **kwargs})
    monkeypatch.setattr(training_module, "_build_gp_recipe", lambda **kwargs: {"recipe": "gp", **kwargs})

    hebo = training_module.build_custom_pfn_training_recipe(
        prior="hebo",
        max_features=8,
        epochs=1,
        steps_per_epoch=2,
        batch_size=4,
        seq_len=16,
        emsize=32,
        nhid=64,
        nlayers=2,
        nhead=2,
        lr=1e-4,
        train_mixed_precision=False,
    )
    gp = training_module.build_custom_pfn_training_recipe(
        prior="gp",
        max_features=8,
        epochs=1,
        steps_per_epoch=2,
        batch_size=4,
        seq_len=16,
        emsize=32,
        nhid=64,
        nlayers=2,
        nhead=2,
        lr=1e-4,
        train_mixed_precision=False,
    )

    assert hebo["recipe"] == "hebo"
    assert gp["recipe"] == "gp"


@pytest.mark.unit
def test_train_custom_pfn_model_persists_checkpoint_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: dict[str, object] = {}

    class FakeTorch:
        @staticmethod
        def save(obj: object, path: Path) -> None:
            saved["model"] = obj
            Path(path).write_text("saved", encoding="utf-8")

    fake_pfns4bo = SimpleNamespace(
        train=SimpleNamespace(train=lambda **kwargs: (0.0, [], {"trained": True}, None))
    )

    monkeypatch.setattr(
        training_module,
        "_load_pfns4bo_training_modules",
        lambda: {"train": fake_pfns4bo.train, "bar_distribution": object(), "encoders": object(), "priors": object(), "utils": object()},
    )
    monkeypatch.setattr(training_module, "require_torch", lambda: FakeTorch)
    monkeypatch.setattr(training_module, "select_pfns_device", lambda requested: "cpu:0")
    monkeypatch.setattr(
        training_module,
        "build_custom_pfn_training_recipe",
        lambda **kwargs: {
            "priordataloader_class": "fake_dl",
            "extra_prior_kwargs_dict": {"num_features": kwargs["max_features"]},
            "batch_size": kwargs["batch_size"],
            "seq_len": kwargs["seq_len"],
        },
    )
    monkeypatch.setattr(training_module, "build_custom_pfn_criterion", lambda recipe, device, num_buckets: "criterion")

    artifact = training_module.train_custom_pfn_model(
        output_dir=tmp_path,
        prior="hebo",
        max_features=12,
        model_name="model.pt",
        epochs=2,
        steps_per_epoch=3,
        batch_size=4,
        seq_len=20,
    )

    assert artifact.model_path.exists()
    assert artifact.metadata_path.exists()
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["prior"] == "hebo"
    assert metadata["max_features"] == 12
    assert metadata["epochs"] == 2
    assert metadata["steps_per_epoch"] == 3
    assert metadata["seq_len"] == 20
    assert saved["model"] == {"trained": True}
