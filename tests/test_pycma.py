from __future__ import annotations

from pathlib import Path

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
from bbo.algorithms import create_algorithm
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
from bbo.tasks import SyntheticFunctionTask, SyntheticFunctionTaskConfig


def test_pycma_runs_on_numeric_task(tmp_path: Path) -> None:
    task = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=14, seed=5))
    logger = JsonlMetricLogger(tmp_path / "pycma.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=create_algorithm("pycma", sigma_fraction=0.15, popsize=4),
        logger_backend=logger,
        config=ExperimentConfig(seed=5, resume=False, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    records = logger.load_records()

    assert summary.n_completed == 14
    assert len(records) == 14
    assert summary.incumbents
    assert summary.best_primary_objective is not None
    assert summary.best_primary_objective <= records[0].objectives[task.spec.primary_objective.name]


def test_pycma_runs_on_mixed_task_via_onehot_converter() -> None:
    task_spec = TaskSpec(
        name="mixed_pycma_demo",
        search_space=SearchSpace(
            [
                FloatParam("lr", low=1e-4, high=1e-1, log=True, default=1e-2),
                IntParam("depth", low=2, high=8, default=4),
                CategoricalParam("activation", choices=("relu", "gelu", "tanh"), default="relu"),
            ]
        ),
        objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
        max_evaluations=8,
    )
    algorithm = create_algorithm("pycma", sigma_fraction=0.15, popsize=4)
    algorithm.setup(task_spec, seed=5)

    seen_activations: set[str] = set()
    for trial_id in range(8):
        suggestion = algorithm.ask()
        seen_activations.add(str(suggestion.config["activation"]))
        loss = float(suggestion.config["lr"]) * 15.0 + float(suggestion.config["depth"])
        loss += {"relu": 0.25, "gelu": 0.1, "tanh": 0.4}[str(suggestion.config["activation"])]
        observation = TrialObservation.from_evaluation(
            TrialSuggestion(
                config=dict(suggestion.config),
                trial_id=trial_id,
                metadata=dict(suggestion.metadata),
            ),
            EvaluationResult(objectives={"loss": loss}),
        )
        algorithm.tell(observation)

    assert seen_activations
    assert seen_activations <= {"relu", "gelu", "tanh"}
    assert algorithm.incumbents()
