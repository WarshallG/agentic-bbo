from __future__ import annotations

from pathlib import Path

from bbo.algorithms import RandomSearchAlgorithm
from bbo.core import (
    Algorithm,
    EvaluationResult,
    ExperimentConfig,
    Experimenter,
    FloatParam,
    Incumbent,
    JsonlMetricLogger,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from bbo.tasks import SyntheticFunctionTask, SyntheticFunctionTaskConfig


class _SingleSuggestionAlgorithm(Algorithm):
    def __init__(self, suggestion: TrialSuggestion) -> None:
        self._suggestion = suggestion
        self._setup = False
        self._observations: list[TrialObservation] = []

    @property
    def name(self) -> str:
        return "single_suggestion"

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs) -> None:
        self._setup = True

    def ask(self) -> TrialSuggestion:
        if not self._setup:
            raise RuntimeError("setup() must run first")
        return TrialSuggestion(
            config=dict(self._suggestion.config),
            trial_id=self._suggestion.trial_id,
            budget=self._suggestion.budget,
            metadata=dict(self._suggestion.metadata),
        )

    def tell(self, observation: TrialObservation) -> None:
        self._observations.append(observation)

    def incumbents(self) -> list[Incumbent]:
        return []


class _StrictNumericTask(Task):
    def __init__(self) -> None:
        self._spec = TaskSpec(
            name="strict_numeric_demo",
            search_space=SearchSpace([FloatParam("x", low=-1.0, high=1.0, default=0.0)]),
            objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=1,
            description_ref=None,
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        value = float(suggestion.config["x"]) ** 2
        return EvaluationResult(objectives={"loss": value})


def test_experimenter_resume_keeps_append_only_history(tmp_path: Path) -> None:
    task = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=6, seed=11))
    logger = JsonlMetricLogger(tmp_path / "trials.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=RandomSearchAlgorithm(),
        logger_backend=logger,
        config=ExperimentConfig(seed=11, resume=False, fail_fast_on_sanity=True),
    )
    summary = experiment.run()
    assert summary.n_completed == 6
    assert len(logger.load_records()) == 6

    task_resume = SyntheticFunctionTask(SyntheticFunctionTaskConfig(problem="sphere_demo", max_evaluations=6, seed=11))
    resume_experiment = Experimenter(
        task=task_resume,
        algorithm=RandomSearchAlgorithm(),
        logger_backend=logger,
        config=ExperimentConfig(seed=11, resume=True, fail_fast_on_sanity=True),
    )
    resumed = resume_experiment.run()
    assert resumed.n_completed == 6
    assert len(logger.load_records()) == 6


def test_experimenter_marks_missing_parameters_invalid(tmp_path: Path) -> None:
    task = _StrictNumericTask()
    logger = JsonlMetricLogger(tmp_path / "invalid_missing.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=_SingleSuggestionAlgorithm(TrialSuggestion(config={})),
        logger_backend=logger,
        config=ExperimentConfig(seed=0, resume=False, fail_fast_on_sanity=False),
    )

    summary = experiment.run()
    record = logger.load_records()[0]

    assert summary.n_completed == 1
    assert record.status == "invalid"
    assert "Missing required parameters" in str(record.error_message)


def test_experimenter_rejects_budget_for_tasks_without_budget_support(tmp_path: Path) -> None:
    task = _StrictNumericTask()
    logger = JsonlMetricLogger(tmp_path / "invalid_budget.jsonl")
    experiment = Experimenter(
        task=task,
        algorithm=_SingleSuggestionAlgorithm(TrialSuggestion(config={"x": 0.5}, budget=0.5)),
        logger_backend=logger,
        config=ExperimentConfig(seed=0, resume=False, fail_fast_on_sanity=False),
    )

    summary = experiment.run()
    record = logger.load_records()[0]

    assert summary.n_completed == 1
    assert record.status == "invalid"
    assert "does not accept suggestion budgets" in str(record.error_message)
