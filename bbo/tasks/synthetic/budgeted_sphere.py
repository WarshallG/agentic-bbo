"""Budget-aware synthetic benchmark definition."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core import (
    EvaluationResult,
    FloatParam,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)
from .base import TASK_DESCRIPTION_ROOT

BUDGETED_SPHERE_TASK_KEY = "budgeted_sphere_demo"


def _budgeted_sphere_search_space() -> SearchSpace:
    return SearchSpace(
        [
            FloatParam("x1", low=-5.0, high=5.0, default=2.0),
            FloatParam("x2", low=-5.0, high=5.0, default=-1.5),
        ]
    )


@dataclass(frozen=True)
class BudgetedSphereTaskConfig:
    """Configuration for one budget-aware synthetic task instance."""

    problem: str = BUDGETED_SPHERE_TASK_KEY
    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BudgetedSphereTask(Task):
    """Synthetic sphere task whose fidelity depends on ``suggestion.budget``."""

    def __init__(self, config: BudgetedSphereTaskConfig) -> None:
        self.config = config
        description_dir = config.description_dir or (TASK_DESCRIPTION_ROOT / BUDGETED_SPHERE_TASK_KEY)
        search_space = _budgeted_sphere_search_space()
        self._spec = TaskSpec(
            name=BUDGETED_SPHERE_TASK_KEY,
            search_space=search_space,
            objectives=(ObjectiveSpec("loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=config.max_evaluations or 30,
            default_budget=1.0,
            budget_range=(0.25, 1.0),
            supports_budget=True,
            description_ref=TaskDescriptionRef.from_directory(BUDGETED_SPHERE_TASK_KEY, description_dir),
            metadata={
                "problem_key": BUDGETED_SPHERE_TASK_KEY,
                "display_name": "Budgeted Sphere (2D)",
                "dimension": 2.0,
                "known_optimum": 0.0,
                "known_optima": [[0.0, 0.0]],
                "plot_resolution": 180,
                "cma_initial_config": search_space.defaults(),
                **config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        budget = suggestion.budget if suggestion.budget is not None else self.spec.default_budget
        if budget is None:  # pragma: no cover - guarded by TaskSpec/Experimenter.
            raise ValueError("BudgetedSphereTask requires a resolved evaluation budget.")
        if self.spec.budget_range is not None:
            low, high = self.spec.budget_range
            if not (low <= budget <= high):
                raise ValueError(f"Budget {budget} is outside {self.spec.budget_range!r}.")

        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        vector = self.spec.search_space.to_numeric_vector(config)
        true_loss = float(np.dot(vector, vector))
        proxy_term = float(np.mean(np.sin(vector) ** 2))
        fidelity_gap = (1.0 / float(budget) - 1.0) * 0.25 * proxy_term
        observed_loss = true_loss + fidelity_gap

        elapsed = time.perf_counter() - start
        metrics = {
            "dimension": 2.0,
            "evaluation_budget": float(budget),
            "true_loss": true_loss,
            "fidelity_gap": fidelity_gap,
            "regret": observed_loss,
        }
        for name, scalar in zip(self.spec.search_space.names(), vector, strict=True):
            metrics[f"coord::{name}"] = float(scalar)

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"loss": observed_loss},
            metrics=metrics,
            elapsed_seconds=elapsed,
            metadata={
                "problem_key": BUDGETED_SPHERE_TASK_KEY,
                "display_name": "Budgeted Sphere (2D)",
                "evaluation_budget": float(budget),
            },
        )

    def surface_grid(self, *, resolution: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bounds = self.spec.search_space.numeric_bounds()
        resolution = resolution or int(self.spec.metadata.get("plot_resolution", 180))
        xs = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
        ys = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.square(xx) + np.square(yy)
        return xx, yy, zz


def create_budgeted_sphere_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> BudgetedSphereTask:
    return BudgetedSphereTask(
        BudgetedSphereTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "BUDGETED_SPHERE_TASK_KEY",
    "BudgetedSphereTask",
    "BudgetedSphereTaskConfig",
    "create_budgeted_sphere_task",
]
