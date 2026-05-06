"""Minimal GuacaMol-inspired QED task using a fixed local candidate pool."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ...core import (
    CategoricalParam,
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
GUACAMOL_QED_TASK_NAME = "guacamol_qed_demo"
GUACAMOL_QED_DEFAULT_MAX_EVALUATIONS = 40
GUACAMOL_QED_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / GUACAMOL_QED_TASK_NAME
GUACAMOL_SOURCE_REPO_URL = "https://github.com/BenevolentAI/guacamol"

# This first integration keeps the task self-contained and offline-friendly by
# using a curated local pool rather than the full GuacaMol generator interface.
GUACAMOL_QED_SMILES_POOL: tuple[str, ...] = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN(CC)CC",
    "COc1ccccc1",
    "O=C(O)c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "CC1(C)C2CCC1(C)C(=O)C2",
    "CC(C)C1CCC(C)CC1O",
    "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",
    "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
    "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1",
    "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
    "COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1",
    "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",
    "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
)


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The GuacaMol QED task requires RDKit. Install it with "
            "`uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return Chem, Descriptors


@dataclass
class GuacamolQEDTaskConfig:
    """Configuration for one GuacaMol-inspired QED task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    description_dir: Path | None = None
    smiles_pool: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GuacamolQEDTask(Task):
    """Offline-friendly fixed-pool task using the GuacaMol QED objective."""

    def __init__(self, config: GuacamolQEDTaskConfig | None = None):
        self.config = config or GuacamolQEDTaskConfig()
        Chem, Descriptors = _require_rdkit()
        self._chem = Chem
        self._descriptors = Descriptors
        self._corrupt_score = -1.0
        self._smiles_pool = self._normalize_pool(self.config.smiles_pool or GUACAMOL_QED_SMILES_POOL)
        if not self._smiles_pool:
            raise ValueError("The GuacaMol QED task requires at least one candidate SMILES string.")

        self._scored_pool = [(smiles, self._score_smiles(smiles)) for smiles in self._smiles_pool]
        self._valid_pool = [(smiles, score) for smiles, score in self._scored_pool if score > self._corrupt_score]
        if not self._valid_pool:
            raise ValueError("The GuacaMol QED task candidate pool does not contain any valid RDKit molecules.")

        default_smiles, default_score = max(self._valid_pool, key=lambda item: item[1])
        search_space = SearchSpace(
            [
                CategoricalParam(
                    "SMILES",
                    choices=self._smiles_pool,
                    default=default_smiles,
                )
            ]
        )
        description_dir = self.config.description_dir or GUACAMOL_QED_DESCRIPTION_DIR
        self._dataset_summary = {
            "candidate_pool_size": int(len(self._smiles_pool)),
            "valid_candidate_count": int(len(self._valid_pool)),
            "default_smiles": default_smiles,
            "default_guacamol_qed_score": float(default_score),
        }
        self._spec = TaskSpec(
            name=GUACAMOL_QED_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("guacamol_qed_loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or GUACAMOL_QED_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(GUACAMOL_QED_TASK_NAME, description_dir),
            metadata={
                "display_name": "GuacaMol QED Demo",
                "source_repo": GUACAMOL_SOURCE_REPO_URL,
                "source_benchmark": "guacamol.standard_benchmarks.qed_benchmark",
                "candidate_pool_origin": "bundled_local_smiles_pool",
                "candidate_pool_size": len(self._smiles_pool),
                "dimension": 1,
                **self.config.metadata,
            },
        )

    @staticmethod
    def _normalize_pool(smiles_pool: Iterable[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for smiles in smiles_pool:
            canonical = str(smiles).strip()
            if not canonical or canonical in seen:
                continue
            normalized.append(canonical)
            seen.add(canonical)
        return tuple(normalized)

    def _score_smiles(self, smiles: str) -> float:
        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return self._corrupt_score
        return float(self._descriptors.qed(mol))

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def candidate_pool(self) -> tuple[str, ...]:
        return self._smiles_pool

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        smiles = str(config["SMILES"])
        score = self._score_smiles(smiles)
        loss = 1.0 - score
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"guacamol_qed_loss": loss},
            metrics={
                "guacamol_qed_score": score,
                "qed": score,
            },
            elapsed_seconds=elapsed,
            metadata={
                "smiles": smiles,
                "valid_smiles": score > self._corrupt_score,
                "score_source": "guacamol_qed_benchmark",
                "candidate_pool_size": len(self._smiles_pool),
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["guacamol_qed_loss"])):
                report.add_error("non_finite_score", "The GuacaMol QED task produced a non-finite loss.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("guacamol_qed_failed", f"The GuacaMol QED task could not score the default SMILES: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def create_guacamol_qed_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    description_dir: Path | None = None,
    smiles_pool: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuacamolQEDTask:
    return GuacamolQEDTask(
        GuacamolQEDTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            description_dir=description_dir,
            smiles_pool=smiles_pool,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "GUACAMOL_QED_DEFAULT_MAX_EVALUATIONS",
    "GUACAMOL_QED_DESCRIPTION_DIR",
    "GUACAMOL_QED_SMILES_POOL",
    "GUACAMOL_QED_TASK_NAME",
    "GUACAMOL_SOURCE_REPO_URL",
    "GuacamolQEDTask",
    "GuacamolQEDTaskConfig",
    "create_guacamol_qed_task",
]
