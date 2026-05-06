"""Molecule/QED scientific benchmark task using RDKit."""

from __future__ import annotations

import math
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from .data_assets import SOURCE_REPO_URL, DatasetAsset, stage_dataset_asset

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION_ROOT = PACKAGE_ROOT / "task_descriptions"
MOLECULE_DATASET_RELATIVE_PATH = "examples/Molecule/zinc.txt.gz"
MOLECULE_DATASET_FILENAME = "zinc.txt.gz"
MOLECULE_ARCHIVE_MEMBER = "zinc.txt"
MOLECULE_TASK_NAME = "molecule_qed_demo"
MOLECULE_DEFAULT_MAX_EVALUATIONS = 40
MOLECULE_SOURCE_PAPER = "Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"
MOLECULE_DESCRIPTION_DIR = TASK_DESCRIPTION_ROOT / MOLECULE_TASK_NAME


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "The molecule/QED task requires RDKit. Install it with `uv sync --extra dev --extra bo-tutorial` "
            "or provide a compatible conda environment."
        ) from exc
    return Chem, QED


def load_zinc_smiles(archive_path: Path) -> list[str]:
    """Read the tutorial's gzipped tar archive and return its SMILES list."""

    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile(MOLECULE_ARCHIVE_MEMBER)
        if member is None:
            raise FileNotFoundError(f"`{MOLECULE_ARCHIVE_MEMBER}` was not found in {archive_path}.")
        return member.read().decode("utf-8").splitlines()


@dataclass
class MoleculeTaskConfig:
    """Configuration for one molecule/QED benchmark task instance."""

    max_evaluations: int | None = None
    seed: int = 0
    source_root: Path | None = None
    cache_root: Path | None = None
    description_dir: Path | None = None
    smiles_limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MoleculeQEDTask(Task):
    """Task wrapper around the tutorial ZINC archive and RDKit QED objective."""

    def __init__(self, config: MoleculeTaskConfig | None = None):
        self.config = config or MoleculeTaskConfig()
        self._asset = stage_dataset_asset(
            MOLECULE_DATASET_RELATIVE_PATH,
            label="Molecule/QED",
            task_name=MOLECULE_TASK_NAME,
            source_root=self.config.source_root,
            cache_root=self.config.cache_root,
        )
        self._smiles_list = load_zinc_smiles(self._asset.cache_path)
        if self.config.smiles_limit is not None:
            limit = int(self.config.smiles_limit)
            if limit <= 0:
                raise ValueError("smiles_limit must be positive when provided.")
            self._smiles_list = self._smiles_list[:limit]
        if not self._smiles_list:
            raise ValueError("The molecule/QED dataset must contain at least one SMILES string.")

        Chem, QED = _require_rdkit()
        self._chem = Chem
        self._qed = QED
        self._first_valid_smiles = next((smiles for smiles in self._smiles_list if Chem.MolFromSmiles(smiles) is not None), None)

        default_smiles = self._first_valid_smiles or self._smiles_list[0]
        search_space = SearchSpace([CategoricalParam("SMILES", choices=tuple(self._smiles_list), default=default_smiles)])
        description_dir = self.config.description_dir or MOLECULE_DESCRIPTION_DIR
        self._dataset_summary = {
            **self._asset.as_metadata(),
            "item_count": int(len(self._smiles_list)),
            "smiles_limit": None if self.config.smiles_limit is None else int(self.config.smiles_limit),
            "archive_member": MOLECULE_ARCHIVE_MEMBER,
            "first_valid_smiles": self._first_valid_smiles,
        }
        self._spec = TaskSpec(
            name=MOLECULE_TASK_NAME,
            search_space=search_space,
            objectives=(ObjectiveSpec("qed_loss", ObjectiveDirection.MINIMIZE),),
            max_evaluations=self.config.max_evaluations or MOLECULE_DEFAULT_MAX_EVALUATIONS,
            description_ref=TaskDescriptionRef.from_directory(MOLECULE_TASK_NAME, description_dir),
            metadata={
                "display_name": "Molecule QED Demo",
                "source_paper": MOLECULE_SOURCE_PAPER,
                "source_repo": SOURCE_REPO_URL,
                "source_ref": self._asset.source_ref,
                "dataset_name": MOLECULE_DATASET_FILENAME,
                "dataset_cache_path": str(self._asset.cache_path),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "dimension": 1,
                **self.config.metadata,
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    @property
    def dataset_asset(self) -> DatasetAsset:
        return self._asset

    @property
    def dataset_summary(self) -> dict[str, Any]:
        return dict(self._dataset_summary)

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        start = time.perf_counter()
        config = self.spec.search_space.coerce_config(suggestion.config, use_defaults=False)
        smiles = str(config["SMILES"])
        molecule = self._chem.MolFromSmiles(smiles)
        qed = 0.0 if molecule is None else float(self._qed.qed(molecule))
        qed_loss = 1.0 - qed
        elapsed = time.perf_counter() - start
        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"qed_loss": qed_loss},
            metrics={"qed": qed},
            elapsed_seconds=elapsed,
            metadata={
                **self._asset.as_metadata(),
                "archive_member": MOLECULE_ARCHIVE_MEMBER,
                "smiles": smiles,
                "valid_smiles": molecule is not None,
            },
        )

    def sanity_check(self):
        report = super().sanity_check()
        if self._first_valid_smiles is None:
            report.add_error("no_valid_smiles", "The molecule/QED archive does not contain a valid SMILES string.")
        try:
            default_result = self.evaluate(TrialSuggestion(config=self.spec.search_space.defaults()))
            if not math.isfinite(float(default_result.objectives["qed_loss"])):
                report.add_error("non_finite_prediction", "The molecule/QED task produced a non-finite QED loss.")
        except Exception as exc:  # pragma: no cover - defensive guard.
            report.add_error("qed_evaluation_failed", f"The molecule/QED task could not score the default SMILES: {exc}")
        report.metadata.update(self._dataset_summary)
        return report


def create_molecule_qed_task(
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    source_root: Path | None = None,
    cache_root: Path | None = None,
    description_dir: Path | None = None,
    smiles_limit: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> MoleculeQEDTask:
    return MoleculeQEDTask(
        MoleculeTaskConfig(
            max_evaluations=max_evaluations,
            seed=seed,
            source_root=source_root,
            cache_root=cache_root,
            description_dir=description_dir,
            smiles_limit=smiles_limit,
            metadata=dict(metadata or {}),
        )
    )


__all__ = [
    "MOLECULE_ARCHIVE_MEMBER",
    "MOLECULE_DATASET_FILENAME",
    "MOLECULE_DATASET_RELATIVE_PATH",
    "MOLECULE_DEFAULT_MAX_EVALUATIONS",
    "MOLECULE_DESCRIPTION_DIR",
    "MOLECULE_TASK_NAME",
    "MoleculeQEDTask",
    "MoleculeTaskConfig",
    "create_molecule_qed_task",
    "load_zinc_smiles",
]
