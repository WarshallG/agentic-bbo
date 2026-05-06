"""Candidate-pool encoders for PFNs4BO-style surrogate backends."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ...core import CategoricalParam, FloatParam, IntParam, SearchSpace, TaskSpec
from ...tasks.scientific.molecule import MOLECULE_TASK_NAME
from ...tasks.scientific.oer import (
    OER_CATEGORICAL_FEATURES,
    OER_FLOAT_FEATURES,
    OER_INTEGER_FEATURES,
    OER_NUMERICAL_FEATURES,
    OER_TASK_NAME,
)
from .pfns4bo_utils import config_identity

MOLECULE_DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
)


@dataclass(frozen=True)
class EncodedCandidatePool:
    """Candidate configs and their encoded feature matrix for PFNs4BO's pool-based interface."""

    task_name: str
    configs: tuple[dict[str, Any], ...]
    features: np.ndarray
    feature_names: tuple[str, ...]
    candidate_metadata: tuple[dict[str, Any], ...]
    full_candidate_count: int


def build_pool_candidates(task_spec: TaskSpec, *, seed: int, pool_size: int) -> EncodedCandidatePool:
    if task_spec.name == OER_TASK_NAME:
        return build_oer_candidate_pool(task_spec.search_space, seed=seed, pool_size=pool_size)
    if task_spec.name == MOLECULE_TASK_NAME:
        return build_molecule_candidate_pool(task_spec.search_space, seed=seed, pool_size=pool_size)
    return build_generic_candidate_pool(task_spec.search_space, seed=seed, pool_size=pool_size, task_name=task_spec.name)


def generic_feature_names(search_space: SearchSpace) -> tuple[str, ...]:
    names: list[str] = []
    for param in search_space:
        if isinstance(param, (FloatParam, IntParam)):
            names.append(param.name)
            continue
        if not isinstance(param, CategoricalParam):
            raise TypeError(
                f"Generic PFNs candidate encoding does not support `{type(param).__name__}` for `{param.name}`."
            )
        names.extend(f"{param.name}::{choice}" for choice in param.choices)
    return tuple(names)


def _normalize_numeric_value(value: float, *, low: float, high: float, log: bool) -> float:
    if log:
        low = float(np.log(low))
        high = float(np.log(high))
        value = float(np.log(value))
    span = high - low
    if abs(span) <= 1e-12:
        return 0.5
    return float((value - low) / span)


def encode_generic_config(config: dict[str, Any], search_space: SearchSpace) -> np.ndarray:
    """Encode one arbitrary benchmark config into a [0, 1] feature vector."""

    normalized = search_space.coerce_config(config, use_defaults=False)
    values: list[float] = []
    for param in search_space:
        raw_value = normalized[param.name]
        if isinstance(param, FloatParam):
            values.append(
                _normalize_numeric_value(
                    float(raw_value),
                    low=float(param.low),
                    high=float(param.high),
                    log=bool(param.log),
                )
            )
            continue
        if isinstance(param, IntParam):
            values.append(
                _normalize_numeric_value(
                    float(raw_value),
                    low=float(param.low),
                    high=float(param.high),
                    log=bool(param.log),
                )
            )
            continue
        if not isinstance(param, CategoricalParam):
            raise TypeError(
                f"Generic PFNs candidate encoding does not support `{type(param).__name__}` for `{param.name}`."
            )
        values.extend(1.0 if raw_value == choice else 0.0 for choice in param.choices)
    return np.asarray(values, dtype=float)


def build_generic_candidate_pool(
    search_space: SearchSpace,
    *,
    seed: int,
    pool_size: int,
    task_name: str = "generic",
) -> EncodedCandidatePool:
    """Sample a deterministic generic candidate pool with normalized one-hot features."""

    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(pool_size * 50, 1024)
    while len(configs) < pool_size and attempts < max_attempts:
        attempts += 1
        candidate = search_space.sample(rng)
        candidate = search_space.coerce_config(candidate, use_defaults=False)
        identity = config_identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        configs.append(candidate)

    if not configs:
        raise RuntimeError(f"Failed to sample any PFNs candidate-pool configs for task `{task_name}`.")

    feature_names = generic_feature_names(search_space)
    features = np.asarray([encode_generic_config(config, search_space) for config in configs], dtype=float)
    metadata = tuple({"config_identity": config_identity(config)} for config in configs)
    return EncodedCandidatePool(
        task_name=task_name,
        configs=tuple(configs),
        features=features,
        feature_names=feature_names,
        candidate_metadata=metadata,
        full_candidate_count=len(configs),
    )


def oer_feature_names(search_space: SearchSpace) -> tuple[str, ...]:
    names: list[str] = []
    for column in OER_CATEGORICAL_FEATURES:
        param = search_space[column]
        if not isinstance(param, CategoricalParam):
            raise TypeError(f"OER categorical column `{column}` must be categorical, got `{type(param).__name__}`.")
        names.extend(f"{column}::{choice}" for choice in param.choices)
    names.extend(OER_NUMERICAL_FEATURES)
    return tuple(names)


def encode_oer_config(config: dict[str, Any], search_space: SearchSpace) -> np.ndarray:
    """Encode one OER config with fixed one-hot and min-max feature ordering."""

    normalized = search_space.coerce_config(config, use_defaults=False)
    values: list[float] = []

    for column in OER_CATEGORICAL_FEATURES:
        param = search_space[column]
        if not isinstance(param, CategoricalParam):
            raise TypeError(f"OER categorical column `{column}` must be categorical, got `{type(param).__name__}`.")
        choice = normalized[column]
        values.extend(1.0 if choice == candidate else 0.0 for candidate in param.choices)

    for column in OER_FLOAT_FEATURES[:3]:
        param = search_space[column]
        span = float(param.high) - float(param.low)
        scaled = 0.0 if span <= 1e-12 else (float(normalized[column]) - float(param.low)) / span
        values.append(float(scaled))
    for column in OER_INTEGER_FEATURES:
        param = search_space[column]
        span = float(param.high) - float(param.low)
        scaled = 0.0 if span <= 1e-12 else (float(normalized[column]) - float(param.low)) / span
        values.append(float(scaled))
    for column in OER_FLOAT_FEATURES[3:]:
        param = search_space[column]
        span = float(param.high) - float(param.low)
        scaled = 0.0 if span <= 1e-12 else (float(normalized[column]) - float(param.low)) / span
        values.append(float(scaled))

    return np.asarray(values, dtype=float)


def build_oer_candidate_pool(search_space: SearchSpace, *, seed: int, pool_size: int) -> EncodedCandidatePool:
    """Sample a deterministic OER candidate pool directly from the task search space."""

    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(pool_size * 50, 1024)
    while len(configs) < pool_size and attempts < max_attempts:
        attempts += 1
        candidate = search_space.sample(rng)
        candidate = search_space.coerce_config(candidate, use_defaults=False)
        identity = config_identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        configs.append(candidate)

    if not configs:
        raise RuntimeError("Failed to sample any OER PFNs4BO pool candidates.")

    feature_names = oer_feature_names(search_space)
    features = np.asarray([encode_oer_config(config, search_space) for config in configs], dtype=float)
    metadata = tuple({"config_identity": config_identity(config)} for config in configs)
    return EncodedCandidatePool(
        task_name=OER_TASK_NAME,
        configs=tuple(configs),
        features=features,
        feature_names=feature_names,
        candidate_metadata=metadata,
        full_candidate_count=len(configs),
    )


def _require_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "PFNs4BO molecule encoding requires RDKit. Install it with `uv sync --extra dev --extra bo-tutorial`."
        ) from exc
    return Chem, Descriptors


@lru_cache(maxsize=2)
def compute_molecule_descriptor_dataset(smiles_choices: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Compute dataset-level normalized molecule descriptors for the full SMILES archive."""

    cache_dir = Path(__file__).resolve().parents[3] / "artifacts" / "pfns4bo_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(smiles_choices).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"molecule_descriptors_{digest}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["raw"], cached["normalized"]

    Chem, Descriptors = _require_rdkit()
    raw_rows: list[list[float]] = []
    for index, smiles in enumerate(smiles_choices):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit failed to parse SMILES at dataset index {index}: {smiles!r}")
        raw_rows.append(
            [
                float(Descriptors.MolWt(molecule)),
                float(Descriptors.MolLogP(molecule)),
                float(Descriptors.TPSA(molecule)),
                float(Descriptors.NumHAcceptors(molecule)),
                float(Descriptors.NumHDonors(molecule)),
                float(Descriptors.NumRotatableBonds(molecule)),
                float(Descriptors.RingCount(molecule)),
                float(Descriptors.FractionCSP3(molecule)),
            ]
        )

    raw = np.asarray(raw_rows, dtype=float)
    minima = raw.min(axis=0)
    maxima = raw.max(axis=0)
    span = maxima - minima
    normalized = np.zeros_like(raw)
    non_constant = span > 1e-12
    normalized[:, non_constant] = (raw[:, non_constant] - minima[non_constant]) / span[non_constant]
    np.savez_compressed(cache_path, raw=raw, normalized=normalized)
    return raw, normalized


def build_molecule_candidate_pool(search_space: SearchSpace, *, seed: int, pool_size: int) -> EncodedCandidatePool:
    """Sample a deterministic molecule candidate pool from the original SMILES list."""

    smiles_param = search_space["SMILES"]
    if not isinstance(smiles_param, CategoricalParam):
        raise TypeError("The molecule PFNs4BO pool encoder requires a categorical `SMILES` parameter.")

    smiles_choices = tuple(str(choice) for choice in smiles_param.choices)
    raw_descriptors, normalized_descriptors = compute_molecule_descriptor_dataset(smiles_choices)
    population = list(range(len(smiles_choices)))
    rng = random.Random(seed)
    if pool_size >= len(population):
        selected_indices = population
    else:
        selected_indices = rng.sample(population, pool_size)

    configs = tuple({"SMILES": smiles_choices[index]} for index in selected_indices)
    features = normalized_descriptors[np.asarray(selected_indices, dtype=int)]
    metadata = tuple(
        {
            "dataset_index": int(index),
            "smiles": smiles_choices[index],
            "raw_descriptors": {
                name: float(raw_descriptors[index, descriptor_index])
                for descriptor_index, name in enumerate(MOLECULE_DESCRIPTOR_NAMES)
            },
        }
        for index in selected_indices
    )
    return EncodedCandidatePool(
        task_name=MOLECULE_TASK_NAME,
        configs=configs,
        features=features,
        feature_names=MOLECULE_DESCRIPTOR_NAMES,
        candidate_metadata=metadata,
        full_candidate_count=len(smiles_choices),
    )


__all__ = [
    "EncodedCandidatePool",
    "MOLECULE_DESCRIPTOR_NAMES",
    "build_generic_candidate_pool",
    "build_molecule_candidate_pool",
    "build_oer_candidate_pool",
    "build_pool_candidates",
    "compute_molecule_descriptor_dataset",
    "encode_generic_config",
    "encode_oer_config",
    "generic_feature_names",
    "oer_feature_names",
]
