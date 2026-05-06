"""Additional PFN-based BO variants built on a shared candidate-pool interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...core import (
    CategoricalParam,
    ExternalOptimizerAdapter,
    FloatParam,
    IntParam,
    ObjectiveDirection,
    TrialObservation,
    TrialSuggestion,
)
from .pfns4bo_encoding import EncodedCandidatePool, build_pool_candidates
from .pfns4bo_utils import (
    DEFAULT_PFNS_ACQUISITION,
    DEFAULT_PFNS_POOL_SIZE,
    config_identity,
    deterministic_seed,
    load_torch_model,
    model_feature_capacity,
    normalize_pool_utilities,
    normalize_torch_device_label,
    require_pandas,
    require_pfns4bo,
    require_tabpfn,
    select_pfns_device,
)

_ACQUISITIONS = {"ei", "ucb", "mean"}


def _configs_to_dataframe(configs: list[dict[str, Any]], search_space) -> Any:
    pandas = require_pandas()
    rows: list[dict[str, Any]] = []
    for config in configs:
        normalized = search_space.coerce_config(config, use_defaults=False)
        row: dict[str, Any] = {}
        for param in search_space:
            row[param.name] = normalized[param.name]
        rows.append(row)

    frame = pandas.DataFrame(rows, columns=search_space.names())
    for param in search_space:
        if isinstance(param, CategoricalParam):
            frame[param.name] = pandas.Categorical(frame[param.name], categories=list(param.choices))
        elif isinstance(param, IntParam):
            frame[param.name] = frame[param.name].astype("int64")
        elif isinstance(param, FloatParam):
            frame[param.name] = frame[param.name].astype("float64")
    return frame


def _score_tabpfn_outputs(*, acquisition: str, outputs: dict[str, Any], best_f: float) -> np.ndarray:
    if acquisition == "mean":
        return np.asarray(outputs["mean"], dtype=float)

    criterion = outputs["criterion"]
    logits = outputs["logits"]
    if acquisition == "ei":
        scores = criterion.ei(logits, best_f=best_f, maximize=True)
    elif acquisition == "ucb":
        scores = criterion.ucb(logits, best_f=best_f, maximize=True)
    else:  # pragma: no cover - guarded by constructor validation.
        raise ValueError(f"Unknown PFN acquisition `{acquisition}`.")
    return np.asarray(scores.detach().cpu().numpy(), dtype=float).reshape(-1)


class _PoolBasedPfnAlgorithm(ExternalOptimizerAdapter):
    """Shared candidate-pool loop for PFN-style BO variants."""

    variant_name = "pfn_variant"

    def __init__(
        self,
        *,
        device: str | None = None,
        pool_size: int = DEFAULT_PFNS_POOL_SIZE,
        acquisition: str = DEFAULT_PFNS_ACQUISITION,
    ) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if acquisition not in _ACQUISITIONS:
            available = ", ".join(sorted(_ACQUISITIONS))
            raise ValueError(f"Unknown PFN acquisition `{acquisition}`. Available: {available}")
        self.requested_device = device
        self.pool_size = int(pool_size)
        self.acquisition = acquisition

        self._seed = 0
        self._device = "cpu:0"
        self._history: list[TrialObservation] = []
        self._pool: EncodedCandidatePool | None = None

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError(f"{type(self).__name__} currently supports exactly one objective.")

        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        self._device = select_pfns_device(self.requested_device)
        self._history = []
        self._pool = build_pool_candidates(task_spec, seed=self._seed, pool_size=self.pool_size)
        self._prepare_backend()
        self._validate_candidate_pool()

    def ask(self) -> TrialSuggestion:
        pool = self._require_pool()
        observed_indices = set(self._history_pool_indices())
        remaining_indices = [index for index in range(len(pool.configs)) if index not in observed_indices]
        if not remaining_indices:
            raise RuntimeError(
                f"{type(self).__name__} exhausted the {pool.task_name} candidate pool of size {len(pool.configs)}."
            )

        if len(self._history) < 2:
            return self._build_pool_suggestion(remaining_indices[0], initial_design=True)

        relative_index = self._select_relative_index(remaining_indices)
        if relative_index < 0 or relative_index >= len(remaining_indices):
            raise RuntimeError(
                f"{type(self).__name__} selected invalid candidate index {relative_index} for {len(remaining_indices)}."
            )
        return self._build_pool_suggestion(remaining_indices[relative_index], initial_design=False)

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self.update_best_incumbent(observation)

    def _prepare_backend(self) -> None:
        """Allow subclasses to initialize lazy runtime state."""

    def _validate_candidate_pool(self) -> None:
        """Allow subclasses to validate the pool against the loaded model."""

    def _select_relative_index(self, remaining_indices: list[int]) -> int:
        raise NotImplementedError

    def _backend_metadata(self) -> dict[str, Any]:
        return {"pfns_variant": self.variant_name}

    def _build_pool_suggestion(self, pool_index: int, *, initial_design: bool) -> TrialSuggestion:
        pool = self._require_pool()
        metadata = {
            "pfns_backend": "candidate_pool",
            "pfns_pool_index": pool_index,
            "pfns_pool_size": len(pool.configs),
            "pfns_pool_full_candidate_count": pool.full_candidate_count,
            "pfns_pool_initial_design": initial_design,
            "pfns_acquisition": self.acquisition,
            "pfns_device": self._device,
            "pfns_history_length": len(self._history),
            "pfns_seed": self._seed,
            **self._backend_metadata(),
        }
        metadata.update(pool.candidate_metadata[pool_index])
        return TrialSuggestion(config=dict(pool.configs[pool_index]), metadata=metadata)

    def _history_pool_indices(self) -> list[int]:
        pool = self._require_pool()
        identity_to_indices: dict[str, list[int]] = {}
        for index, config in enumerate(pool.configs):
            identity_to_indices.setdefault(config_identity(config), []).append(index)

        used: set[int] = set()
        resolved: list[int] = []
        for observation in self._history:
            metadata_index = observation.suggestion.metadata.get("pfns_pool_index")
            if metadata_index is not None:
                index = int(metadata_index)
            else:
                identity = config_identity(observation.suggestion.config)
                candidates = identity_to_indices.get(identity, [])
                index = next((candidate for candidate in candidates if candidate not in used), -1)
                if index < 0:
                    raise ValueError(
                        f"{type(self).__name__} replay could not map a history config back onto the sampled pool."
                    )
            if index in used:
                raise ValueError(f"{type(self).__name__} replay encountered a duplicate pool index: {index}.")
            used.add(index)
            resolved.append(index)
        return resolved

    def _require_pool(self) -> EncodedCandidatePool:
        if self._pool is None:
            raise RuntimeError(f"{type(self).__name__} has not been initialized.")
        return self._pool


class TabPfnV2BoAlgorithm(_PoolBasedPfnAlgorithm):
    """BO adapter that swaps PFNs4BO's surrogate for TabPFN v2 regression."""

    variant_name = "tabpfn_v2"

    def __init__(
        self,
        *,
        device: str | None = None,
        pool_size: int = DEFAULT_PFNS_POOL_SIZE,
        acquisition: str = DEFAULT_PFNS_ACQUISITION,
        n_estimators: int = 8,
        ignore_pretraining_limits: bool = True,
        fit_mode: str = "fit_preprocessors",
    ) -> None:
        super().__init__(device=device, pool_size=pool_size, acquisition=acquisition)
        self.n_estimators = int(n_estimators)
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self.fit_mode = fit_mode

    @property
    def name(self) -> str:
        return "pfns4bo_tabpfn_v2"

    def _select_relative_index(self, remaining_indices: list[int]) -> int:
        require_tabpfn()
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        search_space = self.require_search_space()
        pool = self._require_pool()
        observed_indices = self._history_pool_indices()
        observed_configs = [pool.configs[index] for index in observed_indices]
        pending_configs = [pool.configs[index] for index in remaining_indices]
        target_utilities = normalize_pool_utilities(
            self._history,
            primary_name=self._primary_name or "",
            direction=self._primary_direction,
        )
        regressor = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=normalize_torch_device_label(self._device),
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            fit_mode=self.fit_mode,
            random_state=self._seed,
        )
        regressor.fit(_configs_to_dataframe(observed_configs, search_space), target_utilities)
        outputs = regressor.predict(_configs_to_dataframe(pending_configs, search_space), output_type="full")
        scores = _score_tabpfn_outputs(
            acquisition=self.acquisition,
            outputs=outputs,
            best_f=float(target_utilities.max()) if len(target_utilities) else 0.0,
        )
        return int(np.argmax(scores))

    def _backend_metadata(self) -> dict[str, Any]:
        return {
            "pfns_variant": self.variant_name,
            "tabpfn_model_version": "v2",
            "tabpfn_n_estimators": self.n_estimators,
            "tabpfn_fit_mode": self.fit_mode,
            "tabpfn_ignore_pretraining_limits": self.ignore_pretraining_limits,
        }


class CustomPfnsBoAlgorithm(_PoolBasedPfnAlgorithm):
    """BO adapter that uses a locally trained PFN checkpoint over a candidate pool."""

    variant_name = "custom_pfn"

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str | None = None,
        pool_size: int = DEFAULT_PFNS_POOL_SIZE,
        acquisition: str = DEFAULT_PFNS_ACQUISITION,
    ) -> None:
        if model_path is None or not str(model_path):
            raise ValueError("model_path must be provided for CustomPfnsBoAlgorithm.")
        super().__init__(device=device, pool_size=pool_size, acquisition=acquisition)
        self.model_path = Path(model_path).expanduser().resolve()
        self._model: Any | None = None
        self._metadata_path = self.model_path.with_suffix(".metadata.json")
        self._training_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "pfns4bo_custom"

    def _prepare_backend(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Custom PFN model not found: {self.model_path}")
        self._model = load_torch_model(self.model_path)
        if self._metadata_path.exists():
            self._training_metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        else:
            self._training_metadata = {}

    def _validate_candidate_pool(self) -> None:
        pool = self._require_pool()
        capacity = model_feature_capacity(self._require_model())
        if capacity is not None and pool.features.shape[1] > capacity:
            raise ValueError(
                f"Custom PFN at `{self.model_path}` supports at most {capacity} encoded features, "
                f"but task `{pool.task_name}` encodes to {pool.features.shape[1]}."
            )

    def _select_relative_index(self, remaining_indices: list[int]) -> int:
        pool = self._require_pool()
        observed_indices = self._history_pool_indices()
        observed_matrix = (
            np.empty((0, pool.features.shape[1]), dtype=float)
            if not observed_indices
            else pool.features[np.asarray(observed_indices, dtype=int)]
        )
        pending_matrix = pool.features[np.asarray(remaining_indices, dtype=int)]
        target_utilities = normalize_pool_utilities(
            self._history,
            primary_name=self._primary_name or "",
            direction=self._primary_direction,
        )

        with deterministic_seed(self._seed + len(self._history)):
            require_pfns4bo()
            from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

            selector = TransformerBOMethod(self._require_model(), device=self._device)
            return int(selector.observe_and_suggest(observed_matrix, target_utilities, pending_matrix))

    def _backend_metadata(self) -> dict[str, Any]:
        metadata = {
            "pfns_variant": self.variant_name,
            "pfns_model_path": str(self.model_path),
            "pfns_training_metadata_path": str(self._metadata_path) if self._metadata_path.exists() else None,
        }
        for key in ("prior", "max_features", "epochs", "steps_per_epoch", "seq_len"):
            if key in self._training_metadata:
                metadata[f"pfns_training_{key}"] = self._training_metadata[key]
        return metadata

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Custom PFN model has not been loaded.")
        return self._model


__all__ = [
    "CustomPfnsBoAlgorithm",
    "TabPfnV2BoAlgorithm",
]
