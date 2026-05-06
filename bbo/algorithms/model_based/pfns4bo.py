"""PFNs4BO adapter with fixed backend routing for benchmark smoke validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import (
    ContinuousSearchSpaceConverter,
    ExternalOptimizerAdapter,
    ObjectiveDirection,
    TrialObservation,
    TrialSuggestion,
    build_continuous_converter,
)
from ...tasks.scientific.molecule import MOLECULE_TASK_NAME
from ...tasks.scientific.oer import OER_TASK_NAME
from .pfns4bo_encoding import EncodedCandidatePool, build_pool_candidates
from .pfns4bo_utils import (
    ContinuousPfnsOptimizer,
    DEFAULT_FAILURE_PENALTY,
    DEFAULT_PFNS_MODEL,
    DEFAULT_PFNS_POOL_SIZE,
    PfnsModelInfo,
    build_numeric_api_config,
    config_identity,
    deterministic_seed,
    load_torch_model,
    model_feature_capacity,
    normalize_pool_utilities,
    observation_to_continuous_value,
    require_pfns4bo,
    resolve_pfns_model,
    select_pfns_device,
)

POOL_BACKEND_TASKS = {OER_TASK_NAME, MOLECULE_TASK_NAME}


class Pfns4BoAlgorithm(ExternalOptimizerAdapter):
    """Single public PFNs4BO entrypoint with continuous and pool-based backends."""

    def __init__(
        self,
        *,
        device: str | None = None,
        pool_size: int = DEFAULT_PFNS_POOL_SIZE,
        model_name: str = DEFAULT_PFNS_MODEL,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
    ) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        self.requested_device = device
        self.pool_size = int(pool_size)
        self.model_name = model_name
        self.failure_penalty = float(failure_penalty)

        self._seed = 0
        self._device = "cpu:0"
        self._backend_name = ""
        self._history: list[TrialObservation] = []
        self._model_info: PfnsModelInfo | None = None
        self._pool: EncodedCandidatePool | None = None
        self._continuous_model: Any | None = None
        self._pool_model: Any | None = None
        self._continuous_converter: ContinuousSearchSpaceConverter | None = None

    @property
    def name(self) -> str:
        return "pfns4bo"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def model_info(self) -> PfnsModelInfo:
        if self._model_info is None:
            raise RuntimeError("PFNs4BO model info is not available before setup().")
        return self._model_info

    @property
    def candidate_pool(self) -> EncodedCandidatePool | None:
        return self._pool

    def setup(self, task_spec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("Pfns4BoAlgorithm currently supports exactly one objective.")

        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        self._device = select_pfns_device(self.requested_device)
        self._backend_name = self._select_backend(task_spec.name)
        self._history = []
        self._model_info = resolve_pfns_model(self.model_name)
        self._pool = None
        self._continuous_model = None
        self._pool_model = None
        self._continuous_converter = None

        if self._backend_name == "pool":
            self._pool = build_pool_candidates(task_spec, seed=self._seed, pool_size=self.pool_size)
            self._pool_model = load_torch_model(self.model_info.model_path)
            capacity = model_feature_capacity(self._pool_model)
            if capacity is not None and self._pool.features.shape[1] > capacity:
                raise ValueError(
                    f"PFNs model `{self.model_name}` supports at most {capacity} encoded features, "
                    f"but task `{task_spec.name}` encodes to {self._pool.features.shape[1]}."
                )
        else:
            self._continuous_model = load_torch_model(self.model_info.model_path)
            try:
                task_spec.search_space.numeric_bounds()
            except TypeError:
                self._continuous_converter = build_continuous_converter(task_spec.search_space, strategy="onehot")
            capacity = model_feature_capacity(self._continuous_model)
            feature_count = (
                len(self._continuous_converter.feature_names)
                if self._continuous_converter is not None
                else len(task_spec.search_space)
            )
            if capacity is not None and feature_count > capacity:
                raise ValueError(
                    f"PFNs model `{self.model_name}` supports at most {capacity} numeric features, "
                    f"but task `{task_spec.name}` exposes {feature_count}."
                )

    def ask(self) -> TrialSuggestion:
        if self._backend_name == "continuous":
            return self._ask_continuous()
        if self._backend_name == "pool":
            return self._ask_pool()
        raise RuntimeError("PFNs4BO backend was not initialized. Did setup() run?")

    def tell(self, observation: TrialObservation) -> None:
        self._history.append(observation)
        self.update_best_incumbent(observation)

    def _ask_continuous(self) -> TrialSuggestion:
        assert self._primary_name is not None
        search_space = self.require_search_space()
        converter = self._continuous_converter
        api_config = build_numeric_api_config(search_space) if converter is None else converter.continuous_api_config()

        with deterministic_seed(self._seed + len(self._history)):
            optimizer = ContinuousPfnsOptimizer(
                api_config,
                self._require_continuous_model(),
                minimize=self._primary_direction == ObjectiveDirection.MINIMIZE,
                device=self._device,
                verbose=False,
                rand_bool=False,
                sample_only_valid=False,
                round_suggests_to=8,
                min_initial_design=2,
                max_initial_design=2,
                fixed_initial_guess=0.5,
                rand_sugg_after_x_steps_of_stagnation=None,
                num_grad_steps=32,
                num_candidates=8,
                pre_sample_size=256,
            )

            for observation in self._history:
                value = observation_to_continuous_value(
                    observation,
                    primary_name=self._primary_name,
                    direction=self._primary_direction,
                    failure_penalty=self.failure_penalty,
                )
                observed_config = (
                    dict(observation.suggestion.config)
                    if converter is None
                    else converter.encode_feature_config(observation.suggestion.config)
                )
                optimizer.observe([observed_config], np.asarray([value], dtype=float))
            suggestion = optimizer.suggest(1)[0]

        if converter is None:
            config = search_space.coerce_config(dict(suggestion), use_defaults=False)
        else:
            config = converter.decode_feature_config(dict(suggestion), clip=True)
        return TrialSuggestion(
            config=config,
            metadata={
                **self._common_metadata(),
                "pfns_backend": "continuous",
                "pfns_categorical_to_continuous": None if converter is None else converter.strategy_name,
                "pfns_fixed_initial_guess": 0.5,
            },
        )

    def _ask_pool(self) -> TrialSuggestion:
        assert self._primary_name is not None
        pool = self._require_pool()
        observed_indices = self._history_pool_indices()
        observed_set = set(observed_indices)
        remaining_indices = [index for index in range(len(pool.configs)) if index not in observed_set]
        if not remaining_indices:
            raise RuntimeError(
                f"PFNs4BO exhausted the {pool.task_name} candidate pool of size {len(pool.configs)} before the task budget."
            )

        observed_matrix = (
            np.empty((0, pool.features.shape[1]), dtype=float)
            if not observed_indices
            else pool.features[np.asarray(observed_indices, dtype=int)]
        )
        utilities = normalize_pool_utilities(
            self._history,
            primary_name=self._primary_name,
            direction=self._primary_direction,
        )
        pending_matrix = pool.features[np.asarray(remaining_indices, dtype=int)]

        if len(self._history) < 2:
            pool_index = remaining_indices[0]
            metadata = {
                **self._common_metadata(),
                "pfns_backend": "pool",
                "pfns_pool_index": pool_index,
                "pfns_pool_size": len(pool.configs),
                "pfns_pool_full_candidate_count": pool.full_candidate_count,
                "pfns_pool_initial_design": True,
            }
            metadata.update(pool.candidate_metadata[pool_index])
            return TrialSuggestion(
                config=dict(pool.configs[pool_index]),
                metadata=metadata,
            )

        with deterministic_seed(self._seed + len(self._history)):
            require_pfns4bo()
            from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

            selector = TransformerBOMethod(
                self._require_pool_model(),
                device=self._device,
            )
            relative_index = int(selector.observe_and_suggest(observed_matrix, utilities, pending_matrix))

        pool_index = remaining_indices[relative_index]
        metadata = {
            **self._common_metadata(),
            "pfns_backend": "pool",
            "pfns_pool_index": pool_index,
            "pfns_pool_size": len(pool.configs),
            "pfns_pool_full_candidate_count": pool.full_candidate_count,
        }
        metadata.update(pool.candidate_metadata[pool_index])
        return TrialSuggestion(
            config=dict(pool.configs[pool_index]),
            metadata=metadata,
        )

    def _select_backend(self, task_name: str) -> str:
        if task_name in POOL_BACKEND_TASKS:
            return "pool"
        return "continuous"

    def _common_metadata(self) -> dict[str, Any]:
        return {
            "pfns_model": self.model_info.model_name,
            "pfns_model_path": str(self.model_info.model_path),
            "pfns_model_download_status": self.model_info.download_status,
            "pfns_model_preexisting": self.model_info.existed_before,
            "pfns_device": self._device,
            "pfns_history_length": len(self._history),
            "pfns_seed": self._seed,
        }

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
                        "PFNs4BO replay could not map a history config back onto the sampled candidate pool."
                    )
            if index in used:
                raise ValueError(f"PFNs4BO replay encountered a duplicate pool index: {index}.")
            used.add(index)
            resolved.append(index)
        return resolved

    def _require_pool(self) -> EncodedCandidatePool:
        if self._pool is None:
            raise RuntimeError("PFNs4BO pool backend has not been initialized.")
        return self._pool

    def _require_pool_model(self) -> Any:
        if self._pool_model is None:
            raise RuntimeError("PFNs4BO pool model has not been initialized.")
        return self._pool_model

    def _require_continuous_model(self) -> Any:
        if self._continuous_model is None:
            raise RuntimeError("PFNs4BO continuous model has not been initialized.")
        return self._continuous_model


__all__ = ["POOL_BACKEND_TASKS", "Pfns4BoAlgorithm"]
