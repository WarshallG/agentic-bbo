"""Shared helpers for the PFNs4BO integration."""

from __future__ import annotations

import contextlib
import gzip
import json
import random
import shutil
import subprocess
import typing
import urllib.request
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...core import FloatParam, IntParam, ObjectiveDirection, SearchSpace, TrialObservation

DEFAULT_PFNS_MODEL = "hebo_plus"
DEFAULT_PFNS_POOL_SIZE = 256
DEFAULT_FAILURE_PENALTY = 1e12
DEFAULT_PFNS_ACQUISITION = "ei"

PFNS_MODEL_ATTRS = {
    "hebo_plus": "hebo_plus_model",
    "hebo_plus_userprior": "hebo_plus_userprior_model",
    "bnn": "bnn_model",
}
PFNS_MODEL_BASE_URL = "https://raw.githubusercontent.com/automl/PFNs4BO/main/pfns4bo/final_models"


@dataclass(frozen=True)
class PfnsModelInfo:
    """Resolved PFNs model metadata."""

    model_name: str
    attribute_name: str
    model_path: Path
    existed_before: bool
    exists_after: bool
    auto_download_attempted: bool

    @property
    def download_status(self) -> str:
        if self.exists_after and self.auto_download_attempted:
            return "download_attempted_and_model_present"
        if self.exists_after:
            return "model_already_present"
        return "model_missing_after_prepare"


def require_pfns4bo() -> Any:
    """Import PFNs4BO lazily so the base install stays lightweight."""

    try:
        import pfns4bo
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "`pfns4bo` requires the optional PFNs4BO dependencies. "
            "Install them with `uv sync --extra dev --extra pfns4bo`."
        ) from exc
    patch_pfns_torch_compat()
    return pfns4bo


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "`pfns4bo` requires PyTorch. Install it with `uv sync --extra dev --extra pfns4bo`."
        ) from exc
    return torch


def require_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "This PFN variant requires pandas. Install it with `uv sync --extra dev --extra tabpfn`."
        ) from exc
    return pandas


def require_tabpfn() -> Any:
    try:
        import tabpfn
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "This PFN variant requires TabPFN. Install it with `uv sync --extra dev --extra tabpfn`."
        ) from exc
    return tabpfn


def patch_pfns_torch_compat() -> None:
    """Patch PFNs4BO's legacy Torch imports for newer Torch releases."""

    torch = require_torch()
    import torch.nn.functional as F
    import torch.nn.modules.transformer as transformer_module
    from gpytorch.priors.prior import Prior

    if not hasattr(transformer_module, "Module"):
        transformer_module.Module = torch.nn.Module
    if not hasattr(transformer_module, "Tensor"):
        transformer_module.Tensor = torch.Tensor
    if not hasattr(transformer_module, "Optional"):
        transformer_module.Optional = typing.Optional

    try:
        import botorch.models.gp_regression as gp_regression
        import botorch.fit as botorch_fit
        from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
    except Exception:
        gp_regression = None
        botorch_fit = None
    if gp_regression is not None and not hasattr(gp_regression, "MIN_INFERRED_NOISE_LEVEL"):
        gp_regression.MIN_INFERRED_NOISE_LEVEL = MIN_INFERRED_NOISE_LEVEL
    if botorch_fit is not None and not hasattr(botorch_fit, "fit_gpytorch_model") and hasattr(
        botorch_fit, "fit_gpytorch_mll"
    ):
        botorch_fit.fit_gpytorch_model = botorch_fit.fit_gpytorch_mll

    if not hasattr(builtins, "List"):
        builtins.List = typing.List
    if not hasattr(builtins, "Optional"):
        builtins.Optional = typing.Optional
    if not hasattr(builtins, "Union"):
        builtins.Union = typing.Union
    if not hasattr(builtins, "Tuple"):
        builtins.Tuple = typing.Tuple
    if not hasattr(builtins, "Prior"):
        builtins.Prior = Prior

    try:
        import pfns4bo.layer as layer_module
    except Exception:
        return
    if not hasattr(layer_module, "F"):
        layer_module.F = F


def select_pfns_device(requested: str | None) -> str:
    """Choose a safe PFNs device string."""

    torch = require_torch()
    if requested is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested PFNs device `{requested}` but CUDA is not available.")
    return requested


def normalize_torch_device_label(device: str) -> str:
    """Collapse torch device strings into a form accepted by downstream libraries."""

    if device.startswith("cpu"):
        return "cpu"
    return device


def resolve_pfns_model(model_name: str) -> PfnsModelInfo:
    """Resolve the configured PFNs model path, allowing the upstream helper to download it."""

    pfns4bo = require_pfns4bo()
    attribute_name = PFNS_MODEL_ATTRS.get(model_name)
    if attribute_name is None:
        available = ", ".join(sorted(PFNS_MODEL_ATTRS))
        raise ValueError(f"Unknown PFNs4BO model `{model_name}`. Available: {available}")

    model_dict = getattr(pfns4bo, "model_dict", None)
    if model_dict is None or attribute_name not in model_dict:
        raise RuntimeError(f"PFNs4BO does not expose the expected model attribute `{attribute_name}`.")

    expected_path = Path(str(model_dict[attribute_name])).expanduser().resolve()
    existed_before = expected_path.exists()
    auto_download_attempted = ensure_pfns_model_file(expected_path) if not existed_before else False
    exists_after = expected_path.exists()
    if not exists_after:
        raise FileNotFoundError(
            f"PFNs4BO model `{model_name}` was expected at `{expected_path}` but is still missing after download/unzip."
        )
    return PfnsModelInfo(
        model_name=model_name,
        attribute_name=attribute_name,
        model_path=expected_path,
        existed_before=existed_before,
        exists_after=exists_after,
        auto_download_attempted=auto_download_attempted,
    )


def ensure_pfns_model_file(model_path: Path) -> bool:
    """Download and unzip one PFNs model file when it is missing."""

    if model_path.exists():
        return False

    model_path.parent.mkdir(parents=True, exist_ok=True)
    compressed_path = model_path.with_suffix(model_path.suffix + ".gz")
    if not compressed_path.exists():
        url = f"{PFNS_MODEL_BASE_URL}/{model_path.name}.gz"
        download_error: Exception | None = None
        curl_path = shutil.which("curl")
        if curl_path is not None:
            result = subprocess.run(
                [curl_path, "-L", url, "-o", str(compressed_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not compressed_path.exists() or compressed_path.stat().st_size == 0:
                stderr = result.stderr.strip() or result.stdout.strip()
                download_error = RuntimeError(stderr or f"`curl` exited with status {result.returncode}.")
        if download_error is not None or not compressed_path.exists():
            try:
                with urllib.request.urlopen(url, timeout=300) as response, compressed_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            except Exception as exc:
                raise RuntimeError(f"Failed to download PFNs model from `{url}`.") from (download_error or exc)

    if not model_path.exists():
        with gzip.open(compressed_path, "rb") as source, model_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    return True


def load_torch_model(model_path: Path) -> Any:
    """Load one serialized PFNs model onto CPU."""

    torch = require_torch()
    patch_pfns_torch_compat()
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.eval()
    return model


def model_feature_capacity(model: Any) -> int | None:
    """Best-effort extraction of the PFNs encoder input width."""

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    base_encoder = getattr(encoder, "base_encoder", None)
    if base_encoder is not None:
        return int(getattr(base_encoder, "in_features", 0) or 0) or None
    if hasattr(encoder, "__iter__"):
        for module in encoder:
            nested = model_feature_capacity(type("EncoderWrapper", (), {"encoder": module})())
            if nested is not None:
                return nested
    return None


@contextlib.contextmanager
def deterministic_seed(seed: int) -> Iterator[None]:
    """Temporarily seed Python, NumPy, and Torch RNGs."""

    torch = require_torch()

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_numeric_api_config(search_space: SearchSpace) -> dict[str, dict[str, Any]]:
    """Convert a numeric benchmark search space into PFNs4BO's Bayesmark-style API config."""

    api_config: dict[str, dict[str, Any]] = {}
    for param in search_space:
        if isinstance(param, FloatParam):
            api_config[param.name] = {
                "type": "real",
                "space": "log" if param.log else "linear",
                "range": (float(param.low), float(param.high)),
            }
        elif isinstance(param, IntParam):
            api_config[param.name] = {
                "type": "int",
                "space": "log" if param.log else "linear",
                "range": (int(param.low), int(param.high)),
            }
        else:
            raise TypeError(
                f"PFNs4BO continuous mode only supports numeric parameters; found `{type(param).__name__}`."
            )
    return api_config


def observation_to_continuous_value(
    observation: TrialObservation,
    *,
    primary_name: str,
    direction: ObjectiveDirection,
    failure_penalty: float = DEFAULT_FAILURE_PENALTY,
) -> float:
    """Convert one observation into the scalar value expected by PFNs4BO's continuous API."""

    if observation.success and primary_name in observation.objectives:
        return float(observation.objectives[primary_name])
    if direction == ObjectiveDirection.MAXIMIZE:
        return -abs(float(failure_penalty))
    return abs(float(failure_penalty))


def normalize_pool_utilities(
    history: list[TrialObservation],
    *,
    primary_name: str,
    direction: ObjectiveDirection,
) -> np.ndarray:
    """Map observed objectives onto a stable [0, 1] utility scale for the pool-based PFNs interface."""

    if not history:
        return np.empty((0,), dtype=float)

    converted: list[float | None] = []
    successful_utilities: list[float] = []
    for observation in history:
        if observation.success and primary_name in observation.objectives:
            objective = float(observation.objectives[primary_name])
            utility = -objective if direction == ObjectiveDirection.MINIMIZE else objective
            converted.append(utility)
            successful_utilities.append(utility)
        else:
            converted.append(None)

    if not successful_utilities:
        return np.zeros((len(history),), dtype=float)

    worst_success = min(successful_utilities)
    best_success = max(successful_utilities)
    failure_utility = worst_success - max(1.0, abs(best_success - worst_success) + 1.0)
    filled = np.asarray(
        [failure_utility if value is None else value for value in converted],
        dtype=float,
    )
    low = float(filled.min())
    high = float(filled.max())
    if high - low <= 1e-12:
        return np.full((len(history),), 0.5, dtype=float)
    return (filled - low) / (high - low)


def config_identity(config: dict[str, Any]) -> str:
    """Stable JSON identity for candidate-pool bookkeeping."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class ContinuousPfnsOptimizer:
    """Compatibility copy of PFNs4BO's continuous optimizer without the legacy tuning imports."""

    def __init__(
        self,
        api_config: dict[str, dict[str, Any]],
        model: Any,
        *,
        minimize: bool = True,
        device: str = "cpu:0",
        acqf_optimizer_name: str = "lbfgs",
        sobol_sampler: bool = False,
        verbose: bool = False,
        rand_bool: bool = False,
        sample_only_valid: bool = False,
        round_suggests_to: int | None = 8,
        min_initial_design: int = 0,
        max_initial_design: int | None = None,
        fixed_initial_guess: float | None = 0.5,
        rand_sugg_after_x_steps_of_stagnation: int | None = None,
        minmax_encode_y: bool = False,
        **acqf_kwargs: Any,
    ) -> None:
        from scipy.special import expit, logit
        from torch.quasirandom import SobolEngine

        self._expit = expit
        self._logit = logit
        self.model = model
        self.model.eval()
        self.api_config = {key: value for key, value in sorted(api_config.items())}
        self.device = device
        self.minimize = minimize
        self.acqf_optimizer_name = acqf_optimizer_name
        self.sobol_sampler = sobol_sampler
        self.verbose = verbose
        self.rand_bool = rand_bool
        self.sample_only_valid = sample_only_valid
        self.round_suggests_to = round_suggests_to
        self.min_initial_design = min_initial_design
        self.max_initial_design = max_initial_design
        self.fixed_initial_guess = fixed_initial_guess
        self.rand_sugg_after_x_steps_of_stagnation = rand_sugg_after_x_steps_of_stagnation
        self.minmax_encode_y = minmax_encode_y
        self.acqf_kwargs = dict(acqf_kwargs)

        self.X: list[list[float]] = []
        self.y: list[float] = []
        self.hp_names = list(self.api_config.keys())
        self.rand_prev = False
        self._create_scaler()
        self.sobol = SobolEngine(len(self.max_values), scramble=True)

    def _create_scaler(self) -> None:
        self.min_values = []
        self.max_values = []
        self.spaces = []
        self.types = []
        for feature in self.api_config:
            self.spaces.append(self.api_config[feature].get("space", "bool"))
            self.types.append(self.api_config[feature]["type"])
            if self.types[-1] == "bool":
                feature_range = [0, 1]
            else:
                feature_range = list(self.api_config[feature]["range"])
            feature_range[0] = self.transform_feature(feature_range[0], -1)
            feature_range[1] = self.transform_feature(feature_range[1], -1)
            self.min_values.append(feature_range[0])
            self.max_values.append(feature_range[1])
        self.max_values = np.asarray(self.max_values, dtype=float)
        self.min_values = np.asarray(self.min_values, dtype=float)

    def transform_feature_inverse(self, value: float, feature_index: int) -> Any:
        if self.spaces[feature_index] == "log":
            value = float(np.exp(value))
        elif self.spaces[feature_index] == "logit":
            value = float(self._expit(value))
        if self.types[feature_index] == "int":
            if self.rand_bool:
                value = int(value) + int(np.random.rand() < (value - int(value)))
            else:
                value = int(np.round(value))
        elif self.types[feature_index] == "bool":
            if self.rand_bool:
                value = bool(np.random.rand() < value)
            else:
                value = bool(np.round(value))
        return value

    def transform_feature(self, value: Any, feature_index: int) -> float:
        if np.isinf(value) or np.isnan(value):
            return 0.0
        numeric = float(value)
        if self.spaces[feature_index] == "log":
            numeric = float(np.log(numeric))
        elif self.spaces[feature_index] == "logit":
            numeric = float(self._logit(numeric))
        elif self.types[feature_index] == "bool":
            numeric = float(int(bool(numeric)))
        return numeric

    def transform(self, config: dict[str, Any]) -> list[float]:
        transformed = []
        for feature_index, feature in enumerate(config.keys()):
            transformed.append(self.transform_feature(config[feature], feature_index))
        return transformed

    def transform_inverse(self, values: list[float]) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for feature_index, hp_name in enumerate(self.hp_names):
            config[hp_name] = self.transform_feature_inverse(values[feature_index], feature_index)
        return config

    def transform_back(self, guess: np.ndarray) -> dict[str, Any]:
        if self.round_suggests_to is not None:
            guess = np.round(guess, self.round_suggests_to)
        guess = guess * (self.max_values - self.min_values) + self.min_values
        return self.transform_inverse(guess.tolist())

    def min_max_encode(self, values: np.ndarray) -> Any:
        import torch

        normalized = (values - self.min_values) / (self.max_values - self.min_values)
        tensor = torch.tensor(normalized, dtype=torch.float32)
        return torch.clamp(tensor, min=0.0, max=1.0)

    def random_suggest(self) -> list[dict[str, Any]]:
        self.rand_prev = True
        if self.sobol_sampler:
            guess = self.sobol.draw(1).numpy()[0]
            guess = guess * (self.max_values - self.min_values) + self.min_values
        else:
            guess = np.asarray(
                [np.random.uniform(self.min_values[i], self.max_values[i], 1)[0] for i in range(len(self.max_values))],
                dtype=float,
            )
        config = {
            feature: self.transform_feature_inverse(float(guess[index]), index)
            for index, feature in enumerate(self.api_config)
        }
        return [config]

    def initial_design_suggest(self) -> list[dict[str, Any]]:
        """Return a deterministic initial-design point without falling back to random search."""

        if len(self.X) == 0 and self.fixed_initial_guess is not None:
            return [self.transform_back(np.asarray([self.fixed_initial_guess] * len(self.max_values), dtype=float))]

        sobol_index = len(self.X) - (1 if self.fixed_initial_guess is not None else 0)
        sobol_index = max(sobol_index, 0)
        sobol_points = self.sobol.draw(sobol_index + 1).numpy()
        return [self.transform_back(np.asarray(sobol_points[sobol_index], dtype=float))]

    def suggest(self, n_suggestions: int = 1) -> list[dict[str, Any]]:
        import torch
        from sklearn.preprocessing import MinMaxScaler

        require_pfns4bo()
        from pfns4bo.scripts import acquisition_functions

        if n_suggestions != 1:
            raise AssertionError("Only one suggestion at a time is supported")

        num_initial_design = max(len(self.max_values), self.min_initial_design)
        if self.max_initial_design is not None:
            num_initial_design = min(num_initial_design, self.max_initial_design)

        if len(self.X) < num_initial_design:
            return self.initial_design_suggest()

        temp_X = self.min_max_encode(np.asarray(self.X, dtype=float))
        if self.minmax_encode_y:
            temp_y_np = MinMaxScaler().fit_transform(np.asarray(self.y, dtype=float).reshape(-1, 1)).reshape(-1)
        else:
            temp_y_np = np.asarray(self.y, dtype=float)
        temp_y = torch.tensor(temp_y_np, dtype=torch.float32)

        if (
            self.rand_sugg_after_x_steps_of_stagnation is not None
            and len(self.y) > self.rand_sugg_after_x_steps_of_stagnation
            and not self.rand_prev
        ):
            if temp_y[:-self.rand_sugg_after_x_steps_of_stagnation].max() == temp_y.max():
                return self.random_suggest()

        temp_X = temp_X.to(self.device)
        temp_y = temp_y.to(self.device)

        if self.acqf_optimizer_name != "lbfgs":
            raise ValueError("ContinuousPfnsOptimizer currently supports only `lbfgs`.")

        if self.sample_only_valid:
            def rand_sample_func(sample_count: int):
                pre_samples = torch.rand(sample_count, temp_X.shape[1], device="cpu")
                back_transformed = [self.transform_back(sample.cpu().numpy()) for sample in pre_samples]
                transformed = np.asarray([self.transform(config) for config in back_transformed], dtype=float)
                return self.min_max_encode(transformed).to(self.device)
            dims_wo_gradient_opt = [index for index, type_name in enumerate(self.types) if type_name != "real"]
        else:
            rand_sample_func = None
            dims_wo_gradient_opt = []

        _, x_options, _, x_rs, x_rs_eis = acquisition_functions.optimize_acq_w_lbfgs(
            self.model,
            temp_X,
            temp_y,
            device=self.device,
            verbose=self.verbose,
            rand_sample_func=rand_sample_func,
            dims_wo_gradient_opt=dims_wo_gradient_opt,
            **{"apply_power_transform": False, **self.acqf_kwargs},
        )

        back_transformed_x_options = [self.transform_back(np.asarray(option, dtype=float)) for option in x_options]
        opt_X = np.asarray([self.transform(config) for config in back_transformed_x_options], dtype=float)
        opt_X = self.min_max_encode(opt_X)
        opt_new = ~(opt_X[:, None] == temp_X[None].cpu()).all(-1).any(1)
        for index, _ in enumerate(opt_X):
            if opt_new[index]:
                self.rand_prev = False
                return [back_transformed_x_options[index]]

        back_transformed_x_rs = [self.transform_back(np.asarray(option, dtype=float)) for option in x_rs]
        opt_X = np.asarray([self.transform(config) for config in back_transformed_x_rs], dtype=float)
        opt_X = self.min_max_encode(opt_X)
        opt_new = ~(opt_X[:, None] == temp_X[None].cpu()).all(-1).any(1)
        for index, _ in enumerate(opt_X):
            if opt_new[index]:
                self.rand_prev = False
                return [back_transformed_x_rs[index]]

        raise RuntimeError(
            "PFNs4BO could not find a new continuous candidate without falling back to random_suggest()."
        )

    def observe(self, X: list[dict[str, Any]], y: np.ndarray) -> None:
        if np.isinf(y).any() and np.asarray(y).max() > 0:
            y[:] = 1e10
        if np.isnan(y).any() or np.isinf(y).any():
            raise AssertionError("ContinuousPfnsOptimizer does not accept NaN or inf observations.")
        if len(y) != 1 or len(X) != 1:
            raise AssertionError("Only one suggestion at a time is supported")

        config = {key: value for key, value in sorted(X[0].items())}
        if list(config.keys()) != self.hp_names:
            raise ValueError("Observed config keys do not match the PFNs API config.")
        transformed = self.transform(config)
        self.X.append(transformed)
        observed_value = float(y[0])
        self.y.append(-observed_value if self.minimize else observed_value)


__all__ = [
    "ContinuousPfnsOptimizer",
    "DEFAULT_PFNS_ACQUISITION",
    "DEFAULT_FAILURE_PENALTY",
    "DEFAULT_PFNS_MODEL",
    "DEFAULT_PFNS_POOL_SIZE",
    "PFNS_MODEL_ATTRS",
    "PFNS_MODEL_BASE_URL",
    "PfnsModelInfo",
    "build_numeric_api_config",
    "config_identity",
    "deterministic_seed",
    "ensure_pfns_model_file",
    "load_torch_model",
    "model_feature_capacity",
    "normalize_pool_utilities",
    "normalize_torch_device_label",
    "observation_to_continuous_value",
    "patch_pfns_torch_compat",
    "require_pandas",
    "require_pfns4bo",
    "require_tabpfn",
    "require_torch",
    "resolve_pfns_model",
    "select_pfns_device",
]
