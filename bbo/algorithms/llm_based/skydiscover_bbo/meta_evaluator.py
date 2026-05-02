"""SkyDiscover evaluator: score candidate modules that implement ``suggest_next_config``.

SkyDiscover writes the candidate solution to a temp file and calls ``evaluate(program_path)``.
This script must define ``evaluate(program_path: str) -> dict`` with a ``combined_score`` key.

Environment:
    BBO_SKYDISCOVER_META_CONTEXT: path to JSON written by ``SkydiscoverInterleavedAlgorithm``
    (parameter_specs, objective_direction, recent_history, seed, meta_combined_score_mode, ...).

Scoring:
    - ``meta_combined_score_mode == "contract"`` (default / non-synthetic): key-alignment score in [0, 10].
    - ``meta_combined_score_mode == "distance_to_known_optimum"``: mean Euclidean distance to the
      nearest known global optimum (same semantics as synthetic task metrics); ``combined_score`` is
      ``-mean_distance`` so larger is better and aligns with maximizing fitness in OpenEvolve Native.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import statistics
from typing import Any

import numpy as np

from bbo.algorithms.llm_based.generated_solver.specs import parameter_specs_to_search_space
from bbo.core.space import SearchSpace
from bbo.tasks.registry import SYNTHETIC_PROBLEM_REGISTRY, get_synthetic_problem

logger = logging.getLogger(__name__)

META_MODE_CONTRACT = "contract"
META_MODE_DISTANCE = "distance_to_known_optimum"
_BAD_DISTANCE_SCORE = -1e9


def evaluate(program_path: str) -> dict[str, Any]:
    """Return metrics for a candidate strategy file."""
    ctx_path = os.environ.get("BBO_SKYDISCOVER_META_CONTEXT")
    if not ctx_path or not os.path.isfile(ctx_path):
        return {"combined_score": 0.0, "error": "missing BBO_SKYDISCOVER_META_CONTEXT"}

    with open(ctx_path, encoding="utf-8") as handle:
        ctx = json.load(handle)

    parameter_specs: list[dict[str, Any]] = ctx["parameter_specs"]
    names = {p["name"] for p in parameter_specs}
    seed = int(ctx.get("seed", 0))
    direction = str(ctx.get("objective_direction", "minimize"))
    raw_hist = ctx.get("recent_history", [])
    history: list[tuple[dict[str, Any], float]] = [
        (dict(item[0]), float(item[1])) for item in raw_hist if len(item) >= 2
    ]

    meta_mode = str(ctx.get("meta_combined_score_mode", META_MODE_CONTRACT))

    spec = importlib.util.spec_from_file_location("_bbo_meta_candidate", program_path)
    if spec is None or spec.loader is None:
        return {"combined_score": 0.0, "error": "import_spec_failed"}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load candidate module: %s", exc)
        return {"combined_score": 0.0, "error": f"exec_failed:{type(exc).__name__}"}

    fn = getattr(mod, "suggest_next_config", None)
    if not callable(fn):
        return {"combined_score": 0.0, "error": "no_suggest_next_config"}

    if meta_mode == META_MODE_DISTANCE:
        return _evaluate_distance_mode(
            fn=fn,
            ctx=ctx,
            parameter_specs=parameter_specs,
            names=names,
            history=history,
            direction=direction,
            seed=seed,
        )

    return _evaluate_contract_mode(
        fn=fn,
        parameter_specs=parameter_specs,
        names=names,
        history=history,
        direction=direction,
        seed=seed,
    )


def _evaluate_contract_mode(
    *,
    fn: Any,
    parameter_specs: list[dict[str, Any]],
    names: set[str],
    history: list[tuple[dict[str, Any], float]],
    direction: str,
    seed: int,
) -> dict[str, Any]:
    scores: list[float] = []
    for trial_idx in (0, 1, 2):
        try:
            cfg = fn(
                history=history,
                parameter_specs=parameter_specs,
                objective_direction=direction,
                seed=seed,
                trial_index=trial_idx,
            )
        except Exception:
            scores.append(0.0)
            continue
        if not isinstance(cfg, dict):
            scores.append(0.0)
            continue
        if set(cfg.keys()) != names:
            scores.append(0.2)
            continue
        scores.append(1.0)

    combined = (sum(scores) / max(len(scores), 1)) * 10.0
    return {"combined_score": combined, "meta_trials_ok": sum(1 for s in scores if s >= 1.0)}


def _evaluate_distance_mode(
    *,
    fn: Any,
    ctx: dict[str, Any],
    parameter_specs: list[dict[str, Any]],
    names: set[str],
    history: list[tuple[dict[str, Any], float]],
    direction: str,
    seed: int,
) -> dict[str, Any]:
    resolved = _resolve_space_and_optima(ctx, parameter_specs)
    if resolved is None:
        return _evaluate_contract_mode(
            fn=fn,
            parameter_specs=parameter_specs,
            names=names,
            history=history,
            direction=direction,
            seed=seed,
        )

    space, optima_tuple = resolved
    distances: list[float] = []
    for trial_idx in (0, 1, 2):
        dist = _trial_distance_to_optima(
            fn=fn,
            space=space,
            optima=optima_tuple,
            parameter_specs=parameter_specs,
            names=names,
            history=history,
            direction=direction,
            seed=seed,
            trial_index=trial_idx,
        )
        if dist is not None and math.isfinite(dist):
            distances.append(float(dist))

    if not distances:
        return {
            "combined_score": float(_BAD_DISTANCE_SCORE),
            "mean_distance_to_known_optimum": None,
            "meta_trials_ok": 0,
            "error": "distance_mode_no_valid_trials",
            "meta_mode": META_MODE_DISTANCE,
        }

    mean_d = float(statistics.mean(distances))
    return {
        "combined_score": -mean_d,
        "mean_distance_to_known_optimum": mean_d,
        "trial_distances_to_known_optimum": distances,
        "meta_trials_ok": len(distances),
        "meta_mode": META_MODE_DISTANCE,
    }


def _resolve_space_and_optima(
    ctx: dict[str, Any], parameter_specs: list[dict[str, Any]]
) -> tuple[SearchSpace, tuple[tuple[float, ...], ...]] | None:
    problem_key = str(ctx.get("problem_key", ""))
    raw_optima = ctx.get("known_optima")

    if problem_key and problem_key in SYNTHETIC_PROBLEM_REGISTRY:
        definition = get_synthetic_problem(problem_key)
        return definition.search_space, definition.known_optima

    if not isinstance(raw_optima, list) or not raw_optima:
        logger.warning(
            "Distance mode requested but unknown problem_key=%r without known_optima; "
            "falling back to contract scoring.",
            problem_key,
        )
        return None

    try:
        space = parameter_specs_to_search_space(parameter_specs)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not build SearchSpace from parameter_specs: %s", exc)
        return None

    try:
        optima: list[tuple[float, ...]] = []
        for point in raw_optima:
            if not isinstance(point, (list, tuple)) or len(point) != len(space.numeric_parameters()):
                raise ValueError("known_optima point shape mismatches numeric search space.")
            optima.append(tuple(float(x) for x in point))
        return space, tuple(optima)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not normalize known_optima=%r: %s", raw_optima, exc)
        return None


def _trial_distance_to_optima(
    *,
    fn: Any,
    space: SearchSpace,
    optima: tuple[tuple[float, ...], ...],
    parameter_specs: list[dict[str, Any]],
    names: set[str],
    history: list[tuple[dict[str, Any], float]],
    direction: str,
    seed: int,
    trial_index: int,
) -> float | None:
    try:
        cfg = fn(
            history=history,
            parameter_specs=parameter_specs,
            objective_direction=direction,
            seed=seed,
            trial_index=trial_index,
        )
    except Exception:
        return None
    if not isinstance(cfg, dict) or set(cfg.keys()) != names:
        return None
    try:
        normalized = space.coerce_config(cfg, reject_extra=True, use_defaults=False)
        vector = space.to_numeric_vector(normalized)
    except (KeyError, TypeError, ValueError):
        return None

    if not optima:
        return float("nan")
    best = min(
        float(np.linalg.norm(vector - np.asarray(point, dtype=float))) for point in optima
    )
    return best


__all__ = ["evaluate"]
