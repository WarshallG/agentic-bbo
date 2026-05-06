#!/usr/bin/env python3
"""Example GP/LCB candidate generator for a BBO agent workspace.

The script is intentionally editable by the agent.  It reads only workspace
state, proposes candidates, validates them, and prints the strict benchmark JSON
schema.  It never calls the benchmark evaluator.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbo_tools import BBO


def main() -> int:
    bbo = BBO()
    space = bbo.search_space()
    objective = bbo.objective()["objective"]
    history = bbo.history(mode="all", limit=200)["trials"]
    parameters = list(space["parameters"])
    successful = [
        trial
        for trial in history
        if trial.get("status") == "success" and objective["name"] in trial.get("objectives", {})
    ]

    if len(successful) < max(3, len(parameters) + 1):
        candidates = _fallback_samples(bbo, n=8, reason="not enough evaluated points for a stable GP")
    else:
        try:
            candidates = _gp_lcb_candidates(bbo, parameters, successful, objective, n=8)
        except Exception as exc:
            candidates = _fallback_samples(bbo, n=8, reason=f"GP fallback after {type(exc).__name__}: {exc}")

    validation = bbo.validate(candidates)
    valid = [item["config"] for item in validation["valid"] if not item.get("duplicate")]
    if len(valid) < 4:
        extra = _fallback_samples(bbo, n=8, reason="validation refill")
        refill = bbo.validate(extra)
        for item in refill["valid"]:
            config = item["config"]
            if not item.get("duplicate") and config not in valid:
                valid.append(config)
            if len(valid) >= 4:
                break

    payload = {
        "candidates": [
            {"config": config, "rationale": "GP/LCB workspace analysis or validated fallback sample"}
            for config in valid[:4]
        ]
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _fallback_samples(bbo: BBO, *, n: int, reason: str) -> list[dict[str, Any]]:
    del reason
    incumbent = bbo.incumbent().get("incumbent")
    strategy = "around_incumbent" if incumbent else "random"
    sampled = bbo.sample(n=n, strategy=strategy, seed=17)
    return [{"config": item["config"]} for item in sampled["candidates"]]


def _gp_lcb_candidates(
    bbo: BBO,
    parameters: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    objective: dict[str, Any],
    *,
    n: int,
) -> list[dict[str, Any]]:
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    rng = random.Random(19)
    x_train = np.array([_encode_config(trial["config"], parameters) for trial in trials], dtype=float)
    y_raw = np.array([float(trial["objectives"][objective["name"]]) for trial in trials], dtype=float)
    y_train = -y_raw if objective.get("direction") == "maximize" else y_raw
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    y_scaled = (y_train - y_mean) / y_std

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=np.ones(x_train.shape[1]),
        length_scale_bounds=(0.05, 5.0),
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=0, n_restarts_optimizer=2)
    model.fit(x_train, y_scaled)

    pool = _candidate_pool(bbo, parameters, rng, size=256)
    x_pool = np.array([_encode_config(config, parameters) for config in pool], dtype=float)
    mean, std = model.predict(x_pool, return_std=True)
    scores = mean - 1.5 * std
    order = np.argsort(scores)
    return [{"config": pool[int(index)]} for index in order[:n]]


def _candidate_pool(bbo: BBO, parameters: list[dict[str, Any]], rng: random.Random, *, size: int) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bbo.sample(n=min(size, 128), strategy="random", seed=23)["candidates"]:
        config = item["config"]
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            pool.append(config)
            seen.add(key)
    while len(pool) < size:
        config = _random_config(parameters, rng)
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            pool.append(config)
            seen.add(key)
    return pool


def _encode_config(config: dict[str, Any], parameters: list[dict[str, Any]]) -> list[float]:
    encoded: list[float] = []
    for param in parameters:
        name = param["name"]
        value = config[name]
        if param["type"] == "float":
            low = float(param["low"])
            high = float(param["high"])
            encoded.append((float(value) - low) / (high - low or 1.0))
        elif param["type"] == "int":
            low = float(param["low"])
            high = float(param["high"])
            encoded.append((float(value) - low) / (high - low or 1.0))
        elif param["type"] == "categorical":
            choices = list(param["choices"])
            encoded.append(float(choices.index(value)) / max(1, len(choices) - 1))
        else:
            raise ValueError(f"Unsupported parameter type: {param['type']}")
    return encoded


def _random_config(parameters: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for param in parameters:
        if param["type"] == "float":
            low = float(param["low"])
            high = float(param["high"])
            config[param["name"]] = low + (high - low) * rng.random()
        elif param["type"] == "int":
            config[param["name"]] = rng.randint(int(param["low"]), int(param["high"]))
        elif param["type"] == "categorical":
            config[param["name"]] = rng.choice(list(param["choices"]))
        else:
            raise ValueError(f"Unsupported parameter type: {param['type']}")
    return config


if __name__ == "__main__":
    raise SystemExit(main())
