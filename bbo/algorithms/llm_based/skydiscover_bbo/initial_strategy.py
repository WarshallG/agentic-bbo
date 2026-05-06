"""Seed strategy: deterministic random proposer (evolved by SkyDiscover in runner mode).

This module must expose ``suggest_next_config`` with the BBO contract signature.
"""

from __future__ import annotations

import math
import random
from typing import Any


def suggest_next_config(
    *,
    history: list[tuple[dict[str, Any], float]],
    parameter_specs: list[dict[str, Any]],
    objective_direction: str,
    seed: int,
    trial_index: int,
) -> dict[str, Any]:
    """Propose the next configuration from parameter specs (deterministic in seed + trial_index)."""
    rng = random.Random(_mix_seed(seed, trial_index))
    out: dict[str, Any] = {}
    for spec in parameter_specs:
        name = spec["name"]
        typ = spec["type"]
        if typ == "float":
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log"):
                out[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
            else:
                out[name] = rng.uniform(low, high)
        elif typ == "int":
            out[name] = rng.randint(int(spec["low"]), int(spec["high"]))
        elif typ == "categorical":
            choices = list(spec["choices"])
            out[name] = rng.choice(choices)
        else:
            raise ValueError(f"Unknown parameter type {typ!r} for {name!r}")
    return out


def _mix_seed(seed: int, trial_index: int) -> int:
    return (int(seed) * 100_003 + int(trial_index)) % (2**31)
