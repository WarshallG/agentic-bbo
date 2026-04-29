"""Serialize SearchSpace into JSON-friendly specs for generated strategies."""

from __future__ import annotations

from typing import Any

from ....core.space import CategoricalParam, FloatParam, IntParam, SearchSpace


def parameter_specs_to_search_space(specs: list[dict[str, Any]]) -> SearchSpace:
    """Rebuild a structured ``SearchSpace`` from JSON-ish parameter specs."""

    params: list[FloatParam | IntParam | CategoricalParam] = []
    for spec in specs:
        name = spec["name"]
        typ = spec["type"]
        if typ == "float":
            params.append(
                FloatParam(
                    name=name,
                    low=float(spec["low"]),
                    high=float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            )
        elif typ == "int":
            params.append(
                IntParam(
                    name=name,
                    low=int(spec["low"]),
                    high=int(spec["high"]),
                    log=bool(spec.get("log", False)),
                )
            )
        elif typ == "categorical":
            params.append(
                CategoricalParam(
                    name=name,
                    choices=tuple(spec["choices"]),
                )
            )
        else:
            raise TypeError(f"Unsupported parameter type {typ!r} for `{name}`.")
    return SearchSpace(params)


def search_space_to_parameter_specs(space: SearchSpace) -> list[dict[str, Any]]:
    """Convert a SearchSpace to a list of dict specs for suggest_next_config."""
    specs: list[dict[str, Any]] = []
    for param in space:
        if isinstance(param, FloatParam):
            specs.append(
                {
                    "name": param.name,
                    "type": "float",
                    "low": param.low,
                    "high": param.high,
                    "log": param.log,
                }
            )
        elif isinstance(param, IntParam):
            specs.append(
                {
                    "name": param.name,
                    "type": "int",
                    "low": param.low,
                    "high": param.high,
                    "log": param.log,
                }
            )
        elif isinstance(param, CategoricalParam):
            specs.append(
                {
                    "name": param.name,
                    "type": "categorical",
                    "choices": list(param.choices),
                }
            )
        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
    return specs
