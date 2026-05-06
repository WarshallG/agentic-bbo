"""Knob JSON + normalized vector -> MariaDB string knobs for HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .knob_space import KnobSpaceFromJson


def feature_order_by_rank(knobs_json: str | Path) -> tuple[str, ...]:
    """Stable order: ascending ``important_rank``, then name (matches surrogate-style ordering)."""
    path = Path(knobs_json)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    ranked: list[tuple[int, str]] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        rank = int(spec.get("important_rank", 999))
        ranked.append((rank, str(name)))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return tuple(name for _, name in ranked)


def physical_to_mariadb_strings(
    knobs_json: str | Path,
    feature_names: tuple[str, ...],
    physical: np.ndarray,
) -> dict[str, str]:
    """Map decoded physical vector to ``my.cnf``-style string values."""
    path = Path(knobs_json)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    vec = np.asarray(physical, dtype=np.float64).reshape(-1)
    if vec.shape[0] != len(feature_names):
        raise ValueError(f"Expected dim {len(feature_names)}, got {vec.shape[0]}")
    out: dict[str, str] = {}
    for i, name in enumerate(feature_names):
        v = float(vec[i])
        spec = raw.get(name)
        if not isinstance(spec, dict):
            raise KeyError(f"Knob {name!r} missing from JSON")
        t = str(spec.get("type", "integer"))
        if t == "enum":
            ev = [str(x) for x in (spec.get("enum_values") or [])]
            if not ev:
                raise ValueError(f"Enum knob {name} has no enum_values")
            idx = int(round(v))
            idx = max(0, min(idx, len(ev) - 1))
            out[name] = ev[idx]
        elif t == "integer":
            out[name] = str(int(round(v)))
        else:
            out[name] = str(v)
    return out


def build_knob_space(knobs_json: str | Path, feature_names: tuple[str, ...]) -> KnobSpaceFromJson:
    return KnobSpaceFromJson(knobs_json, list(feature_names))


__all__ = ["build_knob_space", "feature_order_by_rank", "physical_to_mariadb_strings"]
