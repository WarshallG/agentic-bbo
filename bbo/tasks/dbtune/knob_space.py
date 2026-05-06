"""Knob JSON -> physical feature vector aligned to surrogate X-name (BenchEnv-style semantics)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numpy.typing import NDArray
except ImportError:  # numpy<1.20 (e.g. Py3.7 surrogate Docker with old sklearn)
    NDArray = np.ndarray


@dataclass(frozen=True)
class _KnobSpec:
    name: str
    kind: str
    default: Any
    min_v: float | None
    max_v: float | None
    enum_values: list[str] | None


def _load_specs(path: Path) -> dict[str, _KnobSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, _KnobSpec] = {}
    for name, d in raw.items():
        if not isinstance(d, dict):
            continue
        t = str(d.get("type", "integer"))
        default = d.get("default")
        min_v = d.get("min")
        max_v = d.get("max")
        enum_values = d.get("enum_values")
        if isinstance(enum_values, list):
            ev = [str(x) for x in enum_values]
        else:
            ev = None
        out[name] = _KnobSpec(
            name=name,
            kind=t,
            default=default,
            min_v=float(min_v) if min_v is not None else None,
            max_v=float(max_v) if max_v is not None else None,
            enum_values=ev,
        )
    return out


class KnobSpaceFromJson:
    """Maps x in [0, 1]^d to physical knob vector for surrogate.predict (order = feature_names)."""

    def __init__(self, knobs_json: str | Path, feature_names: list[str]) -> None:
        self._specs = _load_specs(Path(knobs_json))
        self.feature_names = list(feature_names)

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def _physical_one(self, name: str, u: float) -> float:
        if not (0.0 <= u <= 1.0):
            raise ValueError(f"Normalized coordinate out of [0, 1]: {u}")
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Knob {name!r} not found in JSON")

        if spec.kind == "enum" and spec.enum_values:
            ev = spec.enum_values
            l_n = len(ev)
            if l_n == 1:
                return 0.0
            idx = int(np.floor(u * l_n))
            idx = min(max(idx, 0), l_n - 1)
            return float(idx)

        if spec.min_v is None or spec.max_v is None:
            raise ValueError(f"Knob {name} missing min/max for numeric type")
        lo, hi = spec.min_v, spec.max_v
        v = lo + u * (hi - lo)
        if spec.kind == "integer":
            return float(int(round(v)))
        return float(v)

    def decode(self, x_norm: NDArray[np.float64]) -> NDArray[np.float64]:
        x_norm = np.asarray(x_norm, dtype=np.float64).reshape(-1)
        if x_norm.shape[0] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {x_norm.shape[0]}")
        row = [self._physical_one(n, float(x_norm[i])) for i, n in enumerate(self.feature_names)]
        return np.array(row, dtype=np.float64)


__all__ = ["KnobSpaceFromJson"]
