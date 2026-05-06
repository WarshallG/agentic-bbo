"""Load and validate generated strategy modules (suggest_next_config entry point)."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

SUGGEST_NEXT_CONFIG_NAME = "suggest_next_config"


def load_suggest_next_config(path: Path) -> Callable[..., dict[str, Any]]:
    """Load ``suggest_next_config`` from a Python file path."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Strategy module not found: {resolved}")

    module_name = f"_bbo_generated_strategy_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {resolved}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, SUGGEST_NEXT_CONFIG_NAME, None)
    if not callable(fn):
        raise AttributeError(
            f"{resolved} must define a callable `{SUGGEST_NEXT_CONFIG_NAME}`"
        )
    return fn
