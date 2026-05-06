"""Assets and helpers for interleaved SkyDiscover ↔ BBO integration."""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent


def meta_evaluator_path() -> Path:
    """Path to the SkyDiscover meta-evaluator script (evaluate(program_path))."""
    return ASSETS_DIR / "meta_evaluator.py"


def initial_strategy_template_path() -> Path:
    """Path to the seed strategy module copied / evolved by SkyDiscover."""
    return ASSETS_DIR / "initial_strategy.py"
