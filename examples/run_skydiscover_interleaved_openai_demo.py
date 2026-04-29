"""Run SkyDiscover-interleaved BBO with a real LLM (OpenAI-compatible credentials).

This uses ``run_single_experiment`` with ``skydiscover_runner=True``. SkyDiscover
loads API keys the same way as standalone SkyDiscover (not via LLAMBO/OPRO flags).

Prerequisites:
    uv sync --extra dev --extra skydiscover

Setup:
    export OPENAI_API_KEY="sk-..."
    # Optional: compatible / proxy endpoints
    export OPENAI_BASE_URL="https://api.openai.com/v1"

Run:
    python examples/run_skydiscover_interleaved_openai_demo.py
    python examples/run_skydiscover_interleaved_openai_demo.py --search-type adaevolve
    python examples/run_skydiscover_interleaved_openai_demo.py --preset adaevolve
    python examples/run_skydiscover_interleaved_openai_demo.py --preset evox
    python examples/run_skydiscover_interleaved_openai_demo.py --preset gepa_native
    python examples/run_skydiscover_interleaved_openai_demo.py --preset openevolve_native

Method-specific hyperparameters (AdaEvolve islands, GEPA epsilon/merge, EvoX meta knobs, …)
live in SkyDiscover YAML under ``search.database`` (and for EvoX also ``search.switch_interval``,
``search.share_llm``). They are **not** mirrored as individual ``bbo.run`` flags: pass a YAML via
``--config`` or use ``--preset`` to load a small example from ``examples/skydiscover_configs/``.

Full field lists match the dataclasses in vendored SkyDiscover ``skydiscover/config.py``:
``AdaEvolveDatabaseConfig``, ``EvoxDatabaseConfig`` / ``SearchConfig``, ``GEPANativeDatabaseConfig``, …
Upstream reference configs: ``bbo/algorithms/llm_based/skydiscover/configs/adaevolve.yaml``,
``.../evox.yaml``, ``.../openevolve_native.yaml``.

Important: the YAML ``search.type`` should match ``--search-type``. If they differ, SkyDiscover’s
``apply_overrides`` may replace ``search.database`` with fresh defaults for the CLI type, dropping
custom hyperparameters loaded from a mismatched YAML.

Notes:
    - There is no ``alphaevolve`` search id in this vendored SkyDiscover; use ``adaevolve``.
    - Some backends (e.g. full GEPA extras) may need additional SkyDiscover optional dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from bbo.run import run_single_experiment

_PRESET_DIR = Path(__file__).resolve().parent / "skydiscover_configs"
_PRESET_TO_SEARCH = {
    "adaevolve": "adaevolve",
    "evox": "evox",
    "gepa_native": "gepa_native",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SkyDiscover interleaved demo with OpenAI-compatible credentials.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(_PRESET_TO_SEARCH.keys()),
        default=None,
        help=(
            "Load examples/skydiscover_configs/<preset>_bbo.yaml (method-specific hyperparameters). "
            "If --search-type is still the default topk, it is aligned to this preset."
        ),
    )
    parser.add_argument(
        "--search-type",
        default="topk",
        choices=[
            "topk",
            "adaevolve",
            "evox",
            "gepa",
            "gepa_native",
            "best_of_n",
            "beam_search",
            "openevolve_native",
            "openevolve",
            "shinkaevolve",
            "claude_code",
        ],
        help="SkyDiscover search.type (same vocabulary as the skydiscover CLI --search).",
    )
    parser.add_argument("--task", default="branin_demo", help="BBO task name.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-evaluations", type=int, default=16)
    parser.add_argument("--interleave-every", type=int, default=4)
    parser.add_argument("--round-iterations", type=int, default=3)
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name injected when the SkyDiscover config has no llm.models.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a SkyDiscover YAML (overrides --preset file when both are set).",
    )
    return parser.parse_args()


def _apply_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    expected = _PRESET_TO_SEARCH[args.preset]
    preset_file = _PRESET_DIR / f"{args.preset}_bbo.yaml"
    if not preset_file.is_file():
        raise FileNotFoundError(f"Preset file missing: {preset_file}")

    if args.config is None:
        args.config = str(preset_file)

    using_preset_file = Path(args.config).resolve() == preset_file.resolve()
    if using_preset_file and args.search_type == "topk":
        args.search_type = expected
    elif using_preset_file and args.search_type != expected:
        warnings.warn(
            f"--preset {args.preset!r} uses YAML with search.type={expected!r}, "
            f"but --search-type is {args.search_type!r}; SkyDiscover may replace search.database.",
            UserWarning,
            stacklevel=1,
        )


if __name__ == "__main__":
    args = _parse_args()
    _apply_preset(args)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set OPENAI_API_KEY before running this demo.\n"
            "Example: export OPENAI_API_KEY='sk-...'\n"
            "Install SkyDiscover extra: uv sync --extra dev --extra skydiscover"
        )

    summary = run_single_experiment(
        task_name=args.task,
        algorithm_name="skydiscover_interleaved",
        seed=args.seed,
        max_evaluations=args.max_evaluations,
        skydiscover_interleave_every=args.interleave_every,
        skydiscover_round_iterations=args.round_iterations,
        skydiscover_config_path=args.config,
        skydiscover_runner=True,
        skydiscover_search_type=args.search_type,
        skydiscover_model=args.model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
