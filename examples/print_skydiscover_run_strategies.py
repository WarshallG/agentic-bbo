"""Print strategy artifacts from a skydiscover_interleaved run directory.

Finds:

- ``generated/strategy.py`` — current merged strategy used by ``ask()`` (overwritten each refresh).
- ``generated/skydiscover_round_*/best/best_program.py`` — snapshots from SkyDiscover Runner rounds (if any).

Usage:

    uv run python examples/print_skydiscover_run_strategies.py runs/demo/branin_demo/skydiscover_interleaved_topk/seed_7

    uv run python examples/print_skydiscover_run_strategies.py \\
        runs/demo/branin_demo/skydiscover_interleaved_openevolve/seed_7 \\
        --list-only

    # Absolute path works too:
    uv run python examples/print_skydiscover_run_strategies.py /path/to/.../seed_7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _strategy_artifacts(run_dir: Path) -> list[Path]:
    """Return sorted paths to print: main strategy.py then each round best_program.py."""
    run_dir = run_dir.expanduser().resolve()
    gen = run_dir / "generated"
    if not gen.is_dir():
        return []

    out: list[Path] = []
    strat = gen / "strategy.py"
    if strat.is_file():
        out.append(strat)

    rounds = sorted(gen.glob("skydiscover_round_*"))
    for rd in rounds:
        best_py = rd / "best" / "best_program.py"
        if best_py.is_file():
            out.append(best_py)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print strategy.py and per-round best_program.py from a skydiscover run.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory (.../seed_N) containing generated/",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print paths, not file contents.",
    )
    args = parser.parse_args()
    artifacts = _strategy_artifacts(args.run_dir)

    if not artifacts:
        print(
            f"No strategy artifacts found under {args.run_dir.resolve()}/generated/",
            file=sys.stderr,
        )
        print(
            "Expected: generated/strategy.py and/or generated/skydiscover_round_*/best/best_program.py",
            file=sys.stderr,
        )
        return 1

    if args.list_only:
        for p in artifacts:
            print(p)
        return 0

    sep = "=" * 72
    for i, path in enumerate(artifacts):
        if i > 0:
            print()
        print(sep)
        print(path)
        print(sep)
        text = path.read_text(encoding="utf-8")
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
