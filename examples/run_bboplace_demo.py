"""Run the BBOPlace-Bench macro placement task via the HTTP evaluator service.

Prerequisite (start evaluator service first):
    docker pull gaozhixuan/bboplace-bench
    docker run --rm -p 8070:8080 gaozhixuan/bboplace-bench

Example:
    uv run python examples/run_bboplace_demo.py --benchmark adaptec1 --n-macro 32 --seed 1 --max-evaluations 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bbo.run import run_single_experiment
from bbo.tasks.bboplace.task import BBOPLACE_TASK_KEY, default_bboplace_definition


def main() -> None:
    parser = argparse.ArgumentParser(description="BBOPlace-Bench demo runner.")
    parser.add_argument("--benchmark", default="adaptec1", help="Benchmark name (e.g., adaptec1).")
    parser.add_argument("--n-macro", type=int, default=32, help="Number of macros to place.")
    parser.add_argument("--seed", type=int, default=1, help="Seed forwarded to evaluator payload.")
    parser.add_argument(
        "--algorithm",
        default="random_search",
        help="Baseline optimizer (numeric-only).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=1,
        help="Number of evaluator calls (1 for a quick smoke test).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume into the same run directory.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plots (faster for smoke tests).",
    )
    parser.add_argument(
        "--sigma-fraction",
        type=float,
        default=0.18,
        help="CMA-ES step size (fraction of bounds).",
    )
    parser.add_argument("--popsize", type=int, default=6, help="CMA-ES population size.")
    args = parser.parse_args()

    definition = default_bboplace_definition(
        benchmark=args.benchmark,
        n_macro=args.n_macro,
        bench_seed=args.seed,
    )

    results_root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "demo"
        / f"bboplace_{args.benchmark}_nmacro{args.n_macro}"
    )
    summary = run_single_experiment(
        task_name=BBOPLACE_TASK_KEY,
        algorithm_name=args.algorithm,
        seed=args.seed,
        max_evaluations=args.max_evaluations,
        results_root=results_root,
        resume=bool(args.resume),
        generate_plots=not bool(args.no_plots),
        sigma_fraction=args.sigma_fraction,
        popsize=args.popsize,
        task_kwargs={"definition": definition},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
