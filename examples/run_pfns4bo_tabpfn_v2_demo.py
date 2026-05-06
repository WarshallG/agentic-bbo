from __future__ import annotations

from pathlib import Path

from bbo.run import run_single_experiment


def main() -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="pfns4bo_tabpfn_v2",
        seed=7,
        max_evaluations=8,
        results_root=Path("artifacts") / "pfns_variants_demo",
        pfns_device="cpu:0",
        pfns_pool_size=256,
        pfns_acquisition="ei",
        pfns_tabpfn_n_estimators=8,
        generate_plots=False,
    )
    print(summary)


if __name__ == "__main__":
    main()
