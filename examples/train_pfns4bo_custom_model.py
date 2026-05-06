from __future__ import annotations

from pathlib import Path

from bbo.algorithms.model_based.pfns4bo_training import train_custom_pfn_model
from bbo.run import run_single_experiment


def main() -> None:
    artifact = train_custom_pfn_model(
        output_dir=Path("artifacts") / "custom_pfn_models",
        prior="hebo",
        max_features=32,
        model_name="hebo_smoke.pt",
        device="cpu:0",
        epochs=1,
        steps_per_epoch=4,
        batch_size=8,
        seq_len=32,
    )
    print(f"trained model: {artifact.model_path}")

    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="pfns4bo_custom",
        seed=7,
        max_evaluations=8,
        results_root=Path("artifacts") / "pfns_variants_demo",
        pfns_device="cpu:0",
        pfns_pool_size=256,
        pfns_acquisition="ei",
        pfns_custom_model_path=artifact.model_path,
        generate_plots=False,
    )
    print(summary)


if __name__ == "__main__":
    main()
