"""Run the SkyDiscover-interleaved Branin demo without LLM (offline baseline).

Uses ``skydiscover_runner=False``: interleave cadence refreshes the bundled seed
``suggest_next_config`` only (no SkyDiscover Runner / API calls).
"""

from __future__ import annotations

import json

from bbo.run import run_single_experiment


if __name__ == "__main__":
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="skydiscover_interleaved",
        seed=7,
        max_evaluations=12,
        skydiscover_interleave_every=4,
        skydiscover_round_iterations=1,
        skydiscover_runner=False,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
