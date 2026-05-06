"""Build a compact sklearn RF surrogate for Sysbench 5-knob (repo self-containment).

Large checkpoints from the original project may be omitted; this writes a small
placeholder so ``bbo.tasks.dbtune`` works without a full-size ``RF_SYSBENCH_5knob.joblib``.

Usage (from repo root)::

    uv sync --extra surrogate
    uv run python -m bbo.tasks.dbtune.build_placeholder_surrogate
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from .paths import SYSBENCH_5_FEATURE_ORDER, bundled_knobs_top5_path

    # 仓库根目录：bbo/tasks/dbtune/ → 上溯三级
    repo_root = Path(__file__).resolve().parents[3]
    out = repo_root / "bbo" / "tasks" / "dbtune" / "assets" / "sysbench_5knob_surrogate.joblib"
    names = list(SYSBENCH_5_FEATURE_ORDER)
    d = len(names)
    rng = np.random.default_rng(0)
    n = max(64, d * 16)
    x = rng.random((n, d))
    y = rng.standard_normal(n)
    model = RandomForestRegressor(n_estimators=32, max_depth=8, random_state=0, n_jobs=-1)
    model.fit(x, y)
    payload = {"model": model, "X-name": names}
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    knobs = bundled_knobs_top5_path()
    print(f"Wrote surrogate: {out.resolve()}")
    print(f"Knobs JSON:      {knobs.resolve()} (exists={knobs.is_file()})")


if __name__ == "__main__":
    main()
