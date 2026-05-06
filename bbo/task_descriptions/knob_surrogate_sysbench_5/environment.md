# Environment Setup

Install optional surrogate dependencies:

```bash
uv sync --extra dev --extra surrogate
```

Generate the small default `sysbench_5knob_surrogate.joblib` next to the bundled knobs JSON (required once per clone unless you copy a `.joblib` yourself):

```bash
uv run python -m bbo.tasks.dbtune.build_placeholder_surrogate
```

To use a **full** RF checkpoint, **download** `RF_SYSBENCH_5knob.joblib` from the link in `bbo/tasks/dbtune/assets/README.md` and place it in assets.

**Option A — copy into assets (no env var):**

```text
cp /path/where/you/downloaded/RF_SYSBENCH_5knob.joblib \
   <this-repo>/bbo/tasks/dbtune/assets/RF_SYSBENCH_5knob.joblib
```

**Option B — any path:**

```bash
export AGENTIC_BBO_SYSBENCH5_SURROGATE=/absolute/path/to/RF_SYSBENCH_5knob.joblib
```

See also `bbo/tasks/dbtune/assets/README.md`.

No MySQL instance or live Sysbench run is required; evaluation is surrogate-only.
