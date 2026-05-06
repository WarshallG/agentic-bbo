# Background

This benchmark optimizes a **surrogate model** of database throughput under Sysbench-style workloads.
The RF checkpoint (`.joblib`) is **not** shipped in git: download the released file and install it as described in ``bbo/tasks/dbtune/assets/README.md`` (shared download link). This repository **implements** the evaluation path (joblib load + knob JSON decoding) only.

The optimizer proposes normalized knob coordinates in `[0, 1]^d`; the task decodes them to physical MySQL knob values and returns the surrogate's predicted objective (higher is better).
