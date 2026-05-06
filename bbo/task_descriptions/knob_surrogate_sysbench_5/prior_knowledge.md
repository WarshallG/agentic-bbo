# Prior Knowledge

- Five Sysbench-related knobs are active (see task metadata `feature_order`).
- The surrogate predicts throughput (TPS) from physical knob vectors (offline RF bundle; file obtained via ``bbo/tasks/dbtune/assets/README.md``).
- This task is intended for optimizer comparison; absolute numbers depend on which `.joblib` file is used.
