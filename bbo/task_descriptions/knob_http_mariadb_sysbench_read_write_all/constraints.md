# Constraints

- **No concurrent in-flight `POST /evaluate` calls** against the same server instance: the container serializes work with a lock.
- **Per-evaluation** wall time can be minutes: set `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC` (or a larger client timeout) accordingly.
- **Fixed workload** for this `task_id`: the sysbench command is always **``oltp_read_write``**; you cannot change it from the BBO search space.
- **Fixed knob schema** for this `task_id` unless the task is constructed with an explicit `knobs_json_path` override (not exposed in `bbo.run` by default).
- The evaluator is **not** a security sandbox: do not point it at production databases; use an isolated host or VM.
