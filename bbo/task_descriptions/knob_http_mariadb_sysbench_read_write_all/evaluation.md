# Evaluation Protocol

- **Data source (inside the container):** a sysbench `prepare` run seeds `sbtest` with fixed `--tables` / `--table-size` in `server.py` (the entrypoint may call `oltp_read_write prepare` once, which is compatible with the other bundled **oltp_*** `run` tests in typical sysbench builds).
- **Metrics:** primary = `throughput` == parsed **per-second** figure from the sysbench text report for **``oltp_read_write``**; if `transactions` is missing for a script, the server falls back to a `queries` line when present.
- **Request contract:** the Python `HttpDatabaseKnobTask` sends:
  - `workload: "read_write"` (server maps to **``oltp_read_write``**)
  - `knobs: {name: "value"}` (strings acceptable to MariaDB for those variables)
- **Reproducibility:** fix seed for the *optimizer* (BBO `seed`) and keep machine load low; the DB is still a noisy environment.

When reporting results, list **image digest / git commit**, this `task_id`, and the knob JSON filename.
