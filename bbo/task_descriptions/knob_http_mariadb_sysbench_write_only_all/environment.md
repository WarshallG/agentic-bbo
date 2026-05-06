# Environment

## Shared Docker build (all eight dbtune MariaDB/sysbench tasks)

```bash
cd bbo/tasks/dbtune/docker_mariadb
docker build -t agentbbo-http-mariadb-eval:v1 .
docker rm -f agentbbo_http_mariadb_eval 2>/dev/null
docker run -d --name agentbbo_http_mariadb_eval -p 8080:8080 agentbbo-http-mariadb-eval:v1
```

After pulling changes to `server.py` (notably the `workload` field), **rebuild** the image so every task can select its sysbench test.

## Client-side environment (Python)

| Variable | Role |
|----------|------|
| `AGENTBBO_HTTP_EVAL_BASE_URL` | Base URL, default `http://127.0.0.1:8080` |
| `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC` | **Per-POST** timeout (seconds), default `300` |

## This task

| Field | Value |
|------|--------|
| `task_id` | `knob_http_mariadb_sysbench_write_only_all` |
| `workload` (JSON) | `write_only` -> `oltp_write_only` |
| Knob JSON (default) | `bbo/tasks/dbtune/assets/knobs_mysql_all_197.json` |

Health check: `GET /health` on the same base URL.
