# `bbo.tasks.dbtune` — database knob task family

This package groups **all database-related knob benchmarks** in one place, in the same spirit as
`bbo.tasks.scientific`: a small `registry.py` documents what exists, and co-located assets live under
clear subfolders.

## Layout

| Area | Role |
|------|------|
| `registry.py` | Re-exports catalog metadata for offline surrogates, MariaDB task specs, and surrogate-service id maps. |
| `catalog.py` | Offline `*.joblib` benchmark specs (`SURROGATE_BENCHMARKS`). |
| `http_mariadb_specs.py` | Eight real **MariaDB + sysbench** dbtune tasks (`DBTUNE_MARIADB_TASK_IDS`). |
| `http_mariadb_task.py` | Task implementation: `HttpDatabaseKnobTask`. |
| `offline_surrogate_task.py` | In-process sklearn surrogate: `SurrogateKnobTask`. |
| `http_surrogate_task.py` | Remote evaluator service (Python 3.7 Docker) for the same surrogates. |
| `cli_*.py` | Hooks for `bbo.tasks.registry` / `python -m bbo.run` (no changes to `bbo.run` needed for new task ids). |
| `assets/` | Shared `knobs_*.json` and downloaded `*.joblib` (large files are not committed; see `assets/README.md`). |
| `docker_mariadb/` | Image for the **live** MariaDB + sysbench evaluator (Flask API). |
| `docker_surrogate/` | Image for **offline** sklearn inference via JSON (isolated old numpy/sklearn). |
| `gen_task_markdown.py` | One-off generator for `bbo/task_descriptions/knob_http_mariadb_sysbench_*/` packs. |

## Import surface

User code typically uses the stable exports from `bbo.tasks` / `bbo.tasks.registry` (e.g.
`create_task("knob_http_surrogate_sysbench_5")` or the MariaDB `knob_http_mariadb_sysbench_*` ids). In-process
`create_surrogate_knob_task("knob_surrogate_sysbench_5", ...)` remains available but is not registered on
`python -m bbo.run`. For a
**direct** import, prefer:

```python
from bbo.tasks.dbtune import create_dbtune_mariadb_task, create_surrogate_knob_task
```

## See also

- `bbo/tasks/scientific/` — same “family + registry + data/” pattern for non-database scientific benchmarks.
