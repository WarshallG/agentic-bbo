"""dbtune tasks: offline surrogates, MariaDB evaluators, and surrogate services."""

from __future__ import annotations

# --- Offline in-process sklearn surrogates (.joblib) ---
from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec, default_knobs_json_path, resolve_bundled_joblib_path
from .paths import (
    SYSBENCH_5_FEATURE_ORDER,
    bundled_knobs_top5_path,
    bundled_surrogate_sysbench5_path,
)
from .http_surrogate_specs import DBTUNE_SURROGATE_SERVICE_TASK_IDS, HTTP_SURROGATE_TASK_IDS
from .http_surrogate_task import (
    HttpSurrogateKnobTask,
    HttpSurrogateKnobTaskConfig,
    create_dbtune_surrogate_service_task,
    create_http_surrogate_knob_task,
)
from .offline_surrogate_task import (
    SurrogateKnobTask,
    SurrogateKnobTaskConfig,
    create_surrogate_knob_task,
    create_sysbench5_surrogate_task,
)

# --- MariaDB + sysbench evaluator service ---
from .http_mariadb_specs import (
    DBTUNE_MARIADB_TASK_IDS,
    DATABASE_TASK_SPECS,
    HTTP_DATABASE_TASK_IDS,
    SYSBENCH_TEST_BY_WORKLOAD,
    HttpDatabaseTaskSpec,
    by_task_id,
    is_database_task_id,
)
from .http_mariadb_task import (
    HttpDatabaseKnobTask,
    HttpDatabaseKnobTaskConfig,
    create_dbtune_mariadb_task,
    create_http_database_sysbench5_task,
    create_http_database_task,
)
from .cli_mariadb_http import (
    DATABASE_TASK_FAMILY,
    DATABASE_TASK_NAMES,
    DBTUNE_MARIADB_TASK_FAMILY,
    DBTUNE_MARIADB_TASK_NAMES,
    create_database_task_for_registry,
    database_registry_entries,
)

# Public alias (tests and docs)
create_surrogate_task = create_surrogate_knob_task

__all__ = [
    "DATABASE_TASK_FAMILY",
    "DATABASE_TASK_NAMES",
    "DBTUNE_MARIADB_TASK_FAMILY",
    "DBTUNE_MARIADB_TASK_IDS",
    "DBTUNE_MARIADB_TASK_NAMES",
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "DATABASE_TASK_SPECS",
    "HTTP_DATABASE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "HttpDatabaseKnobTask",
    "HttpDatabaseKnobTaskConfig",
    "HttpDatabaseTaskSpec",
    "HttpSurrogateKnobTask",
    "HttpSurrogateKnobTaskConfig",
    "SURROGATE_BENCHMARKS",
    "SYSBENCH_5_FEATURE_ORDER",
    "SYSBENCH_TEST_BY_WORKLOAD",
    "SurrogateBenchmarkSpec",
    "SurrogateKnobTask",
    "SurrogateKnobTaskConfig",
    "bundled_knobs_top5_path",
    "bundled_surrogate_sysbench5_path",
    "by_task_id",
    "create_database_task_for_registry",
    "create_dbtune_mariadb_task",
    "create_dbtune_surrogate_service_task",
    "create_http_database_sysbench5_task",
    "create_http_database_task",
    "create_http_surrogate_knob_task",
    "create_surrogate_knob_task",
    "create_surrogate_task",
    "create_sysbench5_surrogate_task",
    "database_registry_entries",
    "default_knobs_json_path",
    "is_database_task_id",
    "resolve_bundled_joblib_path",
]
