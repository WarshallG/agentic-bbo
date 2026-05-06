"""Database / knob task registries (unified :mod:`bbo.tasks.dbtune` package).

This mirrors :mod:`bbo.tasks.scientific.registry` for scientific benchmarks: it groups
**offline** sklearn surrogates, dbtune MariaDB/sysbench evaluators, and dbtune surrogate
services in one import path. The top-level :mod:`bbo.tasks.registry` still dispatches
``create_task(...)`` to the right factory.
"""

from __future__ import annotations

from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec, default_knobs_json_path, resolve_bundled_joblib_path
from .http_mariadb_specs import (
    DBTUNE_MARIADB_TASK_IDS,
    DATABASE_TASK_SPECS,
    HTTP_DATABASE_TASK_IDS,
    SYSBENCH_TEST_BY_WORKLOAD,
    HttpDatabaseTaskSpec,
    assets_dir,
    by_task_id,
    default_knobs_path_for_spec,
    is_database_task_id,
)
from .http_surrogate_specs import (
    DBTUNE_SURROGATE_SERVICE_TASK_IDS,
    HTTP_SURROGATE_TASK_IDS,
    is_dbtune_surrogate_service_task_id,
    is_http_surrogate_task_id,
)

__all__ = [
    "DBTUNE_MARIADB_TASK_IDS",
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "DATABASE_TASK_SPECS",
    "HTTP_DATABASE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "SURROGATE_BENCHMARKS",
    "SYSBENCH_TEST_BY_WORKLOAD",
    "SurrogateBenchmarkSpec",
    "HttpDatabaseTaskSpec",
    "assets_dir",
    "by_task_id",
    "default_knobs_json_path",
    "default_knobs_path_for_spec",
    "is_database_task_id",
    "is_dbtune_surrogate_service_task_id",
    "is_http_surrogate_task_id",
    "resolve_bundled_joblib_path",
]
