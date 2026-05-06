"""Static definitions for the eight dbtune MariaDB/sysbench tasks (2× knobs × 4 workloads)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --- Knob space assets (under ``bbo/tasks/dbtune/assets/``) ---
_TOP5_JSON: Final[str] = "knobs_SYSBENCH_top5.json"
_ALL197_JSON: Final[str] = "knobs_mysql_all_197.json"

# --- API / Docker server workload keys (JSON field ``workload``) ---
WORKLOAD_READ_ONLY: Final[str] = "read_only"
WORKLOAD_WRITE_ONLY: Final[str] = "write_only"
WORKLOAD_READ_WRITE: Final[str] = "read_write"
WORKLOAD_POINT_SELECT: Final[str] = "point_select"

# Maps API workload key -> sysbench test name (``sysbench <name> run``)
SYSBENCH_TEST_BY_WORKLOAD: dict[str, str] = {
    WORKLOAD_READ_ONLY: "oltp_read_only",
    WORKLOAD_WRITE_ONLY: "oltp_write_only",
    WORKLOAD_READ_WRITE: "oltp_read_write",
    WORKLOAD_POINT_SELECT: "oltp_point_select",
}


@dataclass(frozen=True)
class HttpDatabaseTaskSpec:
    """One registered database benchmark task (task_id + knob JSON + sysbench workload)."""

    task_id: str
    workload_key: str
    """Must be a key in ``SYSBENCH_TEST_BY_WORKLOAD``."""
    knob_asset_filename: str
    """File name under ``dbtune/assets/`` (e.g. ``_TOP5_JSON``)."""
    display_name: str
    short_label_en: str
    short_label_zh: str


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def default_knobs_path_for_spec(spec: HttpDatabaseTaskSpec) -> Path:
    return assets_dir() / spec.knob_asset_filename


def is_database_task_id(task_id: str) -> bool:
    return task_id in DATABASE_TASK_SPECS


def by_task_id(task_id: str) -> HttpDatabaseTaskSpec:
    if task_id not in DATABASE_TASK_SPECS:
        known = ", ".join(sorted(DATABASE_TASK_SPECS))
        raise KeyError(f"Unknown database task_id `{task_id}`. Known: {known}")
    return DATABASE_TASK_SPECS[task_id]


# --- Eight tasks: (read_only|write_only|read_write|point_select) × (5|all) ---
DATABASE_TASK_SPECS: dict[str, HttpDatabaseTaskSpec] = {
    "knob_http_mariadb_sysbench_read_only_5": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_read_only_5",
        workload_key=WORKLOAD_READ_ONLY,
        knob_asset_filename=_TOP5_JSON,
        display_name="MariaDB/sysbench read-only, 5 knobs",
        short_label_en="read-only, 5 knobs",
        short_label_zh="只读、5 旋钮",
    ),
    "knob_http_mariadb_sysbench_write_only_5": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_write_only_5",
        workload_key=WORKLOAD_WRITE_ONLY,
        knob_asset_filename=_TOP5_JSON,
        display_name="MariaDB/sysbench write-only, 5 knobs",
        short_label_en="write-only, 5 knobs",
        short_label_zh="只写、5 旋钮",
    ),
    "knob_http_mariadb_sysbench_read_write_5": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_read_write_5",
        workload_key=WORKLOAD_READ_WRITE,
        knob_asset_filename=_TOP5_JSON,
        display_name="MariaDB/sysbench read/write, 5 knobs",
        short_label_en="read/write, 5 knobs",
        short_label_zh="读写在同一事务里、5 旋钮",
    ),
    "knob_http_mariadb_sysbench_point_select_5": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_point_select_5",
        workload_key=WORKLOAD_POINT_SELECT,
        knob_asset_filename=_TOP5_JSON,
        display_name="MariaDB/sysbench point select, 5 knobs",
        short_label_en="point select, 5 knobs",
        short_label_zh="点查询、5 旋钮",
    ),
    "knob_http_mariadb_sysbench_read_only_all": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_read_only_all",
        workload_key=WORKLOAD_READ_ONLY,
        knob_asset_filename=_ALL197_JSON,
        display_name="MariaDB/sysbench read-only, full knob list",
        short_label_en="read-only, full knob list",
        short_label_zh="只读、全量旋钮",
    ),
    "knob_http_mariadb_sysbench_write_only_all": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_write_only_all",
        workload_key=WORKLOAD_WRITE_ONLY,
        knob_asset_filename=_ALL197_JSON,
        display_name="MariaDB/sysbench write-only, full knob list",
        short_label_en="write-only, full knob list",
        short_label_zh="只写、全量旋钮",
    ),
    "knob_http_mariadb_sysbench_read_write_all": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_read_write_all",
        workload_key=WORKLOAD_READ_WRITE,
        knob_asset_filename=_ALL197_JSON,
        display_name="MariaDB/sysbench read/write, full knob list",
        short_label_en="read/write, full knob list",
        short_label_zh="读写、全量旋钮",
    ),
    "knob_http_mariadb_sysbench_point_select_all": HttpDatabaseTaskSpec(
        task_id="knob_http_mariadb_sysbench_point_select_all",
        workload_key=WORKLOAD_POINT_SELECT,
        knob_asset_filename=_ALL197_JSON,
        display_name="MariaDB/sysbench point select, full knob list",
        short_label_en="point select, full knob list",
        short_label_zh="点查询、全量旋钮",
    ),
}

HTTP_DATABASE_TASK_IDS: tuple[str, ...] = tuple(sorted(DATABASE_TASK_SPECS))
DBTUNE_MARIADB_TASK_IDS: tuple[str, ...] = HTTP_DATABASE_TASK_IDS

__all__ = [
    "DBTUNE_MARIADB_TASK_IDS",
    "DATABASE_TASK_SPECS",
    "HTTP_DATABASE_TASK_IDS",
    "HttpDatabaseTaskSpec",
    "SYSBENCH_TEST_BY_WORKLOAD",
    "WORKLOAD_POINT_SELECT",
    "WORKLOAD_READ_ONLY",
    "WORKLOAD_READ_WRITE",
    "WORKLOAD_WRITE_ONLY",
    "assets_dir",
    "by_task_id",
    "default_knobs_path_for_spec",
    "is_database_task_id",
]
