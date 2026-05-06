"""HTTP API: apply MySQL/MariaDB knobs, restart server, run sysbench, return TPS."""

from __future__ import annotations

import re
import subprocess
import threading
from typing import Any, Final

from flask import Flask, jsonify, request

# --- 配置常量（与 entrypoint 中的 root 密码一致） ---
TUNER_CONF: Final = "/etc/mysql/mariadb.conf.d/99-tuner.cnf"
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DB = "sbtest"
SYSBENCH_TABLES = 10
SYSBENCH_TABLE_SIZE = 100_000
SYSBENCH_TIME_SEC = 10

# 与 ``bbo/tasks/dbtune/http_mariadb_specs.py`` 中 API ``workload`` 字段一致
ALLOWED_WORKLOADS: dict[str, str] = {
    "read_only": "oltp_read_only",
    "write_only": "oltp_write_only",
    "read_write": "oltp_read_write",
    "point_select": "oltp_point_select",
}
DEFAULT_WORKLOAD_KEY = "read_write"

# 同一容器内仅允许串行评估，避免并发改配置/重启冲突
_eval_lock = threading.Lock()

app = Flask(__name__)


def _write_tuner_cnf(knobs: dict[str, str]) -> None:
    """将 Agent 下发的 knob 写入独立 drop-in 配置，不污染系统其它 cnf。"""
    lines = ["[mysqld]\n"]
    for key, value in knobs.items():
        lines.append(f"{key} = {value}\n")
    content = "".join(lines)
    with open(TUNER_CONF, "w", encoding="utf-8") as f:
        f.write(content)


def _restart_mariadb() -> None:
    subprocess.run(
        ["service", "mariadb", "restart"],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _run_sysbench(test_name: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "sysbench",
        "--db-driver=mysql",
        f"--mysql-user={MYSQL_USER}",
        f"--mysql-password={MYSQL_PASSWORD}",
        f"--mysql-db={MYSQL_DB}",
        f"--tables={SYSBENCH_TABLES}",
        f"--table-size={SYSBENCH_TABLE_SIZE}",
        f"--time={SYSBENCH_TIME_SEC}",
        test_name,
        "run",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=SYSBENCH_TIME_SEC + 120,
    )


def _parse_throughput(stdout: str) -> float:
    """提取通用 ``(... per sec)`` 指标；优先 transactions，再尝试 queries（point select 等）。"""
    text = stdout or ""
    m = re.search(
        r"transactions:\s+\d+\s+\(([\d.]+)\s+per sec",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    m2 = re.search(
        r"queries?:\s+\d+\s+\(([\d.]+)\s+per sec",
        text,
        re.IGNORECASE,
    )
    if m2:
        return float(m2.group(1))
    m3 = re.search(r"\(([\d.]+)\s+per sec\.\s*\)\s*$", text, re.MULTILINE)
    if m3:
        return float(m3.group(1))
    return 0.0


@app.route("/health", methods=["GET"])
def health() -> Any:
    """轻量探活，便于编排与 Agent 预检。"""
    return jsonify({"status": "ok"})


@app.route("/evaluate", methods=["POST"])
def evaluate() -> Any:
    payload = request.get_json(silent=True) or {}
    knobs_raw = payload.get("knobs")
    if not isinstance(knobs_raw, dict) or not knobs_raw:
        return jsonify({"error": "Missing or empty 'knobs' object"}), 400

    wk = payload.get("workload", DEFAULT_WORKLOAD_KEY)
    wk = str(wk).strip()
    if wk not in ALLOWED_WORKLOADS:
        return (
            jsonify(
                {
                    "error": "Invalid workload",
                    "allowed": list(ALLOWED_WORKLOADS.keys()),
                    "got": wk,
                }
            ),
            400,
        )
    test_name = ALLOWED_WORKLOADS[wk]
    knobs: dict[str, str] = {str(k): str(v) for k, v in knobs_raw.items()}

    with _eval_lock:
        try:
            _write_tuner_cnf(knobs)
            _restart_mariadb()
            result = _run_sysbench(test_name)
            if result.returncode != 0:
                return (
                    jsonify(
                        {
                            "y": 0.0,
                            "status": "error",
                            "message": "sysbench failed",
                            "workload": wk,
                            "stderr": (result.stderr or "")[-4000:],
                        }
                    ),
                    500,
                )
            tps = _parse_throughput(result.stdout or "")
            return jsonify(
                {
                    "y": tps,
                    "status": "success",
                    "tps": tps,
                    "workload": wk,
                }
            )
        except subprocess.TimeoutExpired as e:
            return (
                jsonify(
                    {
                        "y": 0.0,
                        "status": "error",
                        "message": f"timeout: {e}",
                        "workload": wk,
                    }
                ),
                500,
            )
        except OSError as e:
            return (
                jsonify(
                    {
                        "y": 0.0,
                        "status": "error",
                        "message": str(e),
                        "workload": wk,
                    }
                ),
                500,
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
