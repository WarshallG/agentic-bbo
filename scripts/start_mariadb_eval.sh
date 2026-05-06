#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_path="${repo_root}/bbo/tasks/dbtune/docker_mariadb/server.py"

root_pw="${DBTUNE_MYSQL_ROOT_PASSWORD:-123456}"
db_name="${DBTUNE_MYSQL_DB:-sbtest}"
tables="${DBTUNE_SYSBENCH_TABLES:-10}"
table_size="${DBTUNE_SYSBENCH_TABLE_SIZE:-100000}"

mysql_cli="$(command -v mariadb || command -v mysql || true)"
if [[ -z "${mysql_cli}" ]]; then
  echo "mariadb/mysql client not found in PATH" >&2
  exit 1
fi

if ! command -v sysbench >/dev/null 2>&1; then
  echo "sysbench not found in PATH" >&2
  exit 1
fi

if ! uv run python - <<'PY' >/dev/null 2>&1
import flask
PY
then
  echo "Flask is not available in the current Python environment." >&2
  exit 1
fi

echo "1. Starting MariaDB..."
service mariadb start

echo "2. Initializing database and root password..."
"${mysql_cli}" -u root -e "CREATE DATABASE IF NOT EXISTS ${db_name};" || true
"${mysql_cli}" -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '${root_pw}'; FLUSH PRIVILEGES;" || true

echo "3. Checking sysbench tables..."
raw_count="$("${mysql_cli}" -u root -p"${root_pw}" -N -e \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='${db_name}';" \
  2>/dev/null | tr -d '[:space:]' || true)"
table_count="${raw_count:-0}"

if [[ "${table_count}" -eq 0 ]]; then
  echo "   First run: preparing sysbench data (may take several minutes)..."
  sysbench --db-driver=mysql --mysql-user=root --mysql-password="${root_pw}" \
    --mysql-db="${db_name}" --tables="${tables}" --table-size="${table_size}" \
    oltp_read_write prepare
fi

echo "4. Starting evaluation API on :8080 ..."
exec uv run python "${server_path}"
