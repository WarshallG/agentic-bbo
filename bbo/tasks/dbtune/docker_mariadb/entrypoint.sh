#!/usr/bin/env bash
set -euo pipefail

ROOT_PW="123456"
DB_NAME="sbtest"

echo "1. Starting MariaDB..."
service mariadb start

echo "2. Initializing database and root password..."
mysql -u root -e "CREATE DATABASE IF NOT EXISTS ${DB_NAME};" || true
mysql -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '${ROOT_PW}'; FLUSH PRIVILEGES;" || true

echo "3. Checking sysbench tables..."
RAW_COUNT="$(mysql -u root -p"${ROOT_PW}" -N -e \
  "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='${DB_NAME}';" \
  2>/dev/null | tr -d '[:space:]' || true)"
TABLE_COUNT="${RAW_COUNT:-0}"

if [ "${TABLE_COUNT}" -eq 0 ]; then
  echo "   First run: preparing sysbench data (may take several minutes)..."
  sysbench --db-driver=mysql --mysql-user=root --mysql-password="${ROOT_PW}" \
    --mysql-db="${DB_NAME}" --tables=10 --table-size=100000 \
    oltp_read_write prepare
fi

echo "4. Starting evaluation API on :8080 ..."
exec python3 /app/server.py
