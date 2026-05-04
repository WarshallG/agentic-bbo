#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

tasks=(
  knob_http_mariadb_sysbench_read_only_5
  knob_http_mariadb_sysbench_write_only_5
  knob_http_mariadb_sysbench_read_write_5
  knob_http_mariadb_sysbench_point_select_5
  knob_http_mariadb_sysbench_read_only_all
  knob_http_mariadb_sysbench_write_only_all
  knob_http_mariadb_sysbench_read_write_all
  knob_http_mariadb_sysbench_point_select_all
)

algorithms=(
  random_search
  pycma
  optuna_tpe
  pfns4bo_tabpfn_v2
  llambo
  opro
  skydiscover_interleaved
  pablo
)

common_args=("${@}")
if [[ ${#common_args[@]} -eq 0 ]]; then
  common_args=(--max-evaluations 1 --no-plots)
fi

for task_name in "${tasks[@]}"; do
  for algorithm_name in "${algorithms[@]}"; do
    echo "=== ${task_name} :: ${algorithm_name} ==="
    bash scripts/run_problem.sh "${task_name}" "${algorithm_name}" "${common_args[@]}"
  done
done
