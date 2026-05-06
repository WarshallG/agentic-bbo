#!/usr/bin/env bash
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-johnny114/agentic-bbo}"
IMAGE_TAG="${IMAGE_TAG:-20260504}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
SMOKE_TASK="${SMOKE_TASK:-branin_demo}"
SMOKE_ALGO="${SMOKE_ALGO:-random_search}"
SMOKE_EVALS="${SMOKE_EVALS:-1}"

echo "[1/4] Pulling image: ${IMAGE_REF}"
docker pull "${IMAGE_REF}"

echo "[2/4] Verifying image is present locally"
docker image inspect "${IMAGE_REF}" >/dev/null

echo "[3/4] Running smoke check: ${SMOKE_TASK} / ${SMOKE_ALGO}"
docker run --rm "${IMAGE_REF}" \
  bash -lc "cd /workspace && bash scripts/run_problem.sh ${SMOKE_TASK} ${SMOKE_ALGO} --max-evaluations ${SMOKE_EVALS} --no-plots"

echo "[4/4] Install pipeline completed"
echo
echo "Interactive shell:"
echo "  docker run --rm -it ${IMAGE_REF} bash"
echo
echo "Example task runs:"
echo "  docker run --rm ${IMAGE_REF} bash -lc 'cd /workspace && bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots'"
echo "  docker run --rm ${IMAGE_REF} bash -lc 'cd /workspace && bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots'"
echo "  docker run --rm ${IMAGE_REF} bash -lc 'cd /workspace && bash scripts/run_mariadb_baselines.sh --max-evaluations 1 --no-plots --results-root /workspace/runs/mariadb_batch'"
