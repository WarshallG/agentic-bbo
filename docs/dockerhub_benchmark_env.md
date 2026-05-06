# Docker Hub Benchmark Environment

This document records the exact Docker-based setup path that was used to run the benchmark in a fresh, no-preinstalled-environment code directory.

Published image:

- `johnny114/agentic-bbo:20260504`
- `johnny114/agentic-bbo:latest`

Scope covered by this image:

- synthetic tasks
- scientific tasks
- `bboplace_bench`
- `knob_http_mariadb_*`

Scope intentionally excluded from this unified image:

- `knob_http_surrogate_*`

Those surrogate tasks still need the separate legacy Python 3.7 sidecar image.

Validated surrogate sidecar image:

- `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`

## Short Answer

If you only want the environment and the standard runner behavior, you do **not** need to write extra code. Pulling the published image is enough.

Use either:

```bash
docker pull johnny114/agentic-bbo:20260504
```

or:

```bash
bash scripts/install_dockerhub_benchmark_env.sh
```

The extra code I wrote was only for orchestrating one large verification batch across many tasks and baselines. It was not required for pulling the image, starting the environment, or running standard single-task commands.

## What Is Inside The Image

The image includes:

- the benchmark repository under `/workspace`
- a pre-synced `uv` environment
- the `benchmark-main` dependency set
- a self-contained `BBOPlace-miniBench` checkout under `/opt/BBOPlace-miniBench`
- bundled BBOPlace datasets needed by the local bridge
- MariaDB + sysbench runtime for `knob_http_mariadb_*`
- the standard helper scripts under `/workspace/scripts`

## Ports

When you run tasks through `scripts/run_problem.sh`, the benchmark process talks to local services inside the same container. In the normal path, you do **not** need to publish ports to the host.

Ports used inside the container:

- `8070`: local BBOPlace HTTP bridge
- `8080`: local MariaDB/sysbench HTTP evaluator
- `8090`: reserved for the separate surrogate evaluator image, not provided by this unified image

Environment variables used by the scripts:

- `BBOPLACE_BASE_URL=http://127.0.0.1:8070`
- `AGENTBBO_HTTP_EVAL_BASE_URL=http://127.0.0.1:8080`
- `AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090`

If you want host-side debugging, you may publish the ports explicitly:

```bash
docker run --rm -it \
  -p 8070:8070 \
  -p 8080:8080 \
  johnny114/agentic-bbo:20260504 bash
```

Do not expect `8090` to work in this image; that port belongs to the separate surrogate service.

## Prerequisites

You need:

- Docker Engine installed
- permission to talk to the Docker daemon
- network access to Docker Hub
- enough disk for a very large image

If `docker` exists but the current shell still gets `permission denied` on `/var/run/docker.sock`, refresh the group in the current shell:

```bash
newgrp docker
```

or run commands through:

```bash
sg docker -c 'docker version'
```

That exact `sg docker -c '...'` workaround was needed during my validation because the user was already in the `docker` group, but the current shell had not picked it up yet.

## Fresh Working Directory

For a clean verification directory, I created a new code directory without reusing the old virtualenv, run outputs, or temp state.

Example:

```bash
mkdir -p /home/trx/cm/agentic-bbo_runtime_fresh_20260506
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.venv-*' \
  --exclude 'runs' \
  --exclude 'artifacts' \
  --exclude 'tmp' \
  --exclude '.pytest_cache' \
  /home/trx/cm/agentic-bbo/ \
  /home/trx/cm/agentic-bbo_runtime_fresh_20260506/
cd /home/trx/cm/agentic-bbo_runtime_fresh_20260506
```

This step is optional. It is only useful if you want a fresh code checkout directory with no local environment state mixed in.

## Step-By-Step Setup

### 1. Pull The Image

Direct pull:

```bash
docker pull johnny114/agentic-bbo:20260504
```

Or use the repository helper:

```bash
bash scripts/install_dockerhub_benchmark_env.sh
```

The helper does three things:

1. `docker pull johnny114/agentic-bbo:20260504`
2. `docker image inspect ...`
3. a smoke test for `branin_demo` with `random_search`

Note:

- Docker Hub returned a transient `EOF` once during my run
- retrying `docker pull` succeeded

If you hit the same transient registry error, just retry the pull.

### 2. Verify The Image Exists

```bash
docker image inspect johnny114/agentic-bbo:20260504
```

### 3. Run The Minimal Smoke Test

```bash
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots"
```

### 4. Start An Interactive Container And Persist Outputs

```bash
mkdir -p runs
docker run --rm -it \
  -v "$(pwd)/runs:/workspace/runs" \
  johnny114/agentic-bbo:20260504 bash
```

Inside the container:

```bash
cd /workspace
```

## Standard Run Commands

Single task:

```bash
bash scripts/run_problem.sh <task_name> <algorithm_name> [extra args...]
```

Examples:

```bash
bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots
```

Built-in behavior:

- `bboplace_bench`
  - auto-starts the local bridge on `127.0.0.1:8070`
  - uses `/opt/BBOPlace-miniBench`
- `knob_http_mariadb_*`
  - auto-starts the local evaluator on `127.0.0.1:8080`
  - first startup may prepare sysbench tables
- `knob_http_surrogate_*`
  - not supported by this image
  - still requires the separate Python 3.7 surrogate container

MariaDB batch helper already included in the repo:

```bash
bash scripts/run_mariadb_baselines.sh --max-evaluations 1 --no-plots --results-root /workspace/runs/mariadb_batch
```

## Optional Surrogate Sidecar Setup

To run the six `knob_http_surrogate_*` tasks, start the separate Python 3.7 surrogate service on port `8090`.

This prebuilt image was validated:

```bash
docker pull fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
docker rm -f agentbbo_surrogate_http 2>/dev/null || true
docker run -d --name agentbbo_surrogate_http \
  -p 8090:8090 \
  fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

Health check:

```bash
curl -sS http://127.0.0.1:8090/health
```

Expected response:

```json
{"status":"ok"}
```

This image already includes the surrogate `.joblib` files and the matching `knobs_*.json` files under `/app/assets`, so no extra asset mount was needed in my validation.

## Running Surrogate Tasks

The benchmark client must be able to reach the surrogate sidecar at `127.0.0.1:8090`.

The simplest validated setup was:

- run the surrogate image on the host with `-p 8090:8090`
- run the main benchmark image with `--network host`
- set `AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090`

Example:

```bash
docker run --rm --network host \
  johnny114/agentic-bbo:20260504 \
  bash -lc 'cd /workspace && \
    AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090 \
    bash scripts/run_problem.sh knob_http_surrogate_sysbench_5 random_search \
      --max-evaluations 1 --no-plots'
```

If you do not want `--network host`, use a shared Docker network and point `AGENTBBO_HTTP_SURROGATE_BASE_URL` at the sidecar container name instead.

Validated surrogate task IDs:

- `knob_http_surrogate_sysbench_5`
- `knob_http_surrogate_sysbench_all`
- `knob_http_surrogate_job_5`
- `knob_http_surrogate_job_all`
- `knob_http_surrogate_pg_5`
- `knob_http_surrogate_pg_20`

## Full Baseline Sweep For Surrogate Tasks

The following eight baselines were validated on all six surrogate tasks:

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

Validation result:

- `6 tasks × 8 baselines = 48/48` successful combinations

### PFNS cold-start note

When `pfns4bo_tabpfn_v2` is the first PFNS call in a fresh benchmark container, the initial model setup can be much slower than the later runs.

For a full surrogate sweep, I recommend prewarming PFNS once before running the six-task matrix:

```bash
docker run --rm --network host \
  johnny114/agentic-bbo:20260504 \
  bash -lc 'cd /workspace && \
    bash scripts/run_problem.sh branin_demo pfns4bo_tabpfn_v2 \
      --max-evaluations 3 \
      --pfns-pool-size 32 \
      --pfns-tabpfn-n-estimators 2 \
      --no-plots'
```

After this prewarm step, the validated surrogate sweep completed cleanly.

## What I Actually Verified

I verified the image against:

- 11 non-dbtune tasks
- 8 MariaDB HTTP tasks
- 8 baselines

The 8 baselines were:

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

Tasks excluded from the unified image-only path, but runnable with the surrogate sidecar:

- `knob_http_surrogate_job_5`
- `knob_http_surrogate_job_all`
- `knob_http_surrogate_pg_20`
- `knob_http_surrogate_pg_5`
- `knob_http_surrogate_sysbench_5`
- `knob_http_surrogate_sysbench_all`

Result summary of the full run:

- `152` combinations attempted
- `151` combinations succeeded
- only `molecule_qed_demo × pycma` failed

That failure was not an environment failure. It was a task/algorithm compatibility problem: `pycma` tried to allocate an approximately `464 GiB` covariance matrix after categorical expansion.

## Extra Code I Wrote

For environment setup itself:

- no extra code was required

For the one-shot full verification batch:

- I wrote two temporary helper scripts in the fresh verification directory
- they were not needed for standard environment setup
- they were only used to orchestrate the large `19 tasks × 8 baselines` matrix in one container and collect logs

Temporary helper paths used during validation:

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/tmp/dockerhub_batch_runner.py`
- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/tmp/run_dockerhub_in_container.sh`

These helpers are not required if you only want to:

- pull the image
- start the container
- run `scripts/run_problem.sh`
- run `scripts/run_mariadb_baselines.sh`

## Result Locations From My Validation

Fresh validation directory:

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506`

Batch summary:

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/full/results.json`
- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/full/results.csv`

Per-run outputs:

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/runs`

## Known Limitations

- `knob_http_surrogate_*` is still out of scope for this image
- the image is large
- the first `pfns4bo_tabpfn_v2` run can be much slower than the rest
- transient Docker Hub pull failures such as `EOF` may need a retry

## Quick Checklist

```bash
docker pull johnny114/agentic-bbo:20260504
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots"
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots"
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots"
```
