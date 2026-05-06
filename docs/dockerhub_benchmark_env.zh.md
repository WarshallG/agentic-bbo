# Docker Hub Benchmark 环境说明

这份文档记录的是我这次实际跑通 benchmark 时使用的 Docker 环境配置路径，目标是：在一个全新的、没有配过本地 Python 环境的代码目录里，直接复现统一镜像方案。

已发布镜像：

- `johnny114/agentic-bbo:20260504`
- `johnny114/agentic-bbo:latest`

这个统一镜像覆盖的范围：

- synthetic tasks
- scientific tasks
- `bboplace_bench`
- `knob_http_mariadb_*`

这个统一镜像明确不覆盖：

- `knob_http_surrogate_*`

这些 surrogate 任务仍然需要单独的旧版 Python 3.7 sidecar 镜像。

已验证可用的 surrogate sidecar 镜像：

- `fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1`

## 先说结论

如果你只是想把环境配起来，并且按仓库现有的标准入口去跑任务，那么**不需要额外写代码**。直接拉 Docker Hub 上的发布镜像就够了。

你可以直接执行：

```bash
docker pull johnny114/agentic-bbo:20260504
```

也可以用仓库里现成的安装脚本：

```bash
bash scripts/install_dockerhub_benchmark_env.sh
```

我额外写的代码，只是为了把一大批任务和 baseline 组合一次性批量跑完，并顺手收集日志；这些代码**不是**环境搭建所必需的。

## 镜像里包含什么

镜像内已经包含：

- `/workspace` 下的 benchmark 仓库
- 预先同步好的 `uv` 环境
- `benchmark-main` 依赖集合
- `/opt/BBOPlace-miniBench` 下自带的 `BBOPlace-miniBench`
- 本地 BBOPlace bridge 所需的数据
- `knob_http_mariadb_*` 所需的 MariaDB + sysbench 运行时
- `/workspace/scripts` 下的统一运行脚本

## 端口细节

如果你通过 `scripts/run_problem.sh` 跑任务，benchmark 进程会直接和容器内本地服务通信。正常情况下，**不需要**把端口映射到宿主机。

容器内部用到的端口：

- `8070`：BBOPlace 本地 HTTP bridge
- `8080`：MariaDB/sysbench HTTP evaluator
- `8090`：给 surrogate evaluator 预留，但这个统一镜像里**没有**提供

脚本默认使用的环境变量：

- `BBOPLACE_BASE_URL=http://127.0.0.1:8070`
- `AGENTBBO_HTTP_EVAL_BASE_URL=http://127.0.0.1:8080`
- `AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090`

如果你为了调试，想从宿主机直接看这些服务，可以显式映射端口：

```bash
docker run --rm -it \
  -p 8070:8070 \
  -p 8080:8080 \
  johnny114/agentic-bbo:20260504 bash
```

注意：

- `8090` 不属于这个统一镜像
- 如果你要跑 surrogate 任务，需要单独起那个 Python 3.7 镜像

## 前置要求

需要满足：

- 已安装 Docker Engine
- 当前用户可以访问 Docker daemon
- 能访问 Docker Hub
- 磁盘空间足够，因为镜像很大

如果你已经在 `docker` 组里，但当前 shell 还是报 `/var/run/docker.sock` 权限错误，可以刷新当前 shell 的 group：

```bash
newgrp docker
```

或者直接通过：

```bash
sg docker -c 'docker version'
```

我这次实际验证时就碰到了这个问题，所以后续很多 Docker 命令都是通过 `sg docker -c '...'` 跑的。

## 新开一个干净代码目录

如果你想像我一样，在一个没有本地虚拟环境、没有旧 run 结果、没有临时目录污染的新目录里复现，可以这样做：

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

这一步不是必须的，只是为了得到一个干净的代码工作目录。

## 按命令顺序的实际配置步骤

### 1. 拉镜像

直接拉：

```bash
docker pull johnny114/agentic-bbo:20260504
```

或者用仓库脚本：

```bash
bash scripts/install_dockerhub_benchmark_env.sh
```

这个脚本默认会做三件事：

1. `docker pull johnny114/agentic-bbo:20260504`
2. `docker image inspect ...`
3. 在容器里对 `branin_demo + random_search` 做一次 smoke test

我这次实际执行时，Docker Hub 有一次瞬时 `EOF`，重试 `docker pull` 后成功。所以如果你碰到同样的 registry 抖动，直接重试即可。

### 2. 检查镜像是否已经在本地

```bash
docker image inspect johnny114/agentic-bbo:20260504
```

### 3. 跑最小 smoke test

```bash
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots"
```

### 4. 交互式进入容器，并把结果持久化到宿主机

```bash
mkdir -p runs
docker run --rm -it \
  -v "$(pwd)/runs:/workspace/runs" \
  johnny114/agentic-bbo:20260504 bash
```

进入容器后：

```bash
cd /workspace
```

## 标准运行命令

单任务统一入口：

```bash
bash scripts/run_problem.sh <task_name> <algorithm_name> [extra args...]
```

例子：

```bash
bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots
bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots
```

这个入口脚本的内建行为：

- `bboplace_bench`
  - 自动在 `127.0.0.1:8070` 起本地 bridge
  - 使用 `/opt/BBOPlace-miniBench`
- `knob_http_mariadb_*`
  - 自动在 `127.0.0.1:8080` 起本地 evaluator
  - 第一次可能会先准备 sysbench 表
- `knob_http_surrogate_*`
  - 这个镜像不支持
  - 仍然需要单独的 Python 3.7 surrogate 容器

仓库里已经有一个 MariaDB 8 任务批跑脚本：

```bash
bash scripts/run_mariadb_baselines.sh --max-evaluations 1 --no-plots --results-root /workspace/runs/mariadb_batch
```

## 可选：Surrogate Sidecar 安装

如果你要跑 6 个 `knob_http_surrogate_*` 任务，需要额外启动独立的 Python 3.7 surrogate HTTP 服务，端口是 `8090`。

这次实际验证可用的预构建镜像是：

```bash
docker pull fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
docker rm -f agentbbo_surrogate_http 2>/dev/null || true
docker run -d --name agentbbo_surrogate_http \
  -p 8090:8090 \
  fakerstrawberry/agentbbo-dbtune-surrogate-http-py37:v1
```

健康检查：

```bash
curl -sS http://127.0.0.1:8090/health
```

预期返回：

```json
{"status":"ok"}
```

这个镜像已经把 surrogate `.joblib` 和配套的 `knobs_*.json` 都打进 `/app/assets` 了，所以我这次验证时不需要再额外挂载资产目录。

## 跑 Surrogate 任务的方式

benchmark 客户端需要能访问 `127.0.0.1:8090` 上的 surrogate sidecar。

我这次实际验证时，用的是最直接的方式：

- surrogate 镜像跑在宿主机上，并通过 `-p 8090:8090` 暴露
- 主 benchmark 镜像使用 `--network host`
- 设置 `AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090`

示例：

```bash
docker run --rm --network host \
  johnny114/agentic-bbo:20260504 \
  bash -lc 'cd /workspace && \
    AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090 \
    bash scripts/run_problem.sh knob_http_surrogate_sysbench_5 random_search \
      --max-evaluations 1 --no-plots'
```

如果你不想用 `--network host`，也可以让两个容器放在同一个 Docker network 里，然后把 `AGENTBBO_HTTP_SURROGATE_BASE_URL` 指到 sidecar 的容器名。

这次已验证的 surrogate 任务 ID：

- `knob_http_surrogate_sysbench_5`
- `knob_http_surrogate_sysbench_all`
- `knob_http_surrogate_job_5`
- `knob_http_surrogate_job_all`
- `knob_http_surrogate_pg_5`
- `knob_http_surrogate_pg_20`

## Surrogate 任务的全 baseline 扫描

我已经把下面这 8 个 baseline 在上述 6 个 surrogate 任务上全部跑通：

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

验证结果：

- `6 个任务 × 8 个 baseline = 48/48` 全部成功

### PFNS 冷启动说明

如果 `pfns4bo_tabpfn_v2` 是一个全新 benchmark 容器里的第一次 PFNS 调用，它的首个模型初始化可能会明显慢于后续运行。

所以如果你要跑完整的 surrogate baseline 矩阵，建议先预热一次 PFNS：

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

加了这个预热步骤之后，这次 surrogate 全量 baseline 扫描已经稳定跑通。

## 我这次实际验证了什么

我这次实际验证的是：

- 11 个 non-dbtune 任务
- 8 个 MariaDB HTTP 任务
- 8 个 baseline

这 8 个 baseline 是：

- `random_search`
- `pycma`
- `optuna_tpe`
- `pfns4bo_tabpfn_v2`
- `llambo`
- `opro`
- `skydiscover_interleaved`
- `pablo`

这 6 个任务不在“统一主镜像单独运行”的覆盖范围内，但在加上 surrogate sidecar 后已经验证可运行：

- `knob_http_surrogate_job_5`
- `knob_http_surrogate_job_all`
- `knob_http_surrogate_pg_20`
- `knob_http_surrogate_pg_5`
- `knob_http_surrogate_sysbench_5`
- `knob_http_surrogate_sysbench_all`

最终结果：

- 总共尝试 `152` 个组合
- 成功 `151` 个
- 唯一失败的是 `molecule_qed_demo × pycma`

这个失败不是环境没配好，而是算法/任务兼容性问题：`pycma` 在这个任务的类别展开后，试图分配大约 `464 GiB` 的协方差矩阵。

## 我有没有额外写代码

如果只看“环境搭建”和“标准任务运行”：

- 没有
- 直接拉取发布镜像就够了

如果看“把 19 个任务 × 8 个 baseline 一次性批量跑完”：

- 我写了两个临时 helper
- 它们只是在 fresh 验证目录里使用
- 不是环境搭建的必要部分

这两个临时 helper 的路径是：

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/tmp/dockerhub_batch_runner.py`
- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/tmp/run_dockerhub_in_container.sh`

它们的用途只是：

- 在一个容器里串行跑完整任务矩阵
- 收集 stdout/stderr 日志
- 汇总每个组合的结果

如果你只需要：

- 拉镜像
- 进入容器
- 用 `scripts/run_problem.sh` 跑单任务
- 用 `scripts/run_mariadb_baselines.sh` 跑 8 个 MariaDB 任务

那么完全不需要这两个额外脚本。

## 我这次验证结果放在哪里

fresh 验证目录：

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506`

批跑汇总：

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/full/results.json`
- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/full/results.csv`

每个 run 的产物：

- `/home/trx/cm/agentic-bbo_runtime_fresh_20260506/runs/dockerhub_batch_20260506/runs`

## 已知限制

- `knob_http_surrogate_*` 仍然不在这个统一镜像范围内
- 镜像很大
- `pfns4bo_tabpfn_v2` 第一次运行可能明显更慢
- Docker Hub 偶发 `EOF` 之类的拉取抖动需要重试

## 最短检查清单

```bash
docker pull johnny114/agentic-bbo:20260504
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh branin_demo random_search --max-evaluations 1 --no-plots"
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh bboplace_bench random_search --max-evaluations 1 --no-plots"
docker run --rm johnny114/agentic-bbo:20260504 \
  bash -lc "cd /workspace && bash scripts/run_problem.sh knob_http_mariadb_sysbench_read_only_5 random_search --max-evaluations 1 --no-plots"
```
