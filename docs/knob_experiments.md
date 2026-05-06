## Agentbbo 中的 knob（参数）surrogate 实验

本文介绍如何在本仓库中运行 **knob（数据库参数调优）surrogate 实验**。

这些任务是 **离线（offline）** benchmark：底层是序列化的 sklearn 模型（`*.joblib`）。它把归一化后的 knob 向量 \(x \in [0,1]^d\) 映射为预测指标（例如 throughput 或 latency）。**不需要真实数据库实例**。

### 你会用到的入口

- **Task family**: surrogate knob tasks under `bbo/tasks/dbtune/`
- **Examples**: `examples/run_knob_surrogate_demo.py`
- **统一运行入口**：`python -m bbo.run`（推荐）或 `bbo.run.run_single_experiment()`
- **输出位置**：默认写到 `runs/demo/` 下的 JSONL trial 日志（`trials.jsonl`）和汇总（`summary.json`）

### 前置条件

- **Python 环境**：推荐用仓库管理的环境（`uv`）
- **surrogate 依赖**：环境里必须有 `joblib`、`scikit-learn`
- **模型 checkpoint**：需要一个可用的 `*.joblib`（Sysbench-5 支持仓库自带的 placeholder）

用 `uv` 安装（推荐）：

```bash
uv sync --extra dev --extra surrogate
```

### 可用的 surrogate knob 任务

用 Python 列出 **catalog / Docker canonical** id（`knob_surrogate_*`，与 `SURROGATE_BENCHMARKS` 一致）：

```bash
uv run python -c "from bbo.tasks import SURROGATE_TASK_IDS; print('\\n'.join(SURROGATE_TASK_IDS))"
```

**`python -m bbo.run` 与 `ALL_TASK_NAMES` 只注册 HTTP 型任务**（`knob_http_surrogate_*`）。本机直接加载 `.joblib` 请用
`from bbo.tasks.dbtune import create_surrogate_knob_task` 或脚本封装，不通过 CLI。

列出可供 `bbo.run` 的 HTTP surrogate task id：

```bash
uv run python -c "from bbo.tasks import HTTP_SURROGATE_TASK_IDS; print(*HTTP_SURROGATE_TASK_IDS, sep='\\n')"
```

常见 canonical 名（与 `assets/README.md`、Docker `GET /task/<id>` 一致）：`knob_surrogate_sysbench_5`、
`knob_surrogate_sysbench_all`、`knob_surrogate_job_5`、`knob_surrogate_job_all`、
`knob_surrogate_pg_5`、`knob_surrogate_pg_20`；CLI 上对应 `knob_http_surrogate_...`（多 `http_` 前缀）。

### 准备 `*.joblib` surrogate 文件

真实的大模型 checkpoint **不会**提交到仓库。请从**发布网盘**下载与任务对应的文件（见下），再放入 `bbo/tasks/dbtune/assets/` 或使用环境变量指向本地路径。

**下载地址**（与仓库 `assets/README.md` 中一致）：

<https://drive.google.com/drive/folders/1qalYsF7fuCB6MewOTPvr8DDZzIj7tIRt?usp=sharing>

- **方式 A（推荐）**：从上述网盘下载所需 `*.joblib`，保存到 `bbo/tasks/dbtune/assets/`，**文件名**须与 `bbo/tasks/dbtune/assets/README.md` 中的表格一致。
- **方式 B**：设置环境变量，指向本机已下载的 `.joblib` 的**绝对路径**（见下表/同页 README）。

文件名 ↔ `task_id` ↔ 环境变量 的对应关系见 `bbo/tasks/dbtune/assets/README.md`。

#### 示例：Sysbench 5-knob RF

从网盘下载 `RF_SYSBENCH_5knob.joblib` 后，放入 assets（**不需要**环境变量）：

```bash
# 将下载好的文件放入（路径按你本机实际下载位置调整）
cp /你的下载目录/RF_SYSBENCH_5knob.joblib bbo/tasks/dbtune/assets/RF_SYSBENCH_5knob.joblib
```

或用环境变量覆盖路径：

```bash
export AGENTIC_BBO_SYSBENCH5_SURROGATE=/absolute/path/to/RF_SYSBENCH_5knob.joblib
```

### 运行 knob 实验（推荐用 `bbo.run`）

跑 random-search baseline：

```bash
uv run python -m bbo.run \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm random_search \
  --seed 1 \
  --max-evaluations 60
```

跑 CMA-ES（需要你环境里额外安装 `cma` / `pycma` 相关依赖）：

```bash
uv run python -m bbo.run \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm pycma \
  --seed 1 \
  --max-evaluations 60 \
  --sigma-fraction 0.18 \
  --popsize 6
```

进程内覆盖 `*.joblib` / `knobs_*.json` 路径时，在代码里调
`create_surrogate_knob_task("knob_surrogate_sysbench_5", ..., surrogate_path=..., knobs_json_path=...)`（见
`bbo.run` 的 `run_single_experiment` 对 surrogate 的 kwargs）。**HTTP** 型（`--task knob_http_surrogate_*`）在
容器内加载模型，一般不在宿主机传 `--surrogate-path`。

### 运行示例脚本

`examples/run_knob_surrogate_demo.py` 本质上只是对 `run_single_experiment()` 的轻量封装：

```bash
uv run python examples/run_knob_surrogate_demo.py \
  --task knob_http_surrogate_sysbench_5 \
  --algorithm random_search \
  --seed 1 \
  --max-evaluations 60
```

### 输出：结果写到哪里

默认输出目录结构如下：

```text
runs/demo/<task>/<algorithm>/seed_<seed>/
  trials.jsonl
  summary.json
```

- **`trials.jsonl`**：每次评估（trial）一行 JSON 记录
- **`summary.json`**：聚合后的最优值、incumbents、以及 logger 汇总

### 进程内（Python 3.11）与 HTTP + Docker（Python 3.7）两种跑法

| 方式 | `task` 命名 | 说明 |
|------|------------|------|
| 进程内 | `create_surrogate_knob_task("knob_surrogate_sysbench_5", ...)` | 本机 `joblib` + 本机 `predict`；**不在** `bbo.run` / `ALL_TASK_NAMES` 注册。 |
| 侧车 HTTP | `knob_http_surrogate_sysbench_5` 等 | 与**真实数据库任务同一思路**：BBO 只产生归一化点，**`POST` 发一个 `x`（`[0,1]^d` 列表）**，**容器里解码 knobs + 代理模型，返回一个标量 `y`**。模型与 sklearn 3.7 环境只在镜像里。 |

**HTTP 合约（与「真实库：发配置、回吞吐」平行）**：

- `POST /evaluate` 推荐请求体：`{"task_id": "knob_surrogate_sysbench_5", "x": [0.0, …, 1.0]}`，长度 `d` 与元数据一致。容器内用自带 `assets/knobs_*.json` 做 `[0,1]→` 物理量，再 `predict`；响应 `status: success` 与 `y`。
- 另支持旧字段 `features`（已是物理量、不解码）以便调试，新代码路径不必用。

**运行**（默认 `http://127.0.0.1:8090`，与数据库评估器 8080 错开）：

- 起容器：见 `bbo/tasks/dbtune/docker_surrogate/README.md`（在 `bbo/tasks/dbtune` 下 `docker build -f docker_surrogate/Dockerfile ...`）。
- 环境变量（宿主机）：`AGENTBBO_HTTP_SURROGATE_BASE_URL`、`AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC`（默认 120）
- 列出 HTTP 型 task id：

```bash
uv run python -c "from bbo.tasks.registry import HTTP_SURROGATE_TASK_IDS; print(*HTTP_SURROGATE_TASK_IDS, sep='\n')"
```

```bash
export AGENTBBO_HTTP_SURROGATE_BASE_URL=http://127.0.0.1:8090
uv run python -m bbo.run --task knob_http_surrogate_sysbench_5 --algorithm random_search --max-evaluations 20 --seed 1
```

**注意**：`.joblib` 与 `knobs_*.json` 应打进或挂载到容器的 `/app/assets`；宿主机 BBO 仅通过 `GET /task/...` 取维度/名字，**不**在 3.11 上反序列化模型。

### 常见问题排查

- **`joblib.load` 报 `EOF` / `reading array data`**
  - 通常是 `.joblib` 文件不完整（复制了一半、或 Git LFS 没拉全）。请重新拷贝完整的 `*.joblib`。
- **`ModuleNotFoundError: joblib` 或 `sklearn`**
  - 安装 surrogate 依赖：`uv sync --extra surrogate`
- **使用 `--algorithm pycma` 时提示 `ModuleNotFoundError: cma`**
  - 你需要先在环境里安装 `cma` 依赖，然后再使用 `pycma`。

