# 新服务器上跑通 `run_all_registered_tasks.py` 的完整先决条件

本文档面向**从未配置过本项目的机器**（常见为 Linux 服务器）。按顺序完成下方步骤后，应能在**同一台机**上运行：

```bash
cd Agentbbo
uv run python examples/run_all_registered_tasks.py
```

并满足：

- **Python 侧**：`optuna_tpe` 及**全部已注册**任务（含 scientific、HTTP、BBOPlace）的依赖已就绪。  
- **网络侧**：`127.0.0.1:8070`（BBOPlace）、`8080`（MariaDB 评估器）、`8090`（Surrogate 评估器）均有服务监听，批跑脚本的**默认 auto 探测**会把 `knob_http_*` 与 BBOPlace 排进计划（与 `examples/run_all_registered_tasks.md` 描述一致）。  

若你**故意**只跑本地、不依赖 HTTP，请改用 `--skip-http --skip-bboplace` 等；**不在**本文“完整先决”范围内。

---

## 0. 文档与代码的权威关系

- 以仓库内 **`bbo/` 源码** 与 **`pyproject.toml`** 为准。  
- **端口与任务类型**总览见仓库根或协作路径的 **`database.md`**（与本文互补）。  
- 批跑脚本行为说明见 **`examples/run_all_registered_tasks.md`**。  
- 若某镜像构建步骤变更，以各目录下 **`README.md` / `Dockerfile`** 为准，并**更新本文相应小节**。

---

## 1. 硬件与系统假设

| 项目 | 建议 |
|------|------|
| OS | 常见 **x86_64 Linux**（如 Ubuntu 22.04+）。macOS/Windows 需自行对应 Docker Desktop。 |
| CPU / 内存 | 全量批跑会启动真实 **MariaDB + sysbench**，单评估可能较慢；**≥8 GB RAM** 更稳。 |
| 磁盘 | 克隆仓库、Docker 镜像、`.joblib` 资产、结果目录 **`Agentbbo/runs/`**；**≥20 GB 空闲** 更从容。 |
| 网络 | 需能 **`git clone`**、**`docker pull`**（BBOPlace 公共镜像）、以及（Surrogate 资产）**访问 Google Drive 分享链接**或事先拷贝文件。 |
| GPU | BBOPlace 等镜像在多数 smoke 下 **CPU 即可**；需要时再按各镜像说明加 `--gpus all`。 |

---

## 2. 必装：Git、Docker、构建工具、uv、Python 3.11+

### 2.1 Git

```bash
# Debian/Ubuntu 示例
sudo apt-get update
sudo apt-get install -y git
```

### 2.2 Docker Engine

按 [Docker 官方文档](https://docs.docker.com/engine/install/) 安装 **Docker Engine**（**不要**和仅 Swarm/Compose 的极简包混淆，需有 `docker build` / `docker run`）。

将当前用户加入 `docker` 组（**重新登录**后生效），避免每命令 `sudo`：

```bash
sudo usermod -aG docker "$USER"
# 重新登录或 newgrp docker
```

验证：

```bash
docker version
```

### 2.3 构建自定义镜像的依赖

构建 **MariaDB / Surrogate** 镜像时，部分 Dockerfile 会安装系统包。若 `docker build` 报缺 `gcc` 等，在 Ubuntu 上可：

```bash
sudo apt-get install -y build-essential
```

（以实际 `docker build` 报错为准。）

### 2.4 uv 与 Python

本项目要求 **Python ≥ 3.11**（见 `pyproject.toml`）。推荐 [uv](https://github.com/astral-sh/uv) 装依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 将 uv 的 bin 加进 PATH 后
uv --version
```

`uv` 会负责下载/管理 **Python 3.11+** 解释器（随项目 `requires-python` 解析）。

---

## 3. 获取 Agentbbo 代码

在你要放置项目的目录执行（将 URL 换为你的 fork 或上游）：

```bash
git clone <YOUR_AGENTBBO_GIT_URL> Agentbbo
cd Agentbbo
```

下文 **`Agentbbo/`** 均指**本仓库根目录**（含 `pyproject.toml`、`bbo/`、`examples/run_all_registered_tasks.py`）。

---

## 4. 安装 Python 依赖（**必须**含 Optuna 与 scientific 全量任务）

`run_all_registered_tasks.py` 默认算法为 **`optuna_tpe`**，且会跑 **`her_demo` / `oer_demo` / `molecule_qed_demo` 等**需要 **RDKit 等**的 scientific 任务。按主 **`README.md`** 建议执行：

```bash
cd Agentbbo
uv sync --extra dev --extra optuna --extra bo-tutorial
```

含义简述：

- **`optuna`**：提供 `optuna_tpe` 及 Optuna 依赖。  
- **`bo-tutorial`**：提供 RDKit / pandas 等，用于 `molecule_qed_demo`、部分 scientific smoke。  
- **`dev`**：pytest 等（便于自检）。  

> **未安装 `optuna` extra 时，运行到 `optuna_tpe` 会 ImportError/失败。**

**可选**（一般批跑不强制；仅当你要本地进程内调 surrogate 或部分测试）：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial --extra surrogate
```

**可选**：若你将来要用 ConfigSpace 互操作，再加 `--extra interop`（与当前批跑脚本的**最小**成功路径**无**硬性绑定，除非你在本地代码里用到了 `from_configspace`）。

**自检**（应无报错）：

```bash
uv run python -c "import optuna; from rdkit import Chem; from bbo.tasks import ALL_TASK_NAMES; print(len(ALL_TASK_NAMES), 'tasks')"
```

---

## 5. 三个 HTTP 评估器与端口（与批跑脚本的 auto 模式一致）

| 宿主机端口 | 用途 | 典型容器 / 说明 |
|------------|------|------------------|
| **8070** | BBOPlace 评估服务（**映射到容器 8080**） | 公共镜像 `gaozhixuan/bboplace-bench` |
| **8080** | MariaDB + sysbench HTTP 评估 | **本地 `docker build`** 见 §7 |
| **8090** | Sklearn 代理 HTTP 评估 | **本地 `docker build`** 见 §8 |

默认**环境变量**（与代码、database.md 一致，一般**不必**另设，除非改端口/主机名）：

| 变量 | 典型值 |
|------|--------|
| `BBOPLACE_BASE_URL` | `http://127.0.0.1:8070` |
| `AGENTBBO_HTTP_EVAL_BASE_URL` | `http://127.0.0.1:8080` |
| `AGENTBBO_HTTP_SURROGATE_BASE_URL` | `http://127.0.0.1:8090` |

慢环境可调大（可选）：

```bash
export AGENTBBO_HTTP_EVAL_TIMEOUT_SEC=600
export AGENTBBO_HTTP_SURROGATE_TIMEOUT_SEC=300
```

---

## 6. 启动一：BBOPlace（`8070:8080`）

```bash
docker pull gaozhixuan/bboplace-bench
docker rm -f agentbbo_bboplace 2>/dev/null
docker run -d --name agentbbo_bboplace -p 8070:8080 gaozhixuan/bboplace-bench
```

批跑对 8070 做 **TCP 探测**；不强制你 `curl`。**可选**用容器日志确认 Uvicorn 已监听 `0.0.0.0:8080`（在容器内）。  
需 GPU 时按上游镜像说明加 `--gpus all`（**非**本文默认先决条件）。

---

## 7. 启动二：MariaDB HTTP 评估器（`8080`）

在 **`Agentbbo/bbo/tasks/dbtune/docker_mariadb/`** 下构建，并**映射 8080:8080**（与 `database.md` 一致）：

```bash
cd Agentbbo/bbo/tasks/dbtune/docker_mariadb
docker build -t agentbbo-http-mariadb-eval:v1 .
docker rm -f agentbbo_http_mariadb_eval 2>/dev/null
docker run -d --name agentbbo_http_mariadb_eval -p 8080:8080 agentbbo-http-mariadb-eval:v1
```

**健康检查**（应见 JSON 含 `"status":"ok"` 等）：

```bash
curl -sS http://127.0.0.1:8080/health
```

> 首次/全量 `prepare` 可能较慢，属正常；`run_all` 里 MariaDB 子任务**耗时可明显长于**纯合成任务。  

---

## 8. 启动三：Surrogate HTTP 评估器（`8090`）

### 8.1 资产 `*.joblib`（**强烈建议**）

`bbo/tasks/dbtune/assets/README.md` 说明：大 **`*.joblib` 不在 Git 中**；需从**文档内 Google Drive 链接**下载，放到：

```text
Agentbbo/bbo/tasks/dbtune/assets/
```

并与表里 **文件名** 一致。否则镜像内缺模型，**部分 `knob_http_surrogate_*` 在运行时会失败**。  
仅做 **sysbench-5 最小烟测** 时，仓库提供生成占位小模型的方式（见该 README 的 `build_placeholder_surrogate`），**不能**保证覆盖全部 6 个 HTTP surrogate 名。

### 8.2 构建与运行

**“为什么只写了一个 joblib 环境变量？”——不需要只加载一个；其它任务也会用各自的文件。**

- 镜像的 **`Dockerfile` 会 `COPY assets /app/assets`**：你在 §8.1 里放进 `bbo/tasks/dbtune/assets/` 的 **整目录**（多个 `*.joblib`、各 `knobs_*.json`）都会打进镜像。  
- **`docker_surrogate/server.py` 的 `TASK_DEFS`** 为每个 canonical 任务写好了**默认文件名**（如 `SYSBENCH_all.joblib`、`pg_5.joblib` 等）与可选的**覆盖用环境变量名**（`AGENTIC_BBO_SYSBENCH_ALL_SURROGATE` 等）。  
- 对某个 `POST /evaluate` 的 `task_id`，服务会在容器内按默认路径 **`/app/assets/<默认文件名>`** 去 `joblib.load`；**只有**当你要把某个模型改指到**别的路径**时，才设对应的 `AGENTIC_BBO_*`（与 `bbo/tasks/dbtune/assets/README.md` 表一致）。  
- 因此下面 **`docker run` 不自带任何 `-e` 也成立**：与默认 `COPY` 布局一致时，**六个** surrogate 任务各自会在**第一次**用到该 `task_id` 时从 `/app/assets/` 加载对应文件；不是“只装了一个 `RF_SYSBENCH_5knob`”。

**构建目录必须是 `bbo/tasks/dbtune`（父目录）**（与 `database.md`、`docker_surrogate/README.md` 一致）：

```bash
cd Agentbbo/bbo/tasks/dbtune
docker build -f docker_surrogate/Dockerfile -t agentbbo-surrogate-http-py37:v1 .
docker rm -f agentbbo_surrogate_http 2>/dev/null
docker run -d --name agentbbo_surrogate_http -p 8090:8090 agentbbo-surrogate-http-py37:v1
```

（若某个模型在**其他路径**，再按需加一行，例如  
`-e AGENTIC_BBO_SYSBENCH5_SURROGATE=/path/in/container/to/custom.joblib`；  
或**挂载** `-v /你的/assets:/app/assets:ro` 用宿主机上的整套文件替换镜像内 `assets`。）

**健康检查**：

```bash
curl -sS http://127.0.0.1:8090/health
```

> 该镜像为 **Python 3.7** 栈，与宿主机 3.11 分离，用于反序列化旧 sklearn；**与宿主机** `bbo` 的 **HTTP 任务** 通过 JSON 协议对接即可。

若 `joblib` / sklearn 反序列化失败，见 **`docker_surrogate/README.md` 的 “Unpickling / scikit-learn”** 与 **重建镜像** 说明（版本对齐、`--no-cache` 等）。

---

## 9. 跑批跑前：用脚本自检计划（不真正评估）

在 **`Agentbbo/` 根**：

```bash
uv run python examples/run_all_registered_tasks.py --list
```

应看到：

- TCP 对 **8070 / 8080 / 8090** 的探测为 **ok/closed**（你都已起服务时应为可连），  
- **non-BBOPlace** 任务名列表，及 **BBOPlace** 矩阵行（若 8070 可连且未 `--skip-bboplace`）。  

再试：

```bash
uv run python examples/run_all_registered_tasks.py --dry-run
```

会打印**将要执行的实验条数**；不调用评估器。

---

## 10. 正式运行

```bash
cd Agentbbo
uv run python examples/run_all_registered_tasks.py
```

默认在 **`Agentbbo/runs/demo/batch_all_tasks/`** 下为每个子实验建目录、写 `trials.jsonl` / `summary.json` 等；未加 `--no-plots` 时各 `.../plots/` 下有图。  
结束后会写 **`batch_objectives_table.csv` / `batch_objectives_table.json`**（见 `examples/run_all_registered_tasks.md`），除非 **`--no-table`**。

**Optuna 日志**里多次出现 `A new study created in memory` 是**每任务一 Study**，属预期（见同文档）。

### 10.1 无法一次跑完时

- 全量子实验**非常耗时**（尤其 MariaDB 真机压测、高维 surrogate）。可先用 **`--list` / `--dry-run** 控制预期，并调小 **`--max-evaluations` / `--bboplace-max-evaluations`** 做烟测。  
- 仅缺某一类服务时：用 **`--skip-http`** 或 **`--skip-bboplace`**，或修复端口/容器后再跑。

---

## 11. 排障速查

| 现象 | 可能原因与处理 |
|------|----------------|
| 8070/8080/8090 探测失败 | 对应容器未起、端口被占用、或防火墙拦 **本机回环**（少见）。`docker ps` 检查 `-p` 映射。 |
| `import optuna` 失败 | 未 `uv sync --extra optuna`。 |
| `rdkit` / molecule 相关失败 | 未加 **`--extra bo-tutorial`**。 |
| MariaDB 评估超时 | 调大 `AGENTBBO_HTTP_EVAL_TIMEOUT_SEC`；或减轻负载（`--max-evaluations`、先用 5-knob 任务等）。 |
| Surrogate 503 / 反序列化错误 | 缺/坏 `.joblib`；按 `assets/README.md` 重下；或对齐 `docker_surrogate` 的 sklearn 版本并重建镜像。 |
| BBOPlace 连接失败 | 8070 未映射；`BBOPLACE_BASE_URL` 是否指向**宿主机 8070**（不是 8080）。 |
| 权限 / `denied` | 用户未进 `docker` 组，或应使用 `sudo docker`（不推荐长期使用）。 |

---

## 12. 最小依赖清单（复制粘贴用）

- [ ] Git、Docker、可 `docker build` / `docker run`  
- [ ] `uv`、项目 `cd Agentbbo`  
- [ ] `uv sync --extra dev --extra optuna --extra bo-tutorial`  
- [ ] BBOPlace：`docker pull` + `run -d -p 8070:8080`  
- [ ] MariaDB 评估：`docker_mariadb` 下 build + `run -d -p 8080:8080`，`curl /health`  
- [ ] Surrogate 评估：`dbtune` 下 build（assets 已就位）+ `run -d -p 8090:8090`，`curl /health`  
- [ ] `uv run python examples/run_all_registered_tasks.py --list` 探针为三路 ok  
- [ ] 正式：`uv run python examples/run_all_registered_tasks.py`  

---

## 13. 相关文件索引

| 路径 | 内容 |
|------|------|
| `examples/run_all_registered_tasks.py` | 批跑入口 |
| `examples/run_all_registered_tasks.md` | 脚本设计说明 |
| `database.md` | HTTP 任务与端口、MariaDB / Surrogate / BBOPlace 对照 |
| `README.md` / `README.zh.md` | 安装与 Optuna / bo-tutorial 说明 |
| `bbo/task_descriptions/bboplace_bench/environment.md` | BBOPlace 环境与 `BBOPLACE_BASE_URL` |
| `bbo/tasks/dbtune/docker_mariadb/` | MariaDB 镜像与构建说明 |
| `bbo/tasks/dbtune/docker_surrogate/README.md` | Surrogate 镜像与 API |
| `bbo/tasks/dbtune/assets/README.md` | `*.joblib` 下载与路径 |

完成以上步骤后，在新服务器上**应**具备与本文写作时设计一致的先决条件，以运行 `run_all_registered_tasks.py` 的**全任务**（在默认 `auto` 探测与**未**使用 `--skip-http` / `--skip-bboplace` 等的前提下）。若仓库后续增加了新的硬依赖，请同步更新本文件。
