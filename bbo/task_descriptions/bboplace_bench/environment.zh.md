# 环境配置

这个任务不是在本仓库里重建完整的上游 BBOPlace-Bench，而是通过一个已发布的 evaluator 镜像，以外部服务的形式接入。
当前本仓库引用的公开镜像是 Docker Hub 上的 `gaozhixuan/bboplace-bench`。

## 主机要求

- 已安装 Docker Engine，并且有权限拉取和运行公开镜像。
- 本仓库侧需要 Python 3.11+ 与 `uv`。
- 可选：如果你本地部署该镜像时需要 GPU，则还需要可用的 GPU Docker 运行时。

## 拉取 evaluator 镜像

```bash
docker pull gaozhixuan/bboplace-bench
```

## 启动 evaluator service

CPU 风格的最小启动命令：

```bash
docker run --rm -p 8070:8080 gaozhixuan/bboplace-bench
```

（镜像**容器内**仍监听 8080；映射到宿主机 **8070**，与本仓库中 MariaDB 评估默认 **8080** 错开。）

如果你的本地环境要求 GPU，可在镜像名前加上 `--gpus all`。
`--rm` 适合一次性的 smoke test；如果你想保留容器用于调试或查看退出后的日志，可以去掉这个参数。

当前打包任务默认访问 `http://127.0.0.1:8070`，并向 `/evaluate` 发送评估请求。
如果 service 部署在其他地址，可以这样覆盖：

```bash
export BBOPLACE_BASE_URL=http://127.0.0.1:8070
```

## 安装本仓库

```bash
uv sync --extra dev
```

## Smoke test

在 container 运行后，执行：

```bash
export BBOPLACE_BASE_URL=http://127.0.0.1:8070
uv run python -m bbo.run --algorithm random_search --task bboplace_bench --max-evaluations 1
```

如果环境正常，这个命令应当能够完成运行，不会出现连接错误或 JSON schema 错误，并在 `artifacts/` 下写出结果。
如果你需要完整的上游工作流，比如重编译 DREAMPlace、下载 benchmark 数据集，或者运行 SP / HPO / GP-HPWL / PPA 流程，应参考官方 BBOPlace-Bench 仓库，而不是这个轻量级 service wrapper。
