# 环境配置

这个任务设计为在仓库受管的 Python 环境中运行，并且不需要在线下载。
推荐的环境准备方式是：

```bash
uv sync --extra dev --extra bo-tutorial
```

这是一个已经验证可用的环境，但不一定是唯一可行的环境。
这个任务的硬性要求是安装了与仓库 Python 环境兼容的 RDKit。

最小 smoke test：

```bash
uv run python -m bbo.run --algorithm random_search --task guacamol_qed_demo --max-evaluations 3
```

本地同级目录下的 `guacamol/` checkout 只在开发阶段作为参考实现使用。
这个任务在运行时不要求能够导入那个外部仓库。
