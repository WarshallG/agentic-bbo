"""Standalone benchmark core for agentic black-box optimization."""

from __future__ import annotations

from . import core

# NOTE:
# `bbo` 顶层 import 不应强制依赖所有可选算法依赖（例如 `cma`）。
# 因此这里采用惰性导入，避免在只装部分 extra（比如 surrogate）时 import 失败。

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmSpec",
    "ALL_DEMO_TASK_NAMES",
    "ALL_TASK_NAMES",
    "BBOPLACE_DEFAULT_DEFINITION",
    "BBOPLACE_TASK_KEY",
    "BRANIN_DEFINITION",
    "BUDGETED_SPHERE_TASK_KEY",
    "BBOPlaceDefinition",
    "BBOPlaceTask",
    "BBOPlaceTaskConfig",
    "DBTUNE_MARIADB_TASK_IDS",
    "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "LlamboAlgorithm",
    "OproAlgorithm",
    "PyCmaAlgorithm",
    "RandomSearchAlgorithm",
    "SCIENTIFIC_TASK_REGISTRY",
    "SPHERE_DEFINITION",
    "SURROGATE_TASK_IDS",
    "SYNTHETIC_PROBLEM_REGISTRY",
    "TASK_FAMILIES",
    "TASK_REGISTRY",
    "SyntheticFunctionDefinition",
    "SyntheticFunctionTask",
    "SyntheticFunctionTaskConfig",
    "algorithms_by_family",
    "core",
    "create_algorithm",
    "create_bboplace_task",
    "create_demo_task",
    "create_task",
    "default_bboplace_definition",
    "get_synthetic_problem",
    "run_demo_suite",
    "run_single_experiment",
]


def __getattr__(name: str):
    if name in {"run_demo_suite", "run_single_experiment"}:
        from .run import run_demo_suite, run_single_experiment

        return {"run_demo_suite": run_demo_suite, "run_single_experiment": run_single_experiment}[name]
    if name in {
        "ALGORITHM_REGISTRY",
        "AlgorithmSpec",
        "LlamboAlgorithm",
        "OproAlgorithm",
        "PyCmaAlgorithm",
        "RandomSearchAlgorithm",
        "algorithms_by_family",
        "create_algorithm",
    }:
        from . import algorithms as _algorithms

        return getattr(_algorithms, name)
    if name in {
        "BRANIN_DEFINITION",
        "BUDGETED_SPHERE_TASK_KEY",
        "SPHERE_DEFINITION",
        "ALL_DEMO_TASK_NAMES",
        "ALL_TASK_NAMES",
        "BBOPLACE_DEFAULT_DEFINITION",
        "BBOPLACE_TASK_KEY",
        "BBOPlaceDefinition",
        "BBOPlaceTask",
        "BBOPlaceTaskConfig",
        "DBTUNE_MARIADB_TASK_IDS",
        "DBTUNE_SURROGATE_SERVICE_TASK_IDS",
        "HTTP_SURROGATE_TASK_IDS",
        "SCIENTIFIC_TASK_REGISTRY",
        "SURROGATE_TASK_IDS",
        "SYNTHETIC_PROBLEM_REGISTRY",
        "TASK_FAMILIES",
        "TASK_REGISTRY",
        "SyntheticFunctionDefinition",
        "SyntheticFunctionTask",
        "SyntheticFunctionTaskConfig",
        "create_bboplace_task",
        "create_demo_task",
        "create_task",
        "default_bboplace_definition",
        "get_synthetic_problem",
    }:
        from . import tasks as _tasks

        return getattr(_tasks, name)
    raise AttributeError(f"module 'bbo' has no attribute {name!r}")
