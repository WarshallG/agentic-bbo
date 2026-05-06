"""Register dbtune surrogate-service tasks with ``bbo.tasks.registry`` / ``python -m bbo.run``."""

from __future__ import annotations

from typing import Any

from ...core import Task
from .http_surrogate_specs import (
    DBTUNE_SURROGATE_SERVICE_TASK_IDS,
    HTTP_SURROGATE_TASK_IDS,
    is_dbtune_surrogate_service_task_id,
)
from .http_surrogate_task import create_dbtune_surrogate_service_task

DBTUNE_SURROGATE_SERVICE_TASK_FAMILY = "dbtune_surrogate_service"
DBTUNE_SURROGATE_SERVICE_TASK_NAMES: frozenset[str] = frozenset(DBTUNE_SURROGATE_SERVICE_TASK_IDS)

# Backward-compatible aliases for older imports.
HTTP_SURROGATE_TASK_FAMILY = DBTUNE_SURROGATE_SERVICE_TASK_FAMILY
HTTP_SURROGATE_TASK_NAMES = DBTUNE_SURROGATE_SERVICE_TASK_NAMES


def dbtune_surrogate_service_registry_entries() -> dict[str, str]:
    """task_id -> family label for ``TASK_REGISTRY``."""

    return {name: DBTUNE_SURROGATE_SERVICE_TASK_FAMILY for name in DBTUNE_SURROGATE_SERVICE_TASK_IDS}


def create_dbtune_surrogate_service_task_for_registry(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
    **kwargs: Any,
) -> Task:
    """Dispatch when ``name`` is a registered dbtune surrogate-service task id."""

    _ = noise_std
    if not is_dbtune_surrogate_service_task_id(name):
        known = ", ".join(sorted(DBTUNE_SURROGATE_SERVICE_TASK_IDS))
        raise ValueError(f"Unknown dbtune surrogate-service task `{name}`. Known: {known}")
    return create_dbtune_surrogate_service_task(
        name,
        max_evaluations=max_evaluations,
        seed=seed,
        base_url=kwargs.get("http_surrogate_base_url"),
        request_timeout_sec=kwargs.get("http_surrogate_timeout_sec"),
        skip_health_check=bool(kwargs.get("http_surrogate_skip_health_check", False)),
    )


http_surrogate_registry_entries = dbtune_surrogate_service_registry_entries
create_http_surrogate_task_for_registry = create_dbtune_surrogate_service_task_for_registry


__all__ = [
    "DBTUNE_SURROGATE_SERVICE_TASK_FAMILY",
    "DBTUNE_SURROGATE_SERVICE_TASK_NAMES",
    "HTTP_SURROGATE_TASK_FAMILY",
    "HTTP_SURROGATE_TASK_NAMES",
    "create_dbtune_surrogate_service_task_for_registry",
    "create_http_surrogate_task_for_registry",
    "dbtune_surrogate_service_registry_entries",
    "http_surrogate_registry_entries",
]
