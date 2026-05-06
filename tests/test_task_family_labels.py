from __future__ import annotations

from bbo.tasks import TASK_FAMILIES


def test_task_families_use_dbtune_labels_for_dbtune_tasks() -> None:
    assert "dbtune_mariadb" in TASK_FAMILIES
    assert "dbtune_surrogate" in TASK_FAMILIES
    assert "dbtune_surrogate_service" in TASK_FAMILIES
    assert "http_surrogate" not in TASK_FAMILIES
    assert "database" not in TASK_FAMILIES
    assert "surrogate" not in TASK_FAMILIES
