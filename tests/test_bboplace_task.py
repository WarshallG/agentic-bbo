from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from bbo.core import TrialSuggestion
from bbo.tasks.bboplace import BBOPlaceTask, BBOPlaceTaskConfig


@pytest.mark.unit
def test_bboplace_evaluate_success_payload_and_metrics() -> None:
    capture: dict[str, Any] = {}

    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        capture["url"] = url
        capture["payload"] = payload
        capture["timeout"] = timeout
        return {"hpwl": [123.0]}

    task = BBOPlaceTask(
        config=BBOPlaceTaskConfig(post_json=post_json, http_timeout_seconds=1.5),
        definition=definition,
    )

    config = task.spec.search_space.defaults()
    result = task.evaluate(TrialSuggestion(config=config, trial_id=0))

    assert result.success
    assert result.objectives["hpwl"] == 123.0
    assert capture["url"].endswith("/evaluate")
    assert capture["payload"]["benchmark"] == definition.benchmark
    assert capture["payload"]["placer"] == definition.placer
    assert capture["payload"]["n_macro"] == definition.n_macro
    assert capture["payload"]["seed"] == 0
    assert isinstance(capture["payload"]["x"], list)
    assert len(capture["payload"]["x"]) == 1
    assert len(capture["payload"]["x"][0]) == definition.dimension
    assert capture["timeout"] == 1.5

    assert math.isfinite(float(result.metrics["dimension"]))
    assert "coord::x_0" in result.metrics
    assert "coord::y_0" in result.metrics


@pytest.mark.unit
def test_bboplace_evaluate_invalid_response_shape() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {"hpwl": []}

    task = BBOPlaceTask(config=BBOPlaceTaskConfig(post_json=post_json), definition=definition)
    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults(), trial_id=0))

    assert not result.success
    assert result.status.value == "failed"
    assert result.error_type == "InvalidResponse"


@pytest.mark.unit
def test_bboplace_user_facing_metadata_avoids_http_benchmark_language() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    task = BBOPlaceTask(config=BBOPlaceTaskConfig(), definition=default_bboplace_definition())

    assert "HTTP" not in str(task.spec.metadata["display_name"])
    assert "HTTP" not in task.definition.description


@pytest.mark.unit
def test_bboplace_evaluate_nonfinite_hpwl_is_failed() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    definition = default_bboplace_definition()

    def post_json(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {"hpwl": [float("inf")]}

    task = BBOPlaceTask(config=BBOPlaceTaskConfig(post_json=post_json), definition=definition)
    result = task.evaluate(TrialSuggestion(config=task.spec.search_space.defaults(), trial_id=0))

    assert not result.success
    assert result.status.value == "failed"
    assert result.error_type == "InvalidResponse"


@pytest.mark.unit
def test_bboplace_default_definition_rejects_n_macro_above_benchmark_cap() -> None:
    from bbo.tasks.bboplace.task import default_bboplace_definition

    default_bboplace_definition(benchmark="bigblue1", n_macro=560)

    with pytest.raises(ValueError, match="560"):
        default_bboplace_definition(benchmark="bigblue1", n_macro=561)


@pytest.mark.unit
def test_bboplace_max_n_macro_for_benchmark_normalizes_path() -> None:
    from bbo.tasks.bboplace.task import max_n_macro_for_benchmark

    assert max_n_macro_for_benchmark("ispd2005/bigblue1") == 560
    assert max_n_macro_for_benchmark("UNKNOWN_BENCH") is None


@pytest.mark.unit
def test_bboplace_sanity_check_flags_n_macro_above_cap() -> None:
    from bbo.tasks.bboplace.task import (
        DEFAULT_N_GRID,
        _build_macro_placement_space,
        default_bboplace_definition,
    )

    base = default_bboplace_definition(benchmark="bigblue3", n_macro=1298)
    space_over = _build_macro_placement_space(
        n_macro=1299,
        n_grid_x=DEFAULT_N_GRID,
        n_grid_y=DEFAULT_N_GRID,
    )
    definition = replace(
        base,
        n_macro=1299,
        search_space=space_over,
        display_name="BBOPlace-Bench (bigblue3, 1299 macros)",
    )
    task = BBOPlaceTask(config=BBOPlaceTaskConfig(), definition=definition)
    report = task.sanity_check()
    assert not report.ok
    assert any(issue.code == "n_macro_exceeds_benchmark_cap" for issue in report.errors)
    assert "1298" in next(i.message for i in report.errors if i.code == "n_macro_exceeds_benchmark_cap")
