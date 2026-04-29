from __future__ import annotations

from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY, SkydiscoverInterleavedAlgorithm
from bbo.algorithms.llm_based.generated_solver import load_suggest_next_config
from bbo.algorithms.llm_based.skydiscover_bbo import initial_strategy_template_path
from bbo.core import Experimenter, ExperimentConfig, JsonlMetricLogger
from bbo.run import build_arg_parser, run_single_experiment
from bbo.tasks import create_task


@pytest.mark.unit
def test_skydiscover_bbo_yaml_resolves_seed_prompt_file() -> None:
    """Preset YAML embeds IO contract + initial_strategy reference via prompt file."""
    pytest.importorskip("skydiscover")
    from skydiscover.config import load_config

    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / "examples" / "skydiscover_configs" / "gepa_native_bbo.yaml"
    assert cfg_path.is_file()
    config = load_config(str(cfg_path))
    system = config.context_builder.system_message
    assert "suggest_next_config" in system
    assert "_mix_seed" in system
    assert "Reference seed implementation" in system
    assert "FULL-REWRITE OUTPUT" in system


@pytest.mark.unit
def test_parse_full_rewrite_skips_diff_only_fence() -> None:
    from skydiscover.utils.code_utils import parse_full_rewrite

    raw = """Explanation

```diff
<<<<<<< SEARCH
a
=======
b
>>>>>>> REPLACE
```
"""
    assert parse_full_rewrite(raw, "python") is None


@pytest.mark.unit
def test_parse_full_rewrite_prefers_python_over_diff_block() -> None:
    from skydiscover.utils.code_utils import parse_full_rewrite

    raw = """Text
```diff
garbage
```
```python
x = 2
```
"""
    out = parse_full_rewrite(raw, "python")
    assert out is not None and "x = 2" in out and "garbage" not in out


    from skydiscover.utils.code_utils import try_extract_fenced_python_when_diff_markers_present

    raw = """Intro
<<<<<<< SEARCH
a
=======
b
>>>>>>> REPLACE

```python
x = 1
```
"""
    out = try_extract_fenced_python_when_diff_markers_present(raw, "python")
    assert out is not None and "x = 1" in out

    parser = build_arg_parser()
    algo_action = next(a for a in parser._actions if a.dest == "algorithm")
    assert "skydiscover_interleaved" in ALGORITHM_REGISTRY
    assert "skydiscover_meta" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["skydiscover_interleaved"].family == "llm_based"
    assert "skydiscover_interleaved" in algo_action.choices


@pytest.mark.unit
def test_ensure_skydiscover_llm_evaluator_guide_syncs_after_late_model_injection() -> None:
    """Late assignment to ``llm.models`` must refresh empty evaluator/guide pools."""
    pytest.importorskip("skydiscover")
    from skydiscover.config import LLMModelConfig, load_config

    from bbo.algorithms.llm_based.skydiscover_interleaved import (
        _ensure_skydiscover_llm_evaluator_guide,
    )

    config = load_config(None)
    assert not config.llm.models
    assert not config.llm.evaluator_models
    assert not config.llm.guide_models

    config.llm.models = [LLMModelConfig(name="gpt-4o-mini")]
    assert not config.llm.evaluator_models
    assert not config.llm.guide_models

    _ensure_skydiscover_llm_evaluator_guide(config.llm)
    assert len(config.llm.evaluator_models) == 1
    assert len(config.llm.guide_models) == 1
    assert config.llm.evaluator_models[0].name == config.llm.models[0].name


@pytest.mark.unit
def test_load_initial_strategy_contract(tmp_path: Path) -> None:
    fn = load_suggest_next_config(initial_strategy_template_path())
    cfg = fn(
        history=[],
        parameter_specs=[
            {"name": "x", "type": "float", "low": -5.0, "high": 5.0, "log": False},
        ],
        objective_direction="minimize",
        seed=1,
        trial_index=0,
    )
    assert set(cfg.keys()) == {"x"}
    assert isinstance(cfg["x"], float)


@pytest.mark.unit
def test_skydiscover_interleaved_smoke_without_runner(tmp_path: Path) -> None:
    task = create_task("branin_demo", max_evaluations=6, seed=3)
    algo = SkydiscoverInterleavedAlgorithm(
        run_dir=tmp_path,
        interleave_every=2,
        skydiscover_round_iterations=1,
        skydiscover_runner_enabled=False,
    )
    log = tmp_path / "trials.jsonl"
    exp = Experimenter(
        task=task,
        algorithm=algo,
        logger_backend=JsonlMetricLogger(log),
        config=ExperimentConfig(seed=3, resume=False, fail_fast_on_sanity=True),
    )
    summary = exp.run()
    assert summary.n_completed == 6
    assert (tmp_path / "generated" / "strategy.py").is_file()


@pytest.mark.integration
def test_run_single_experiment_skydiscover_interleaved(tmp_path: Path) -> None:
    summary = run_single_experiment(
        task_name="branin_demo",
        algorithm_name="skydiscover_interleaved",
        seed=2,
        max_evaluations=4,
        results_root=tmp_path,
        resume=False,
        generate_plots=False,
        skydiscover_interleave_every=2,
        skydiscover_runner=False,
    )
    assert summary["n_completed"] == 4
    assert Path(summary["run_dir"]).joinpath("generated", "strategy.py").is_file()
    assert "skydiscover_interleaved_topk" in summary["run_dir"].replace("\\", "/")
    assert summary.get("skydiscover_search_type") == "topk"
