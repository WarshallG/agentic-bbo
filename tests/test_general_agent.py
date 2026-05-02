from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bbo.algorithms import ALGORITHM_REGISTRY
from bbo.algorithms.agentic import (
    GeneralAgentValidationError,
    MockAgentEngine,
    NanobotBBOAlgorithm,
    parse_agent_candidate_payload,
)
from bbo.core import ExperimentConfig, Experimenter, JsonlMetricLogger
from bbo.run import build_arg_parser
from bbo.tasks import create_task


class ReasoningMetadataMockEngine(MockAgentEngine):
    async def run_agent(self, *args, extra_env=None, **kwargs):  # type: ignore[no-untyped-def]
        result = await super().run_agent(*args, extra_env=extra_env, **kwargs)
        if extra_env:
            call_id = extra_env["BBO_AGENT_CALL_ID"]
            trace_dir = Path(extra_env["BBO_NANOBOT_REASONING_DIR"])
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{call_id}_reasoning.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "call_id": call_id,
                        "reasoning_visible": True,
                        "reasoning_content": "mock visible reasoning",
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = Path(extra_env["BBO_NANOBOT_REASONING_METADATA_PATH"])
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "call_id": call_id,
                            "reasoning_visible": True,
                            "reasoning_chars": len("mock visible reasoning"),
                            "trace_path": str(trace_path),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return result


def test_general_agent_algorithms_are_registered_and_cli_visible() -> None:
    parser = build_arg_parser()
    algorithm_action = next(action for action in parser._actions if action.dest == "algorithm")

    assert "agentic_nanobot" in ALGORITHM_REGISTRY
    assert "nanobot" in ALGORITHM_REGISTRY
    assert "agentic_claude_code" in ALGORITHM_REGISTRY
    assert "claude_code" in ALGORITHM_REGISTRY
    assert "claude-code" in ALGORITHM_REGISTRY
    assert "agentic_openai_compatible" in ALGORITHM_REGISTRY
    assert "openai_compatible_agent" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["nanobot"].family == "agentic"
    assert ALGORITHM_REGISTRY["claude_code"].family == "agentic"
    assert ALGORITHM_REGISTRY["agentic_openai_compatible"].family == "agentic"
    assert "nanobot" in algorithm_action.choices
    assert "claude_code" in algorithm_action.choices
    assert "agentic_openai_compatible" in algorithm_action.choices
    assert parser.parse_args(["--algorithm", "claude-code"]).algorithm == "claude-code"
    assert parser.parse_args(["--algorithm", "agentic_openai_compatible"]).agent_tool_mode == "function_calling"
    assert parser.parse_args(["--algorithm", "nanobot", "--agent-require-visible-cot"]).agent_require_visible_cot is True


def test_parse_agent_candidate_payload_accepts_config_wrappers() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {"config": {"x1": 0.5, "x2": 0.5}, "rationale": "center probe"},
                {"config": {"x1": 0.1, "x2": 0.9}},
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}, {"x1": 0.1, "x2": 0.9}]
    assert candidates[0].metadata["rationale"] == "center probe"


def test_parse_agent_candidate_payload_accepts_cli_preamble() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)
    payload = """
    Using config: /tmp/config.json

    nanobot
    {"candidates": [{"config": {"x1": 0.5, "x2": 0.5}}]}
    """

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}]


def test_parse_agent_candidate_payload_repairs_wrapped_string_newlines() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)
    payload = '''nanobot
    {
      "candidates": [
        {
          "config": {"x1": 0.5, "x2": 0.5},
          "rationale": "line one
line two"
        }
      ]
    }
    '''

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert candidates[0].config == {"x1": 0.5, "x2": 0.5}
    assert "line two" in candidates[0].metadata["rationale"]


def test_parse_agent_candidate_payload_skips_invalid_candidates() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)
    payload = json.dumps(
        {
            "candidates": [
                {"config": {"x1": 0.5, "x2": 0.5}},
                {"config": {"x1": 11.0, "x2": 0.5}},
            ]
        }
    )

    candidates = parse_agent_candidate_payload(payload, task.spec.search_space)

    assert [candidate.config for candidate in candidates] == [{"x1": 0.5, "x2": 0.5}]


def test_parse_agent_candidate_payload_rejects_markdown_and_invalid_configs() -> None:
    task = create_task("branin_demo", max_evaluations=3, seed=7)

    with pytest.raises(GeneralAgentValidationError, match="raw JSON"):
        parse_agent_candidate_payload('```json\n{"candidates": []}\n```', task.spec.search_space)

    with pytest.raises(GeneralAgentValidationError, match="expects"):
        parse_agent_candidate_payload(
            json.dumps({"candidates": [{"config": {"x1": 11.0, "x2": 0.5}}]}),
            task.spec.search_space,
        )


def test_nanobot_bbo_algorithm_with_mock_engine_writes_artifacts(tmp_path: Path) -> None:
    task = create_task("branin_demo", max_evaluations=4, seed=3)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=11),
        run_dir=tmp_path / "agent_run",
        timeout_seconds=5.0,
        candidates_per_call=3,
    )
    logger = JsonlMetricLogger(tmp_path / "trials.jsonl")

    summary = Experimenter(
        task=task,
        algorithm=algorithm,
        logger_backend=logger,
        config=ExperimentConfig(seed=3, resume=False, fail_fast_on_sanity=True),
    ).run()

    assert summary.n_completed == 4
    assert summary.best_primary_objective is not None
    artifacts = algorithm.artifact_paths
    assert Path(artifacts["agent_workspace"]).exists()
    assert Path(artifacts["agent_calls_jsonl"]).exists()
    assert Path(artifacts["llm_logs_dir"]).exists()
    assert Path(artifacts["agent_llm_logs_dir"]) == Path(artifacts["llm_logs_dir"])
    assert Path(artifacts["agent_state_json"]).exists()
    assert Path(artifacts["agent_manifest_json"]).exists()
    assert Path(artifacts["agent_tool_specs_json"]).exists()
    assert Path(artifacts["agent_workspace_tool_py"]).exists()
    assert Path(artifacts["agent_workspace_bbo_tools_py"]).exists()
    assert Path(artifacts["agent_workspace_tool_config_json"]).exists()
    assert Path(artifacts["agent_workspace_gp_example_py"]).exists()
    assert Path(artifacts["agent_workspace_python_environment_md"]).exists()
    assert Path(artifacts["agent_tool_calls_jsonl"]).exists()
    assert Path(artifacts["agent_memory_jsonl"]).parent.exists()
    assert Path(artifacts["agent_reasoning_traces_dir"]).exists()
    assert Path(artifacts["agent_reasoning_metadata_jsonl"]).exists()
    assert (Path(artifacts["agent_workspace"]) / "instructions.md").exists()
    assert (Path(artifacts["agent_workspace"]) / "manifest.json").exists()
    assert (Path(artifacts["agent_workspace"]) / "tool_specs.json").exists()
    assert Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").strip()
    records = logger.load_records()
    assert records[0].suggestion_metadata["agent_framework"] == "nanobot"
    assert records[0].suggestion_metadata["agent_engine"] == "mock"


def test_workspace_tool_bridge_calls_every_advertised_tool(tmp_path: Path) -> None:
    task = create_task("branin_demo", max_evaluations=4, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        code_backend="mock",
        web_search_provider="mock",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    artifacts = algorithm.artifact_paths
    workspace = Path(artifacts["agent_workspace"])
    tool_script = Path(artifacts["agent_workspace_tool_py"])
    assert "from bbo." not in tool_script.read_text(encoding="utf-8")
    assert "from bbo." not in Path(artifacts["agent_workspace_bbo_tools_py"]).read_text(encoding="utf-8")
    calls = {
        "get_task_context": {"max_chars_per_section": 500},
        "get_search_space": {},
        "get_trial_history": {"mode": "all", "limit": 5},
        "get_incumbent": {},
        "validate_candidates": {"candidates": [{"config": {"x1": 0.0, "x2": 0.0}}]},
        "sample_candidates": {"n": 2, "seed": 5},
        "analyze_history": {},
        "memory_write": {"kind": "note", "content": "workspace bridge probe", "tags": ["healthcheck"]},
        "memory_read": {"tags": ["healthcheck"]},
        "code_interpreter": {"code": "print(1)", "language": "python"},
        "web_search": {"query": "branin optimization prior", "limit": 1},
        "fetch_url": {"url": "https://example.com", "max_chars": 200},
    }

    for tool_name, arguments in calls.items():
        completed = subprocess.run(
            [sys.executable, str(tool_script), tool_name, json.dumps(arguments)],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True

    for tool_name in ("get_space", "get_history", "get_objective", "get_tool_specs", "get_manifest"):
        completed = subprocess.run(
            [sys.executable, str(tool_script), tool_name, "{}"],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True

    records = [
        json.loads(line)
        for line in Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {record["tool_name"] for record in records} >= set(calls)
    assert all(record["success"] is True for record in records)


def test_workspace_python_api_and_gp_example_run_in_workspace(tmp_path: Path) -> None:
    task = create_task("branin_demo", max_evaluations=4, seed=5)
    algorithm = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=23),
        run_dir=tmp_path / "agent_run",
        tool_mode="workspace_json",
        code_backend="mock",
        web_search_provider="mock",
    )
    algorithm.setup(task.spec, seed=5, task_description=task.get_description())
    artifacts = algorithm.artifact_paths
    workspace = Path(artifacts["agent_workspace"])

    api_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from bbo_tools import BBO; "
                "bbo = BBO(); "
                "payload = {"
                "'dimension': bbo.search_space()['dimension'], "
                "'history_total': bbo.history(mode='all')['total'], "
                "'valid_count': bbo.validate([{'config': {'x1': 0.0, 'x2': 0.0}}])['valid_count'], "
                "'code_backend': bbo.code_interpreter('print(1)')['backend'], "
                "'search_enabled': bbo.web_search('branin optimization', limit=1)['enabled']"
                "}; "
                "print(json.dumps(payload, sort_keys=True))"
            ),
        ],
        cwd=workspace,
        check=False,
        text=True,
        capture_output=True,
    )
    assert api_probe.returncode == 0, api_probe.stderr
    api_payload = json.loads(api_probe.stdout)
    assert api_payload["dimension"] == 2
    assert api_payload["valid_count"] == 1
    assert api_payload["code_backend"] == "mock"
    assert api_payload["search_enabled"] is True

    gp_probe = subprocess.run(
        [sys.executable, str(Path(artifacts["agent_workspace_gp_example_py"]))],
        cwd=workspace,
        check=False,
        text=True,
        capture_output=True,
    )
    assert gp_probe.returncode == 0, gp_probe.stderr
    gp_payload = json.loads(gp_probe.stdout)
    assert set(gp_payload) == {"candidates"}
    assert 1 <= len(gp_payload["candidates"]) <= 4
    for item in gp_payload["candidates"]:
        task.spec.search_space.validate_config(item["config"])

    records = [
        json.loads(line)
        for line in Path(artifacts["agent_tool_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record["interface"] == "workspace_python_api" for record in records)
    assert any(record["tool_name"] == "validate_candidates" for record in records)


def test_general_agent_visible_cot_requirement_uses_reasoning_metadata(tmp_path: Path) -> None:
    task = create_task("branin_demo", max_evaluations=4, seed=8)
    missing = NanobotBBOAlgorithm(
        engine=MockAgentEngine(seed=31),
        run_dir=tmp_path / "missing_reasoning",
        tool_mode="workspace_json",
        require_visible_cot=True,
        max_retries=0,
    )
    missing.setup(task.spec, seed=8, task_description=task.get_description())
    with pytest.raises(RuntimeError, match="visible CoT"):
        missing.ask()

    present = NanobotBBOAlgorithm(
        engine=ReasoningMetadataMockEngine(seed=32),
        run_dir=tmp_path / "present_reasoning",
        tool_mode="workspace_json",
        require_visible_cot=True,
        max_retries=0,
    )
    present.setup(task.spec, seed=8, task_description=task.get_description())
    suggestion = present.ask()
    task.spec.search_space.validate_config(suggestion.config)
    metadata_records = [
        json.loads(line)
        for line in Path(present.artifact_paths["agent_reasoning_metadata_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metadata_records
    assert metadata_records[-1]["reasoning_visible"] is True
    call_records = [
        json.loads(line)
        for line in Path(present.artifact_paths["agent_calls_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert call_records[-1]["reasoning"]["reasoning_visible"] is True


def test_general_agent_replay_resume_extends_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "agent_run"
    results_path = tmp_path / "trials.jsonl"

    first_task = create_task("branin_demo", max_evaluations=2, seed=4)
    first_algorithm = NanobotBBOAlgorithm(engine=MockAgentEngine(seed=17), run_dir=run_dir, resume=False)
    first_logger = JsonlMetricLogger(results_path)
    Experimenter(
        task=first_task,
        algorithm=first_algorithm,
        logger_backend=first_logger,
        config=ExperimentConfig(seed=4, resume=False, fail_fast_on_sanity=True),
    ).run()

    second_task = create_task("branin_demo", max_evaluations=3, seed=4)
    second_algorithm = NanobotBBOAlgorithm(engine=MockAgentEngine(seed=17), run_dir=run_dir, resume=True)
    second_logger = JsonlMetricLogger(results_path)
    summary = Experimenter(
        task=second_task,
        algorithm=second_algorithm,
        logger_backend=second_logger,
        config=ExperimentConfig(seed=4, resume=True, fail_fast_on_sanity=True),
    ).run()

    assert summary.n_completed == 3
    assert len(second_logger.load_records()) == 3
    state = json.loads((run_dir / "agent_state.json").read_text(encoding="utf-8"))
    assert state["history_size"] == 3
