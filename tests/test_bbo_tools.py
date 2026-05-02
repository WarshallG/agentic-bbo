from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bbo.algorithms.agentic.tools import (
    BBOMemoryStore,
    BBOToolCallLogger,
    BBOToolContext,
    BBOToolRegistry,
    BBOWebSourceLogger,
    CodeInterpreterTool,
    MockBBOCodeBackend,
    MockBBOWebSearchProvider,
    SearchR1BBOWebSearchProvider,
    WebSearchTool,
    create_core_BBO_tools,
)
from bbo.core import Incumbent, TrialObservation, TrialStatus, TrialSuggestion, load_BBO_manifest
from bbo.tasks import create_task


def _run(coro):
    return asyncio.run(coro)


class SearchR1MockHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append({"path": self.path, "payload": payload})
        body = json.dumps(
            {
                "result": [
                    [
                        {
                            "document": {
                                "contents": '"Branin prior"\nPrefer structured exploration near known basins.',
                                "url": "https://example.test/branin",
                            },
                            "score": 0.87,
                        }
                    ]
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _tool_context(tmp_path: Path) -> BBOToolContext:
    task = create_task("branin_demo", max_evaluations=6, seed=3)
    description = task.get_description()
    history = [
        TrialObservation(
            suggestion=TrialSuggestion(config={"x1": 0.2, "x2": 0.8}, trial_id=0),
            status=TrialStatus.SUCCESS,
            objectives={"loss": 10.0},
        ),
        TrialObservation(
            suggestion=TrialSuggestion(config={"x1": 0.6, "x2": 0.4}, trial_id=1),
            status=TrialStatus.SUCCESS,
            objectives={"loss": 4.0},
        ),
    ]
    incumbent = Incumbent(config={"x1": 0.6, "x2": 0.4}, score=4.0, objectives={"loss": 4.0}, trial_id=1)
    return BBOToolContext(
        task_spec=task.spec,
        description=description,
        manifest=load_BBO_manifest(task.spec),
        workspace_dir=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        history=history,
        incumbent=incumbent,
        memory_store=BBOMemoryStore(tmp_path / "memory" / "memory.jsonl", tmp_path / "memory" / "summary.json"),
        code_backend=MockBBOCodeBackend(stdout="42\n"),
        web_search_provider=MockBBOWebSearchProvider(),
        source_logger=BBOWebSourceLogger(tmp_path / "sources.jsonl"),
        seed=13,
    )


def test_BBO_tool_registry_specs_and_logging(tmp_path: Path) -> None:
    context = _tool_context(tmp_path)
    registry = BBOToolRegistry(
        create_core_BBO_tools(),
        logger=BBOToolCallLogger(tmp_path / "tool_calls.jsonl"),
    )

    specs = registry.get_tool_specs()
    assert {spec["function"]["name"] for spec in specs} >= {"get_task_context", "validate_candidates", "memory_write"}

    raw = _run(registry.execute_tool("get_search_space", {}, context, call_id="call-1", tool_call_id="tool-1"))
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["result"]["dimension"] == 2
    records = (tmp_path / "tool_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["tool_name"] == "get_search_space"


def test_BBO_candidate_validation_sampling_and_history_analysis(tmp_path: Path) -> None:
    context = _tool_context(tmp_path)
    registry = BBOToolRegistry(create_core_BBO_tools())

    raw = _run(
        registry.execute_tool(
            "validate_candidates",
            {
                "candidates": [
                    {"config": {"x1": 0.6, "x2": 0.4}},
                    {"config": {"x1": 11.0, "x2": 0.5}},
                    {"x1": 0.1, "x2": 0.9},
                ]
            },
            context,
        )
    )
    validation = json.loads(raw)["result"]
    assert validation["valid_count"] == 2
    assert validation["invalid_count"] == 1
    assert validation["valid"][0]["duplicate"] is True

    sampled = json.loads(
        _run(registry.execute_tool("sample_candidates", {"n": 3, "strategy": "around_incumbent", "seed": 9}, context))
    )["result"]
    assert sampled["count"] == 3
    for item in sampled["candidates"]:
        context.task_spec.search_space.validate_config(item["config"])

    analysis = json.loads(_run(registry.execute_tool("analyze_history", {}, context)))["result"]
    assert analysis["best_trial"]["trial_id"] == 1
    assert "x1" in analysis["numeric_correlations"]


def test_BBO_memory_tools_are_append_only(tmp_path: Path) -> None:
    context = _tool_context(tmp_path)
    registry = BBOToolRegistry(create_core_BBO_tools())

    written = json.loads(
        _run(
            registry.execute_tool(
                "memory_write",
                {"kind": "hypothesis", "content": "Try local perturbations near the incumbent.", "tags": ["local"]},
                context,
            )
        )
    )["result"]
    read_back = json.loads(_run(registry.execute_tool("memory_read", {"tags": ["local"]}, context)))["result"]

    assert written["written"] is True
    assert read_back["count"] == 1
    assert read_back["records"][0]["kind"] == "hypothesis"
    assert (tmp_path / "memory" / "summary.json").exists()


def test_BBO_code_and_web_tools_use_pluggable_backends(tmp_path: Path) -> None:
    context = _tool_context(tmp_path)
    registry = BBOToolRegistry([CodeInterpreterTool(), WebSearchTool()])

    code_result = json.loads(
        _run(registry.execute_tool("code_interpreter", {"code": "print(6 * 7)", "language": "python"}, context))
    )["result"]
    assert code_result["backend"] == "MockBBOCodeBackend"
    assert code_result["sandbox_result"]["run_result"]["stdout"] == "42\n"
    assert code_result["budget_consumed"] is False

    search_result = json.loads(
        _run(registry.execute_tool("web_search", {"query": "BBOPlace macro placement prior", "limit": 3}, context))
    )["result"]
    assert search_result["count"] == 1
    assert search_result["results"][0]["source_id"].startswith("src_")
    source_records = (tmp_path / "sources.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(source_records[0])["kind"] == "search_result"


def test_Search_R1_web_provider_calls_retrieve_api(tmp_path: Path) -> None:
    SearchR1MockHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SearchR1MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        context = _tool_context(tmp_path)
        context.web_search_provider = SearchR1BBOWebSearchProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            timeout_seconds=5.0,
        )
        registry = BBOToolRegistry([WebSearchTool()])

        result = json.loads(
            _run(registry.execute_tool("web_search", {"query": "branin optimization prior", "limit": 3}, context))
        )["result"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert SearchR1MockHandler.requests == [
        {
            "path": "/retrieve",
            "payload": {"queries": ["branin optimization prior"], "topk": 3, "return_scores": True},
        }
    ]
    assert result["count"] == 1
    assert result["results"][0]["title"] == "Branin prior"
    assert result["results"][0]["score"] == 0.87
