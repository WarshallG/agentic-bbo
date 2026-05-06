"""Workspace-local Python API for BBO shell/file agents.

This file is copied into each agent workspace as ``bbo_tools.py``.  It must stay
free of imports from the installed ``bbo`` package because shell/file agents may
run it from an isolated workspace or a framework-created environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bbo_tool as _bridge


class BBOToolError(RuntimeError):
    """Raised when a workspace BBO tool call fails."""


class BBO:
    """Python API for the BBO workspace tool bridge."""

    def __init__(self, config_path: str | Path = "bbo_tool_config.json") -> None:
        path = Path(config_path)
        self.config_path = path if path.is_absolute() else Path.cwd() / path

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one BBO workspace tool and return its result object."""

        args = dict(arguments or {})
        raw_arguments = json.dumps(args, ensure_ascii=False, sort_keys=True)
        config = _bridge._read_json(self.config_path)
        started = time.monotonic()
        timestamp = time.time()
        try:
            result = _bridge._execute(tool_name, args, config)
            payload = {"ok": True, "result": result}
            success = True
        except Exception as exc:
            payload = {"ok": False, "error": "exception", "message": str(exc)}
            success = False
        _bridge._log_call(
            config,
            tool_name,
            raw_arguments,
            payload,
            started,
            timestamp,
            success,
            interface="workspace_python_api",
        )
        if not success:
            raise BBOToolError(f"{tool_name} failed: {payload['message']}")
        return payload["result"]

    def task_context(self, **kwargs: Any) -> dict[str, Any]:
        return self.call("get_task_context", kwargs)

    def manifest(self) -> dict[str, Any]:
        return self.call("get_manifest", {})

    def search_space(self) -> dict[str, Any]:
        return self.call("get_search_space", {})

    def objective(self) -> dict[str, Any]:
        return self.call("get_objective", {})

    def tool_specs(self) -> dict[str, Any]:
        return self.call("get_tool_specs", {})

    def history(self, mode: str = "recent", limit: int = 20, offset: int = 0) -> dict[str, Any]:
        return self.call("get_trial_history", {"mode": mode, "limit": limit, "offset": offset})

    def incumbent(self) -> dict[str, Any]:
        return self.call("get_incumbent", {})

    def validate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call("validate_candidates", {"candidates": candidates})

    def sample(
        self,
        n: int = 4,
        *,
        strategy: str = "random",
        seed: int | None = None,
        jitter_fraction: float = 0.1,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"n": n, "strategy": strategy, "jitter_fraction": jitter_fraction}
        if seed is not None:
            args["seed"] = seed
        return self.call("sample_candidates", args)

    def analyze_history(self, limit: int = 100) -> dict[str, Any]:
        return self.call("analyze_history", {"limit": limit})

    def memory_read(
        self,
        *,
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"limit": limit}
        if kind is not None:
            args["kind"] = kind
        if tags is not None:
            args["tags"] = tags
        return self.call("memory_read", args)

    def memory_write(
        self,
        *,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        source_call_id: str | None = None,
        trial_range: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"kind": kind, "content": content}
        if tags is not None:
            args["tags"] = tags
        if source_call_id is not None:
            args["source_call_id"] = source_call_id
        if trial_range is not None:
            args["trial_range"] = trial_range
        if metadata is not None:
            args["metadata"] = metadata
        return self.call("memory_write", args)

    def code_interpreter(self, code: str, language: str = "python") -> dict[str, Any]:
        return self.call("code_interpreter", {"code": code, "language": language})

    def web_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        return self.call("web_search", {"query": query, "limit": limit})

    def fetch_url(self, url: str, max_chars: int = 4000) -> dict[str, Any]:
        return self.call("fetch_url", {"url": url, "max_chars": max_chars})


__all__ = ["BBO", "BBOToolError"]
