"""BBO code execution tools backed by SandboxFusion-compatible services."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from .base import BaseBBOTool
from .context import BBOToolContext


@dataclass
class SandboxFusionBBOCodeBackend:
    """HTTP backend for Bytedance SandboxFusion's `/run_code` API."""

    base_url: str
    timeout_seconds: float = 120.0

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        return await asyncio.to_thread(self._execute_blocking, code=code, language=language)

    def _execute_blocking(self, *, code: str, language: str) -> dict[str, Any]:
        url = urljoin(self.base_url.rstrip("/") + "/", "run_code")
        body = json.dumps({"code": code, "language": language}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            return {"status": "Error", "message": f"SandboxFusion request failed: {exc}", "run_result": None}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "Error", "message": "SandboxFusion returned non-JSON response.", "raw": raw}
        return payload if isinstance(payload, dict) else {"status": "Error", "message": "Unexpected response shape."}


class DisabledBBOCodeBackend:
    """Code backend that reports a clear disabled result."""

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        del code, language
        return {
            "status": "Disabled",
            "message": "BBO code execution is disabled. Configure SandboxFusion to enable this tool.",
            "run_result": None,
        }


@dataclass
class MockBBOCodeBackend:
    """Deterministic backend for tests."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    async def execute(self, *, code: str, language: str = "python") -> dict[str, Any]:
        return {
            "status": "Success",
            "message": "",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "execution_time": 0.0,
                "return_code": self.return_code,
                "stdout": self.stdout or f"mock {language}: {len(code)} chars\n",
                "stderr": self.stderr,
            },
            "files": {},
        }


class CodeInterpreterTool(BaseBBOTool):
    name = "code_interpreter"
    description = (
        "Run analysis code in the configured BBO sandbox. This tool must not call task evaluators or consume trial budget."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Code to execute for offline analysis."},
            "language": {"type": "string", "default": "python", "description": "SandboxFusion language id."},
        },
        "required": ["code"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        code: str,
        language: str = "python",
        **_: Any,
    ) -> dict[str, Any]:
        if not code.strip():
            raise ValueError("code must be non-empty.")
        policy = context.manifest.tool_policy.get("code_interpreter", {})
        if isinstance(policy, dict) and policy.get("enabled") is False and context.code_backend is None:
            return {
                "enabled": False,
                "message": "The BBO manifest disables code_interpreter for this benchmark.",
            }
        backend = context.code_backend or DisabledBBOCodeBackend()
        if not hasattr(backend, "execute"):
            raise TypeError("context.code_backend must provide an async execute(code=..., language=...) method.")
        result = await backend.execute(code=code, language=language)  # type: ignore[attr-defined]
        return {
            "backend": type(backend).__name__,
            "language": language,
            "sandbox_result": result,
            "budget_consumed": False,
        }


__all__ = [
    "CodeInterpreterTool",
    "DisabledBBOCodeBackend",
    "MockBBOCodeBackend",
    "SandboxFusionBBOCodeBackend",
]
