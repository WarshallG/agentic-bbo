"""ClawArena-inspired agent engines for general-agent BBO methods."""

from __future__ import annotations

import asyncio
import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


BBOToolExecutor = Callable[[str, dict[str, Any], str | None], Awaitable[str]]


@dataclass
class AgentResult:
    """Result of one external agent invocation."""

    status: Literal["success", "failed", "timeout"]
    answer: str
    error: str | None = None
    returncode: int | None = None
    raw: Any = None
    llm_log: dict[str, Any] | None = None


@dataclass
class AgentWorkCopy:
    """Workspace and framework state handed to one agent engine."""

    state_dir: Path
    config_path: Path | None
    project_root: Path
    workspace_root: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class GeneralAgentEngine(ABC):
    """Minimal async agent execution interface borrowed from ClawArena."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework name surfaced in logs."""

    @abstractmethod
    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        """Execute a single agent call."""


class NanobotEngine(GeneralAgentEngine):
    """Nanobot engine using the ClawArena compatibility runner."""

    @property
    def name(self) -> str:
        return "nanobot"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del tool_executor, max_tool_calls
        if tools:
            return AgentResult(
                status="failed",
                answer="",
                error="NanobotEngine does not support injected BBO function-calling tools in this runtime.",
                returncode=-2,
            )
        cfg = work_copy.extra.get("nanobot_config", {})
        workspace_path = _resolve_workspace(work_copy, agent_id or "")
        cmd = [sys.executable, "-m", "bbo.algorithms.agentic.nanobot_runner", "agent", "-m", message, "--no-markdown"]
        if session_id:
            cmd.extend(["-s", session_id])
        if workspace_path:
            cmd.extend(["-w", str(workspace_path)])
        if work_copy.config_path:
            cmd.extend(["-c", str(work_copy.config_path)])

        env = {
            **os.environ,
            **(cfg.get("env") or {}),
            **(extra_env or {}),
            "BBO_NANOBOT_NO_MAX_TOKENS": "1",
        }
        if log_dir := work_copy.extra.get("log_dir"):
            env["BBO_NANOBOT_LOG_DIR"] = str(log_dir)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            cwd=str(workspace_path) if workspace_path else str(work_copy.project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return AgentResult(
                status="timeout",
                answer="",
                error=f"Timeout after {timeout}s",
                returncode=-1,
            )
        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()
        error_text = None
        if proc.returncode != 0:
            error_text = "\n".join(part for part in (stderr_text, stdout_text) if part) or None
        return AgentResult(
            status="success" if proc.returncode == 0 else "failed",
            answer=stdout_text,
            error=error_text,
            returncode=proc.returncode,
        )


class ClaudeCodeEngine(GeneralAgentEngine):
    """Claude Code engine using ``claude_agent_sdk`` when available."""

    @property
    def name(self) -> str:
        return "claude_code"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del tool_executor, max_tool_calls
        if tools:
            return AgentResult(
                status="failed",
                answer="",
                error="ClaudeCodeEngine does not support injected BBO function-calling tools in this runtime.",
                returncode=-2,
            )
        try:
            from claude_agent_sdk import (  # type: ignore
                AssistantMessage,
                ClaudeAgentOptions,
                CLINotFoundError,
                ProcessError,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:
            return AgentResult(
                status="failed",
                answer="",
                error=(
                    "Claude Code backend requires `claude-agent-sdk`. "
                    "Install the agent dependency before using `claude_code`."
                ),
                raw=exc,
            )

        cc = work_copy.extra.get("claude_config", {})
        workspace_path = _resolve_workspace(work_copy, agent_id or "")
        stderr_lines: list[str] = []

        def _collect_stderr(line: str) -> None:
            stderr_lines.append(line)

        opts = ClaudeAgentOptions(
            cwd=str(workspace_path) if workspace_path else str(work_copy.project_root),
            env={
                "CLAUDE_CONFIG_DIR": str(work_copy.state_dir),
                **(cc.get("env") or {}),
                **(extra_env or {}),
            },
            permission_mode=cc.get("permission_mode"),
            allowed_tools=cc.get("allowed_tools", []),
            disallowed_tools=cc.get("disallowed_tools", []),
            model=cc.get("model"),
            max_turns=cc.get("max_turns"),
            stderr=_collect_stderr,
        )
        if session_id:
            opts.resume = session_id

        async def _query_once() -> AgentResult:
            answer_parts: list[str] = []
            messages: list[dict[str, Any]] = []
            result_msg: ResultMessage | None = None
            async for msg in query(prompt=message, options=opts):
                if isinstance(msg, AssistantMessage):
                    messages.append(_serialize_assistant_message(msg))
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            answer_parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    result_msg = msg

            answer_text = "\n".join(part for part in answer_parts if part).strip()
            llm_log = _build_claude_llm_log(messages, result_msg, session_id, agent_id)
            if result_msg is not None and result_msg.is_error:
                return AgentResult(
                    status="failed",
                    answer=answer_text,
                    error=str(result_msg.result),
                    raw=result_msg,
                    llm_log=llm_log,
                )
            return AgentResult(status="success", answer=answer_text, raw=result_msg, llm_log=llm_log)

        try:
            if timeout is None:
                return await _query_once()
            return await asyncio.wait_for(_query_once(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return AgentResult(status="timeout", answer="", error=f"Timeout after {timeout}s", returncode=-1)
        except CLINotFoundError as exc:
            return AgentResult(status="failed", answer="", error=str(exc))
        except ProcessError as exc:
            stderr_text = "\n".join(stderr_lines) if stderr_lines else exc.stderr
            return AgentResult(
                status="failed",
                answer="",
                error=f"exit={exc.exit_code}: {stderr_text}",
                returncode=exc.exit_code,
            )
        except Exception as exc:  # pragma: no cover - depends on external agent behavior.
            stderr_text = "\n".join(stderr_lines)
            detail = f"{exc}"
            if stderr_text:
                detail = f"{detail}\nstderr: {stderr_text}"
            return AgentResult(status="failed", answer="", error=detail)


class OpenAICompatibleToolEngine(GeneralAgentEngine):
    """OpenAI-compatible chat-completions engine with BBO function-calling support."""

    @property
    def name(self) -> str:
        return "openai_compatible"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del session_id, agent_id, extra_env
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency.
            return AgentResult(
                status="failed",
                answer="",
                error="OpenAI-compatible BBO engine requires the optional `pablo` extra (`openai>=1.0`).",
                raw=exc,
            )
        cfg = work_copy.extra.get("openai_compatible_config", {})
        api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return AgentResult(status="failed", answer="", error="OpenAI-compatible BBO engine requires an API key.")
        model = cfg.get("model") or "gpt-4.1-mini"
        client = AsyncOpenAI(api_key=api_key, base_url=cfg.get("api_base"))
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
        tool_calls_used = 0

        async def _query_once() -> AgentResult:
            nonlocal tool_calls_used
            while True:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                try:
                    response = await client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except TypeError:
                    response = await client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                tool_calls = list(getattr(msg, "tool_calls", None) or [])
                content = getattr(msg, "content", None) or ""
                if not tool_calls:
                    return AgentResult(status="success", answer=str(content), raw=response)
                if tool_executor is None:
                    return AgentResult(status="failed", answer=str(content), error="Model requested tools but no BBO tool executor was provided.")
                if tool_calls_used >= max_tool_calls:
                    return AgentResult(status="failed", answer=str(content), error=f"Exceeded max BBO tool calls ({max_tool_calls}).")
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    if tool_calls_used >= max_tool_calls:
                        break
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = await tool_executor(call.function.name, arguments, call.id)
                    tool_calls_used += 1
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        try:
            if timeout is None:
                return await _query_once()
            return await asyncio.wait_for(_query_once(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return AgentResult(status="timeout", answer="", error=f"Timeout after {timeout}s", returncode=-1)
        except Exception as exc:  # pragma: no cover - provider-specific failures.
            return AgentResult(status="failed", answer="", error=str(exc))


class MockAgentEngine(GeneralAgentEngine):
    """Deterministic local agent used by tests and offline examples."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    async def run_agent(
        self,
        session_id: str,
        message: str,
        work_copy: AgentWorkCopy,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: BBOToolExecutor | None = None,
        max_tool_calls: int = 0,
    ) -> AgentResult:
        del session_id, message, agent_id, timeout, extra_env, max_tool_calls
        import json

        if tools and tool_executor is not None:
            sample_raw = await tool_executor(
                "sample_candidates",
                {"n": 4, "seed": self.seed + self.calls, "strategy": "random"},
                "mock_sample_candidates",
            )
            self.calls += 1
            try:
                sample_payload = json.loads(sample_raw)
                candidates = [
                    {"config": item["config"], "rationale": "mock tool sample"}
                    for item in sample_payload["result"]["candidates"]
                ]
            except Exception:
                candidates = []
            if candidates:
                return AgentResult(status="success", answer=json.dumps({"candidates": candidates}, sort_keys=True))

        space_path = (work_copy.workspace_root or work_copy.project_root) / "space.json"
        payload = json.loads(space_path.read_text(encoding="utf-8"))
        rng = random.Random(self.seed + self.calls)
        self.calls += 1
        candidates = []
        for _ in range(4):
            config: dict[str, Any] = {}
            for param in payload["parameters"]:
                if param["type"] == "float":
                    config[param["name"]] = rng.uniform(float(param["low"]), float(param["high"]))
                elif param["type"] == "int":
                    config[param["name"]] = rng.randint(int(param["low"]), int(param["high"]))
                elif param["type"] == "categorical":
                    config[param["name"]] = rng.choice(list(param["choices"]))
                else:
                    raise ValueError(f"Unsupported mock parameter type: {param['type']}")
            candidates.append({"config": config, "rationale": "mock deterministic sample"})
        return AgentResult(status="success", answer=json.dumps({"candidates": candidates}, sort_keys=True))


def create_general_agent_engine(framework: str) -> GeneralAgentEngine:
    normalized = normalize_agent_framework(framework)
    if normalized == "nanobot":
        return NanobotEngine()
    if normalized == "claude_code":
        return ClaudeCodeEngine()
    if normalized == "openai_compatible":
        return OpenAICompatibleToolEngine()
    if normalized == "mock":
        return MockAgentEngine()
    raise ValueError(f"Unknown general-agent framework `{framework}`.")


def normalize_agent_framework(framework: str) -> str:
    normalized = framework.strip().lower().replace("-", "_")
    if normalized in {"claude", "claude_code", "claudecode"}:
        return "claude_code"
    if normalized in {"nanobot", "nano_bot"}:
        return "nanobot"
    if normalized in {"openai", "openai_compatible", "openai_compat"}:
        return "openai_compatible"
    if normalized == "mock":
        return "mock"
    return normalized


def _resolve_workspace(work_copy: AgentWorkCopy, agent_id: str) -> Path | None:
    if work_copy.workspace_root and agent_id:
        candidate = work_copy.workspace_root / agent_id
        if candidate.exists():
            return candidate
    return work_copy.workspace_root


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [_serialize_claude_block(block) for block in msg.content],
        **({"model": msg.model} if getattr(msg, "model", None) else {}),
        **({"usage": msg.usage} if getattr(msg, "usage", None) else {}),
        **({"stopReason": msg.stop_reason} if getattr(msg, "stop_reason", None) else {}),
    }


def _serialize_claude_block(block: Any) -> dict[str, Any]:
    try:
        from claude_agent_sdk import TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock  # type: ignore
    except ImportError:  # pragma: no cover - guarded by caller imports.
        return {"type": "unknown", "data": str(block)}
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "toolResult",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    return {"type": "unknown", "data": str(block)}


def _build_claude_llm_log(
    messages: list[dict[str, Any]],
    result_msg: Any,
    session_id: str | None,
    agent_id: str | None,
) -> dict[str, Any]:
    log: dict[str, Any] = {
        "agentId": agent_id or "",
        "sessionId": session_id or getattr(result_msg, "session_id", "") or "",
        "success": bool(result_msg and getattr(result_msg, "subtype", "") == "success"),
        "messageCount": len(messages),
        "messages": messages,
    }
    if result_msg is not None:
        for source, target in (
            ("duration_ms", "durationMs"),
            ("num_turns", "numTurns"),
            ("usage", "usage"),
            ("total_cost_usd", "totalCostUsd"),
            ("model_usage", "modelUsage"),
        ):
            value = getattr(result_msg, source, None)
            if value is not None:
                log[target] = value
    return log


__all__ = [
    "AgentResult",
    "AgentWorkCopy",
    "ClaudeCodeEngine",
    "GeneralAgentEngine",
    "MockAgentEngine",
    "NanobotEngine",
    "OpenAICompatibleToolEngine",
    "create_general_agent_engine",
    "normalize_agent_framework",
]
