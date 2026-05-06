"""Nanobot CLI runner with compatibility patches for BBO agent calls."""

from __future__ import annotations

import os


def _patch_strip_max_tokens() -> None:
    """Strip legacy ``max_tokens`` for compatible endpoints that reject it."""

    try:
        from nanobot.providers import openai_compat_provider as provider_module  # type: ignore

        original = provider_module.OpenAICompatProvider._build_kwargs

        def patched(
            self,
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        ):
            kwargs = original(
                self,
                messages,
                tools,
                model,
                max_tokens,
                temperature,
                reasoning_effort,
                tool_choice,
            )
            kwargs.pop("max_tokens", None)
            return kwargs

        provider_module.OpenAICompatProvider._build_kwargs = patched
    except Exception:
        pass


def _patch_log_llm() -> None:
    """Write nanobot's final message snapshot when a log directory is provided."""

    import contextvars
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    log_root = Path(os.environ["BBO_NANOBOT_LOG_DIR"])
    session_key_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "bbo_nanobot_session_key",
        default=None,
    )

    def iso_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def filename_ts() -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}Z"

    def reasoning_tokens(usage: dict) -> int | None:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            try:
                return int(details["reasoning_tokens"])
            except Exception:
                return None
        if usage.get("reasoning_tokens") is not None:
            try:
                return int(usage["reasoning_tokens"])
            except Exception:
                return None
        return None

    def reasoning_entries(messages: list) -> list[dict]:
        entries = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            reasoning = message.get("reasoning_content")
            thinking_blocks = message.get("thinking_blocks")
            if not reasoning and not thinking_blocks:
                continue
            entry = {
                "message_index": index,
                "reasoning_content": reasoning if isinstance(reasoning, str) else "",
                "thinking_blocks": thinking_blocks if isinstance(thinking_blocks, list) else [],
                "content_preview": str(message.get("content") or "")[:500],
                "tool_call_count": len(message.get("tool_calls") or []),
            }
            entries.append(entry)
        return entries

    def write_reasoning_trace(session_key: str, messages: list, usage: dict) -> None:
        trace_dir_raw = os.environ.get("BBO_NANOBOT_REASONING_DIR")
        metadata_path_raw = os.environ.get("BBO_NANOBOT_REASONING_METADATA_PATH")
        if not trace_dir_raw and not metadata_path_raw:
            return
        call_id = os.environ.get("BBO_AGENT_CALL_ID") or session_key.replace(":", "_")
        entries = reasoning_entries(messages)
        combined = "\n\n".join(entry["reasoning_content"] for entry in entries if entry.get("reasoning_content"))
        visible = bool(combined.strip() or any(entry.get("thinking_blocks") for entry in entries))
        trace_path: Path | None = None
        payload = {
            "stage": "agent_reasoning",
            "timestamp": iso_now(),
            "sessionKey": session_key,
            "call_id": call_id,
            "model_requested": os.environ.get("BBO_AGENT_MODEL_REQUESTED") or None,
            "provider": os.environ.get("BBO_AGENT_PROVIDER") or None,
            "reasoning_visible": visible,
            "reasoning_content": combined,
            "entries": entries,
            "usage": usage,
            "reasoning_tokens": reasoning_tokens(usage),
        }
        if trace_dir_raw:
            try:
                trace_dir = Path(trace_dir_raw)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"{call_id}_{filename_ts()}_reasoning.json"
                trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                trace_path = None
        if metadata_path_raw:
            try:
                metadata_path = Path(metadata_path_raw)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "timestamp": payload["timestamp"],
                    "sessionKey": session_key,
                    "call_id": call_id,
                    "model_requested": payload["model_requested"],
                    "provider": payload["provider"],
                    "reasoning_visible": visible,
                    "reasoning_chars": len(combined),
                    "reasoning_entry_count": len(entries),
                    "reasoning_tokens": payload["reasoning_tokens"],
                    "trace_path": None if trace_path is None else str(trace_path),
                }
                with metadata_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception:
                pass

    def write_agent_end(session_key: str, messages: list, duration_s: float, usage: dict, success: bool) -> None:
        session_dir = log_root / session_key
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / f"{filename_ts()}_agent-end.json").write_text(
                json.dumps(
                    {
                        "stage": "agent_end",
                        "timestamp": iso_now(),
                        "sessionKey": session_key,
                        "success": success,
                        "durationMs": round(duration_s * 1000, 1),
                        "messageCount": len(messages),
                        "messages": messages,
                        "usage": usage,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_reasoning_trace(session_key, messages, usage)
        except Exception:
            pass

    try:
        from nanobot.agent import loop as loop_module  # type: ignore

        original_process = loop_module.AgentLoop._process_message
        original_run = loop_module.AgentLoop._run_agent_loop

        async def patched_process(self, msg, session_key=None, **kwargs):
            key = session_key or msg.session_key
            token = session_key_var.set(key)
            try:
                return await original_process(self, msg, session_key=session_key, **kwargs)
            finally:
                session_key_var.reset(token)

        async def patched_run_agent_loop(self, initial_messages, **kwargs):
            start = time.monotonic()
            result = await original_run(self, initial_messages, **kwargs)
            final_content = result[0]
            messages = result[2]
            session_key = session_key_var.get()
            if session_key:
                write_agent_end(
                    session_key=session_key,
                    messages=messages,
                    duration_s=time.monotonic() - start,
                    usage=dict(getattr(self, "_last_usage", {})),
                    success=final_content is not None,
                )
            return result

        loop_module.AgentLoop._process_message = patched_process
        loop_module.AgentLoop._run_agent_loop = patched_run_agent_loop
    except Exception:
        pass


if os.environ.get("BBO_NANOBOT_NO_MAX_TOKENS") == "1":
    _patch_strip_max_tokens()

if os.environ.get("BBO_NANOBOT_LOG_DIR"):
    _patch_log_llm()


from nanobot.cli.commands import app  # noqa: E402  # type: ignore

app()
