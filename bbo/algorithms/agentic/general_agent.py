"""General coding-agent optimizer for black-box optimization tasks."""

from __future__ import annotations

import asyncio
import json
import os
import random
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...core import (
    CategoricalParam,
    FloatParam,
    Incumbent,
    IntParam,
    ObjectiveDirection,
    SearchSpace,
    load_BBO_manifest,
    TaskDescriptionBundle,
    TaskSpec,
    TrialObservation,
    TrialSuggestion,
)
from ...core.algo import Algorithm
from .general_agent_engines import (
    AgentResult,
    AgentWorkCopy,
    GeneralAgentEngine,
    create_general_agent_engine,
    normalize_agent_framework,
)
from .serialization import append_jsonl, dump_json, stable_config_identity, to_jsonable
from .tools import (
    BBOMemoryStore,
    BBOToolCallLogger,
    BBOToolContext,
    BBOToolRegistry,
    BBOWebSourceLogger,
    CodeInterpreterTool,
    DisabledBBOCodeBackend,
    FetchURLTool,
    MockBBOCodeBackend,
    SandboxFusionBBOCodeBackend,
    WebSearchTool,
    create_BBO_web_search_provider,
    create_core_BBO_tools,
)
from .validation import PabloValidationError, parse_json_object


DEFAULT_AGENT_TIMEOUT_SECONDS = 180.0
DEFAULT_AGENT_HISTORY_LIMIT = 40
DEFAULT_AGENT_CANDIDATES_PER_CALL = 4


class GeneralAgentValidationError(ValueError):
    """Raised when a general agent returns an invalid candidate payload."""


@dataclass(frozen=True)
class ParsedAgentCandidate:
    """One parsed candidate from a raw agent response."""

    config: dict[str, Any]
    candidate_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCandidateEntry:
    """Queued candidate ready to be surfaced through ask()."""

    config: dict[str, Any]
    call_id: str
    candidate_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneralAgentConfig:
    """Configuration for the general-agent optimizer."""

    framework: str
    algorithm_name: str
    timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS
    max_retries: int = 1
    history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT
    candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL
    model: str | None = None
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    initial_random: int = 0
    run_dir: Path | None = None
    resume: bool = False
    tool_mode: str = "function_calling"
    max_tool_calls: int = 16
    enable_memory: bool = True
    enable_code_interpreter: bool = True
    code_backend: str = "sandboxfusion"
    sandbox_fusion_base_url: str | None = None
    web_search_provider: str = "disabled"
    web_search_api_key_env: str | None = None


class GeneralAgentBBOAlgorithm(Algorithm):
    """Ask/tell wrapper that lets an external general agent propose configs."""

    def __init__(
        self,
        *,
        framework: str,
        algorithm_name: str | None = None,
        engine: GeneralAgentEngine | None = None,
        timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS,
        max_retries: int = 1,
        history_limit: int = DEFAULT_AGENT_HISTORY_LIMIT,
        candidates_per_call: int = DEFAULT_AGENT_CANDIDATES_PER_CALL,
        model: str | None = None,
        provider: str | None = None,
        api_base: str | None = None,
        api_key_env: str | None = None,
        initial_random: int = 0,
        run_dir: Path | str | None = None,
        resume: bool = False,
        tool_mode: str = "function_calling",
        max_tool_calls: int = 16,
        enable_memory: bool = True,
        enable_code_interpreter: bool = True,
        code_backend: str = "sandboxfusion",
        sandbox_fusion_base_url: str | None = None,
        web_search_provider: str = "disabled",
        web_search_api_key_env: str | None = None,
    ) -> None:
        normalized = normalize_agent_framework(framework)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if history_limit < 0:
            raise ValueError("history_limit must be non-negative.")
        if candidates_per_call <= 0:
            raise ValueError("candidates_per_call must be positive.")
        if initial_random < 0:
            raise ValueError("initial_random must be non-negative.")
        normalized_tool_mode = tool_mode.strip().lower().replace("-", "_")
        if normalized_tool_mode not in {"function_calling", "workspace_json"}:
            raise ValueError("tool_mode must be `function_calling` or `workspace_json`.")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative.")

        self.config = GeneralAgentConfig(
            framework=normalized,
            algorithm_name=algorithm_name or f"agentic_{normalized}",
            timeout_seconds=float(timeout_seconds),
            max_retries=int(max_retries),
            history_limit=int(history_limit),
            candidates_per_call=int(candidates_per_call),
            model=model,
            provider=provider,
            api_base=api_base,
            api_key_env=api_key_env,
            initial_random=int(initial_random),
            run_dir=None if run_dir is None else Path(run_dir),
            resume=bool(resume),
            tool_mode=normalized_tool_mode,
            max_tool_calls=int(max_tool_calls),
            enable_memory=bool(enable_memory),
            enable_code_interpreter=bool(enable_code_interpreter),
            code_backend=code_backend,
            sandbox_fusion_base_url=sandbox_fusion_base_url,
            web_search_provider=web_search_provider,
            web_search_api_key_env=web_search_api_key_env,
        )
        self._engine = engine or create_general_agent_engine(normalized)
        self._task_spec: TaskSpec | None = None
        self._description = TaskDescriptionBundle.empty(task_id="unknown")
        self._search_space: SearchSpace | None = None
        self._primary_name: str | None = None
        self._primary_direction = ObjectiveDirection.MINIMIZE
        self._seed = 0
        self._rng = random.Random(0)
        self._history: list[TrialObservation] = []
        self._queue: list[AgentCandidateEntry] = []
        self._seen_config_ids: set[str] = set()
        self._best: Incumbent | None = None
        self._call_index = 0
        self._run_dir: Path | None = None
        self._workspace_dir: Path | None = None
        self._state_dir: Path | None = None
        self._memory_dir: Path | None = None
        self._work_copy: AgentWorkCopy | None = None
        self._manifest = None
        self._memory_store: BBOMemoryStore | None = None
        self._tool_registry: BBOToolRegistry | None = None
        self._artifacts: dict[str, str] = {}
        self._loaded_resume_snapshot: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self.config.algorithm_name

    @property
    def artifact_paths(self) -> dict[str, str]:
        return dict(self._artifacts)

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        self._task_spec = task_spec
        self._search_space = task_spec.search_space
        self._primary_name = task_spec.primary_objective.name
        self._primary_direction = task_spec.primary_objective.direction
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        description = kwargs.get("task_description")
        self._description = (
            description if isinstance(description, TaskDescriptionBundle) else TaskDescriptionBundle.empty(task_id=task_spec.name)
        )

        self._run_dir = Path(kwargs.get("run_dir") or self.config.run_dir or Path.cwd()).resolve()
        self._workspace_dir = self._run_dir / "agent_workspace"
        self._state_dir = self._run_dir / "agent_state"
        self._memory_dir = self._run_dir / "agent_memory"
        log_dir = self._run_dir / "agent_llm_logs"
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = load_BBO_manifest(task_spec)
        self._memory_store = (
            BBOMemoryStore(self._agent_memory_path, self._agent_memory_summary_path)
            if self.config.enable_memory
            else None
        )
        self._tool_registry = self._build_tool_registry()

        config_path = self._build_framework_config(log_dir)
        self._work_copy = AgentWorkCopy(
            state_dir=self._state_dir,
            config_path=config_path,
            project_root=self._workspace_dir,
            workspace_root=self._workspace_dir,
            extra={
                "nanobot_config": {"env": self._agent_env()},
                "claude_config": self._claude_config(),
                "openai_compatible_config": self._openai_compatible_config(),
                "log_dir": log_dir,
            },
        )
        self._artifacts = {
            "agent_workspace": str(self._workspace_dir),
            "agent_state_dir": str(self._state_dir),
            "agent_calls_jsonl": str(self._agent_calls_path),
            "agent_prompts_jsonl": str(self._agent_prompts_path),
            "agent_state_json": str(self._agent_state_path),
            "agent_history_jsonl": str(self._workspace_dir / "history.jsonl"),
            "agent_space_json": str(self._workspace_dir / "space.json"),
            "agent_task_md": str(self._workspace_dir / "task.md"),
            "agent_manifest_json": str(self._workspace_dir / "manifest.json"),
            "agent_tool_specs_json": str(self._agent_tool_specs_path),
            "agent_tool_calls_jsonl": str(self._agent_tool_calls_path),
            "agent_sources_jsonl": str(self._agent_sources_path),
            "agent_memory_jsonl": str(self._agent_memory_path),
            "agent_memory_summary_json": str(self._agent_memory_summary_path),
        }
        for path in (self._agent_calls_path, self._agent_prompts_path, self._agent_tool_calls_path, self._agent_sources_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        if self._tool_registry is not None:
            dump_json(self._agent_tool_specs_path, {"tools": self._tool_registry.get_tool_specs()})

        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        self._call_index = 0
        self._loaded_resume_snapshot = self._load_resume_snapshot()
        self._write_workspace_context()
        self._persist_state()

    def ask(self) -> TrialSuggestion:
        self._require_ready()
        if len(self._history) < self.config.initial_random:
            return self._initial_random_suggestion()
        if not self._queue:
            self._fill_queue_from_agent()
        if not self._queue:
            raise RuntimeError(f"{self.name} could not produce any valid candidate configurations.")
        entry = self._queue.pop(0)
        metadata = {
            "agent_framework": self.config.framework,
            "agent_engine": self._engine.name,
            "agent_call_id": entry.call_id,
            "agent_candidate_index": entry.candidate_index,
            "agent_model": self.config.model,
            "agent_provider": self.config.provider,
            **entry.metadata,
        }
        self._persist_state()
        return TrialSuggestion(config=dict(entry.config), metadata=metadata)

    def tell(self, observation: TrialObservation) -> None:
        self._ingest_observation(observation)
        self._write_workspace_context()
        self._persist_state()

    def replay(self, history: list[TrialObservation]) -> None:
        self._require_ready()
        self._history = []
        self._queue = []
        self._seen_config_ids = set()
        self._best = None
        for observation in history:
            self._ingest_observation(observation, replay=True)
        self._restore_queue_from_snapshot()
        self._write_workspace_context()
        self._persist_state()

    def incumbents(self) -> list[Incumbent]:
        return [self._best] if self._best is not None else []

    def _initial_random_suggestion(self) -> TrialSuggestion:
        search_space = self._require_search_space()
        for _ in range(100):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity not in self._seen_config_ids:
                self._seen_config_ids.add(identity)
                self._persist_state()
                return TrialSuggestion(
                    config=config,
                    metadata={
                        "agent_framework": self.config.framework,
                        "agent_source": "initial_random",
                    },
                )
        config = search_space.sample(self._rng)
        self._seen_config_ids.add(stable_config_identity(config))
        self._persist_state()
        return TrialSuggestion(config=config, metadata={"agent_framework": self.config.framework, "agent_source": "initial_random"})

    def _fill_queue_from_agent(self) -> None:
        search_space = self._require_search_space()
        last_error: str | None = None
        for attempt_index in range(self.config.max_retries + 1):
            self._write_workspace_context()
            call_id = f"agent_call_{self._call_index:05d}"
            self._call_index += 1
            prompt = self._build_agent_prompt(call_id=call_id, attempt_index=attempt_index)
            append_jsonl(
                self._agent_prompts_path,
                {
                    "call_id": call_id,
                    "attempt_index": attempt_index,
                    "prompt": prompt,
                    "timestamp": time.time(),
                },
            )
            result = self._run_engine(prompt, call_id=call_id)
            call_record = {
                "call_id": call_id,
                "attempt_index": attempt_index,
                "framework": self.config.framework,
                "engine": self._engine.name,
                "status": result.status,
                "returncode": result.returncode,
                "error": result.error,
                "answer": result.answer,
                "llm_log": result.llm_log,
                "timestamp": time.time(),
            }
            if result.status != "success":
                append_jsonl(self._agent_calls_path, call_record)
                last_error = result.error or result.answer or result.status
                continue
            try:
                parsed = parse_agent_candidate_payload(result.answer, search_space)
            except GeneralAgentValidationError as exc:
                call_record["validation_error"] = str(exc)
                append_jsonl(self._agent_calls_path, call_record)
                last_error = str(exc)
                continue

            accepted = self._enqueue_candidates(call_id, parsed)
            call_record["accepted_candidates"] = accepted
            append_jsonl(self._agent_calls_path, call_record)
            self._persist_state()
            if accepted > 0:
                return
            last_error = "Agent returned only duplicate candidate configurations."

        fallback = self._fallback_candidate(last_error or "agent_failed")
        if fallback is not None:
            self._queue.append(fallback)
            self._persist_state()
            append_jsonl(
                self._agent_calls_path,
                {
                    "call_id": fallback.call_id,
                    "framework": self.config.framework,
                    "engine": self._engine.name,
                    "status": "fallback",
                    "reason": last_error,
                    "accepted_candidates": 1,
                    "timestamp": time.time(),
                },
            )
            return
        raise RuntimeError(f"{self.name} failed to produce a valid candidate after retries: {last_error}")

    def _run_engine(self, prompt: str, *, call_id: str) -> AgentResult:
        self._require_ready()
        assert self._work_copy is not None
        tools = None
        tool_executor = None
        if self.config.tool_mode == "function_calling":
            registry = self._require_tool_registry()
            context = self._build_tool_context()
            tools = registry.get_tool_specs()

            async def _execute_tool(tool_name: str, arguments: dict[str, Any], tool_call_id: str | None = None) -> str:
                return await registry.execute_tool(
                    tool_name,
                    arguments,
                    context,
                    call_id=call_id,
                    tool_call_id=tool_call_id,
                )

            tool_executor = _execute_tool
        coro = self._engine.run_agent(
            "",
            prompt,
            self._work_copy,
            agent_id="bbo",
            timeout=self.config.timeout_seconds,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_calls=self.config.max_tool_calls,
        )
        return _run_coro_sync(coro)

    def _build_tool_registry(self) -> BBOToolRegistry:
        tools = create_core_BBO_tools(enable_memory=self.config.enable_memory)
        if self.config.enable_code_interpreter:
            tools.append(CodeInterpreterTool())
        tools.extend([WebSearchTool(), FetchURLTool()])
        return BBOToolRegistry(tools, logger=BBOToolCallLogger(self._agent_tool_calls_path))

    def _build_tool_context(self) -> BBOToolContext:
        self._require_ready()
        assert self._workspace_dir is not None
        assert self._state_dir is not None
        assert self._manifest is not None
        return BBOToolContext(
            task_spec=self._require_task_spec(),
            description=self._description,
            manifest=self._manifest,
            workspace_dir=self._workspace_dir,
            state_dir=self._state_dir,
            history=self._history,
            incumbent=self._best,
            memory_store=self._memory_store,
            code_backend=self._build_code_backend(),
            web_search_provider=self._build_web_search_provider(),
            source_logger=BBOWebSourceLogger(self._agent_sources_path),
            seed=self._seed,
        )

    def _build_code_backend(self) -> object:
        backend = self.config.code_backend.strip().lower().replace("-", "_")
        if backend == "mock":
            return MockBBOCodeBackend()
        if backend in {"disabled", "local_disabled", "none"}:
            return DisabledBBOCodeBackend()
        if backend == "sandboxfusion":
            base_url = self.config.sandbox_fusion_base_url or os.environ.get("SANDBOX_FUSION_BASE_URL")
            if not base_url:
                return DisabledBBOCodeBackend()
            return SandboxFusionBBOCodeBackend(base_url=base_url)
        raise ValueError(f"Unknown BBO code backend `{self.config.code_backend}`.")

    def _build_web_search_provider(self) -> object:
        return create_BBO_web_search_provider(
            self.config.web_search_provider,
            api_key_env=self.config.web_search_api_key_env,
        )

    def _require_tool_registry(self) -> BBOToolRegistry:
        if self._tool_registry is None:
            raise RuntimeError("BBO tool registry is not initialized.")
        return self._tool_registry

    def _enqueue_candidates(self, call_id: str, candidates: list[ParsedAgentCandidate]) -> int:
        accepted = 0
        for candidate in candidates:
            identity = stable_config_identity(candidate.config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            self._queue.append(
                AgentCandidateEntry(
                    config=dict(candidate.config),
                    call_id=call_id,
                    candidate_index=candidate.candidate_index,
                    metadata=dict(candidate.metadata),
                )
            )
            accepted += 1
        return accepted

    def _fallback_candidate(self, reason: str) -> AgentCandidateEntry | None:
        search_space = self._require_search_space()
        for index in range(500):
            config = search_space.sample(self._rng)
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            self._seen_config_ids.add(identity)
            return AgentCandidateEntry(
                config=config,
                call_id=f"fallback_random_{self._call_index:05d}",
                candidate_index=index,
                metadata={
                    "agent_source": "fallback_random",
                    "agent_fallback_reason": reason,
                },
            )
        return None

    def _ingest_observation(self, observation: TrialObservation, *, replay: bool = False) -> None:
        del replay
        assert self._primary_name is not None
        self._history.append(observation)
        self._seen_config_ids.add(stable_config_identity(observation.suggestion.config))
        if observation.success and self._primary_name in observation.objectives:
            score = float(observation.objectives[self._primary_name])
            incumbent = Incumbent(
                config=dict(observation.suggestion.config),
                score=score,
                objectives=dict(observation.objectives),
                trial_id=observation.suggestion.trial_id,
                metadata={"algorithm": self.name, "agent_framework": self.config.framework},
            )
            if self._best is None:
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MINIMIZE and score < float(self._best.score):
                self._best = incumbent
            elif self._primary_direction == ObjectiveDirection.MAXIMIZE and score > float(self._best.score):
                self._best = incumbent

    def _write_workspace_context(self) -> None:
        self._require_ready()
        assert self._workspace_dir is not None
        task_spec = self._require_task_spec()
        history = self._history[-self.config.history_limit :] if self.config.history_limit else []
        (self._workspace_dir / "task.md").write_text(self._render_task_markdown(), encoding="utf-8")
        dump_json(self._workspace_dir / "space.json", {"parameters": search_space_schema(task_spec.search_space)})
        if self._manifest is not None:
            dump_json(self._workspace_dir / "manifest.json", self._manifest.to_dict())
        if self._tool_registry is not None:
            dump_json(self._workspace_dir / "tool_specs.json", {"tools": self._tool_registry.get_tool_specs()})
        dump_json(
            self._workspace_dir / "objective.json",
            {
                "name": task_spec.primary_objective.name,
                "direction": task_spec.primary_objective.direction.value,
                "all_objectives": [
                    {"name": objective.name, "direction": objective.direction.value} for objective in task_spec.objectives
                ],
            },
        )
        dump_json(
            self._workspace_dir / "incumbent.json",
            {
                "config": None if self._best is None else self._best.config,
                "score": None if self._best is None else self._best.score,
                "objectives": {} if self._best is None else self._best.objectives,
                "trial_id": None if self._best is None else self._best.trial_id,
            },
        )
        self._write_history_jsonl(history)
        (self._workspace_dir / "instructions.md").write_text(self._render_instructions(), encoding="utf-8")

    def _write_history_jsonl(self, history: list[TrialObservation]) -> None:
        assert self._workspace_dir is not None
        path = self._workspace_dir / "history.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for observation in history:
                handle.write(json.dumps(to_jsonable(_observation_summary(observation)), sort_keys=True) + "\n")

    def _render_task_markdown(self) -> str:
        task_spec = self._require_task_spec()
        if self._description.rendered_context:
            return self._description.rendered_context
        return f"# {task_spec.name}\n\nNo structured task description was available."

    def _render_instructions(self) -> str:
        task_spec = self._require_task_spec()
        return textwrap.dedent(
            f"""
            # Agentic BBO Candidate Protocol

            You are proposing configurations for a black-box optimization benchmark.
            Do not evaluate the objective yourself and do not modify benchmark result files.

            Files in this workspace:
            - task.md: task background, goal, constraints, and prior knowledge.
            - manifest.json: agent benchmark construction, tool policy, and provenance.
            - space.json: exact parameter schema. Every candidate must include every parameter exactly once.
            - objective.json: primary objective name and optimization direction.
            - history.jsonl: recent evaluated trials.
            - incumbent.json: current best known configuration, if any.
            - tool_specs.json: available BBO function-calling tools when the backend supports tools.

            Task: {task_spec.name}
            Primary objective: {task_spec.primary_objective.name}
            Direction: {task_spec.primary_objective.direction.value}

            Print only raw JSON to stdout, with this exact shape:
            {{"candidates": [{{"config": {{"param_name": "value"}}, "rationale": "short reason"}}]}}

            Requirements:
            - Return {self.config.candidates_per_call} candidate configurations when possible.
            - Use BBO tools for task context, history, memory, validation, code analysis, or web research when available.
            - Do not include markdown fences, comments, prose, or partial configurations.
            - Float and integer values must stay within their declared bounds.
            - Categorical values must be one of the declared choices.
            """
        ).strip()

    def _build_agent_prompt(self, *, call_id: str, attempt_index: int) -> str:
        task_spec = self._require_task_spec()
        best_score = None if self._best is None else self._best.score
        return textwrap.dedent(
            f"""
            You are an optimization agent for task `{task_spec.name}`.
            Workspace path: {self._workspace_dir}
            Call id: {call_id}
            Attempt: {attempt_index}

            Read `instructions.md`, `task.md`, `manifest.json`, `space.json`,
            `objective.json`, `history.jsonl`, `incumbent.json`, and `tool_specs.json`
            in the workspace.

            Current best primary objective: {best_score}
            Objective direction: {task_spec.primary_objective.direction.value}

            Produce candidate configurations now. Your entire stdout must be the
            strict JSON object described in `instructions.md`.
            """
        ).strip()

    def _build_framework_config(self, log_dir: Path) -> Path | None:
        assert self._state_dir is not None
        if self.config.framework == "nanobot":
            config_path = self._state_dir / "config.json"
            provider = self.config.provider
            provider_key = _nanobot_provider_key(provider or "custom")
            model = self.config.model
            cfg: dict[str, Any] = {
                "agents": {
                    "defaults": {
                        "workspace": str(self._workspace_dir),
                        "provider": provider_key if provider else "auto",
                    }
                },
                "providers": {},
                "channels": {
                    "send_progress": False,
                    "send_tool_hints": False,
                },
                "tools": {
                    "restrict_to_workspace": True,
                },
            }
            if model:
                cfg["agents"]["defaults"]["model"] = model
            api_key = self._api_key()
            if provider or self.config.api_base or api_key:
                entry: dict[str, str] = {}
                if api_key:
                    entry["api_key"] = api_key
                if self.config.api_base:
                    entry["api_base"] = self.config.api_base
                cfg["providers"][provider_key] = entry
            config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
            return config_path
        if self.config.framework == "claude_code":
            settings_path = self._state_dir / "settings.json"
            if not settings_path.exists():
                settings_path.write_text("{}", encoding="utf-8")
        del log_dir
        return None

    def _agent_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        api_key = self._api_key()
        provider = (self.config.provider or "").lower()
        if api_key:
            if provider == "openai":
                env["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                env["ANTHROPIC_API_KEY"] = api_key
            elif provider == "google":
                env["GOOGLE_API_KEY"] = api_key
            elif self.config.api_key_env:
                env[self.config.api_key_env] = api_key
        if self.config.api_base:
            if provider == "openai":
                env["OPENAI_BASE_URL"] = self.config.api_base
            elif provider == "anthropic":
                env["ANTHROPIC_BASE_URL"] = self.config.api_base
        return env

    def _openai_compatible_config(self) -> dict[str, Any]:
        return {
            "api_key": self._api_key(),
            "api_base": self.config.api_base,
            "model": self.config.model,
        }

    def _claude_config(self) -> dict[str, Any]:
        provider = (self.config.provider or "").lower()
        api_key = self._api_key()
        env: dict[str, str] = {}
        if provider == "claude":
            if api_key:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = api_key
        elif provider == "anthropic" or not provider:
            if self.config.api_base:
                env["ANTHROPIC_BASE_URL"] = self.config.api_base
            if api_key:
                env["ANTHROPIC_API_KEY"] = api_key
        else:
            if self.config.api_base:
                env["ANTHROPIC_BASE_URL"] = self.config.api_base
            if api_key:
                env["ANTHROPIC_AUTH_TOKEN"] = api_key
        return {
            "env": env,
            "model": self.config.model,
        }

    def _api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        return os.environ.get(self.config.api_key_env)

    def _restore_queue_from_snapshot(self) -> None:
        if not self.config.resume or not self._loaded_resume_snapshot:
            return
        search_space = self._require_search_space()
        restored: list[AgentCandidateEntry] = []
        for item in self._loaded_resume_snapshot.get("queue", []):
            if not isinstance(item, Mapping):
                continue
            try:
                config = search_space.coerce_config(dict(item.get("config", {})), use_defaults=False)
            except Exception:
                continue
            identity = stable_config_identity(config)
            if identity in self._seen_config_ids:
                continue
            restored.append(
                AgentCandidateEntry(
                    config=config,
                    call_id=str(item.get("call_id", "restored")),
                    candidate_index=int(item.get("candidate_index", 0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
            self._seen_config_ids.add(identity)
        self._queue = restored
        self._call_index = max(self._call_index, int(self._loaded_resume_snapshot.get("call_index", 0)))

    def _load_resume_snapshot(self) -> dict[str, Any]:
        if not self.config.resume or not self._agent_state_path.exists():
            return {}
        try:
            data = json.loads(self._agent_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _persist_state(self) -> None:
        if self._run_dir is None:
            return
        dump_json(
            self._agent_state_path,
            {
                "algorithm": self.name,
                "framework": self.config.framework,
                "engine": self._engine.name,
                "call_index": self._call_index,
                "history_size": len(self._history),
                "queue": [to_jsonable(entry) for entry in self._queue],
                "seen_config_ids": sorted(self._seen_config_ids),
                "best_config": None if self._best is None else self._best.config,
                "best_score": None if self._best is None else self._best.score,
                "model": self.config.model,
                "provider": self.config.provider,
                "tool_mode": self.config.tool_mode,
                "max_tool_calls": self.config.max_tool_calls,
                "enable_memory": self.config.enable_memory,
                "web_search_provider": self.config.web_search_provider,
                "code_backend": self.config.code_backend,
            },
        )

    @property
    def _agent_calls_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_calls.jsonl"

    @property
    def _agent_prompts_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_prompts.jsonl"

    @property
    def _agent_state_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_state.json"

    @property
    def _agent_tool_calls_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_tool_calls.jsonl"

    @property
    def _agent_tool_specs_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_tool_specs.json"

    @property
    def _agent_sources_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir / "agent_web_sources.jsonl"

    @property
    def _agent_memory_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory.jsonl"

    @property
    def _agent_memory_summary_path(self) -> Path:
        assert self._memory_dir is not None
        return self._memory_dir / "memory_summary.json"

    def _require_ready(self) -> None:
        if self._task_spec is None or self._search_space is None:
            raise RuntimeError(f"{self.__class__.__name__}.setup() must be called before use.")

    def _require_task_spec(self) -> TaskSpec:
        self._require_ready()
        assert self._task_spec is not None
        return self._task_spec

    def _require_search_space(self) -> SearchSpace:
        self._require_ready()
        assert self._search_space is not None
        return self._search_space


class NanobotBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Nanobot."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="nanobot", algorithm_name="agentic_nanobot", **kwargs)


class ClaudeCodeBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by Claude Code."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="claude_code", algorithm_name="agentic_claude_code", **kwargs)


class OpenAICompatibleBBOAlgorithm(GeneralAgentBBOAlgorithm):
    """General-agent optimizer backed by OpenAI-compatible function calling."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(framework="openai_compatible", algorithm_name="agentic_openai_compatible", **kwargs)


def parse_agent_candidate_payload(raw_text: str, search_space: SearchSpace) -> list[ParsedAgentCandidate]:
    try:
        payload = parse_json_object(raw_text)
    except PabloValidationError as exc:
        extracted = _extract_candidates_json_object(raw_text)
        if extracted is None:
            raise GeneralAgentValidationError(str(exc)) from exc
        payload = extracted
    if set(payload) != {"candidates"}:
        raise GeneralAgentValidationError("Agent response must contain exactly one top-level key: `candidates`.")
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise GeneralAgentValidationError("Agent response must provide a non-empty `candidates` list.")

    parsed: list[ParsedAgentCandidate] = []
    seen: set[str] = set()
    errors: list[str] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            errors.append(f"Candidate {index} is not a JSON object.")
            continue
        item_dict = dict(item)
        metadata: dict[str, Any] = {}
        if "config" in item_dict:
            raw_config = item_dict.pop("config")
            metadata = dict(item_dict)
            if not isinstance(raw_config, Mapping):
                errors.append(f"Candidate {index} `config` is not a JSON object.")
                continue
            config_mapping = dict(raw_config)
        else:
            config_mapping = item_dict
        try:
            config = search_space.coerce_config(config_mapping, use_defaults=False)
        except Exception as exc:
            errors.append(f"Candidate {index} is invalid: {exc}")
            continue
        identity = stable_config_identity(config)
        if identity in seen:
            continue
        seen.add(identity)
        parsed.append(ParsedAgentCandidate(config=config, candidate_index=index, metadata=metadata))
    if not parsed:
        detail = "; ".join(errors) if errors else "Agent response did not contain any valid unique configurations."
        raise GeneralAgentValidationError(detail)
    return parsed


def _extract_candidates_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text or text.startswith("```"):
        return None
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "candidates" in payload:
            return payload
    for object_text in _balanced_json_object_texts(text):
        payload = _loads_lenient_json_object(object_text)
        if isinstance(payload, dict) and "candidates" in payload:
            return payload
    return None


def _balanced_json_object_texts(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char != "{":
                continue
            start = index
            depth = 1
            in_string = False
            escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _loads_lenient_json_object(object_text: str) -> dict[str, Any] | None:
    candidates = [object_text, _escape_control_chars_in_strings(object_text)]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    try:
        from json_repair import loads as repair_loads  # type: ignore
    except ImportError:
        return None
    try:
        payload = repair_loads(object_text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _escape_control_chars_in_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
            elif char == "\\":
                result.append(char)
                escaped = True
            elif char == '"':
                result.append(char)
                in_string = False
            elif char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            else:
                result.append(char)
            continue
        result.append(char)
        if char == '"':
            in_string = True
    return "".join(result)


def search_space_schema(search_space: SearchSpace) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for param in search_space:
        if isinstance(param, FloatParam):
            schema.append(
                {
                    "name": param.name,
                    "type": "float",
                    "low": float(param.low),
                    "high": float(param.high),
                    "log": bool(param.log),
                    "default": param.effective_default(),
                }
            )
        elif isinstance(param, IntParam):
            schema.append(
                {
                    "name": param.name,
                    "type": "int",
                    "low": int(param.low),
                    "high": int(param.high),
                    "log": bool(param.log),
                    "default": param.effective_default(),
                }
            )
        elif isinstance(param, CategoricalParam):
            schema.append(
                {
                    "name": param.name,
                    "type": "categorical",
                    "choices": list(param.choices),
                    "default": param.effective_default(),
                }
            )
        else:
            raise TypeError(f"Unsupported parameter type for agent schema: {type(param).__name__}")
    return schema


def _observation_summary(observation: TrialObservation) -> dict[str, Any]:
    return {
        "trial_id": observation.suggestion.trial_id,
        "config": observation.suggestion.config,
        "budget": observation.suggestion.budget,
        "status": observation.status.value,
        "objectives": observation.objectives,
        "metrics": observation.metrics,
        "elapsed_seconds": observation.elapsed_seconds,
        "error_type": observation.error_type,
        "error_message": observation.error_message,
        "timestamp": observation.timestamp,
    }


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["result"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive cross-thread propagation.
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_box:
        raise error_box["error"]
    return result_box["result"]


def _nanobot_provider_key(provider: str) -> str:
    return {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "gemini",
        "ollama": "ollama",
        "azure": "azure_openai",
    }.get(provider, "custom")


__all__ = [
    "ClaudeCodeBBOAlgorithm",
    "GeneralAgentBBOAlgorithm",
    "GeneralAgentConfig",
    "GeneralAgentValidationError",
    "NanobotBBOAlgorithm",
    "OpenAICompatibleBBOAlgorithm",
    "parse_agent_candidate_payload",
    "search_space_schema",
]
