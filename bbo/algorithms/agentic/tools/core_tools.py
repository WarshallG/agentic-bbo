"""Core BBO tools for task context, history, validation, sampling, and memory."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

import numpy as np

from ....core import CategoricalParam, FloatParam, IntParam, ObjectiveDirection, SearchSpace, TrialObservation
from ..serialization import stable_config_identity
from .base import BaseBBOTool
from .context import BBOToolContext


class GetTaskContextTool(BaseBBOTool):
    name = "get_task_context"
    description = "Return task documentation, manifest, objective metadata, and benchmark constraints."
    parameters_schema = {
        "type": "object",
        "properties": {
            "sections": {"type": "array", "items": {"type": "string"}},
            "max_chars_per_section": {"type": "integer", "minimum": 200, "default": 4000},
            "include_manifest": {"type": "boolean", "default": True},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        sections: list[str] | None = None,
        max_chars_per_section: int = 4000,
        include_manifest: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        wanted = {section for section in sections or context.description.section_map}
        docs = {
            name: _truncate(text, int(max_chars_per_section))
            for name, text in context.description.section_map.items()
            if name in wanted
        }
        return {
            "task_id": context.task_spec.name,
            "objectives": [
                {"name": objective.name, "direction": objective.direction.value}
                for objective in context.task_spec.objectives
            ],
            "max_evaluations": context.task_spec.max_evaluations,
            "metadata": context.task_spec.metadata,
            "sections": docs,
            "manifest": context.manifest.to_dict() if include_manifest else None,
        }


class GetSearchSpaceTool(BaseBBOTool):
    name = "get_search_space"
    description = "Return the exact BBO search-space schema, defaults, and parameter ordering."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, context: BBOToolContext, **_: Any) -> dict[str, Any]:
        return {
            "parameters": search_space_schema(context.task_spec.search_space),
            "defaults": context.task_spec.search_space.defaults(),
            "dimension": len(context.task_spec.search_space),
        }


class GetTrialHistoryTool(BaseBBOTool):
    name = "get_trial_history"
    description = "Return evaluated BBO trials without consuming objective budget."
    parameters_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["recent", "best", "all"], "default": "recent"},
            "limit": {"type": "integer", "minimum": 1, "default": 20},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        mode: str = "recent",
        limit: int = 20,
        offset: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        observations = list(context.history)
        if mode == "best":
            observations = _sort_by_primary(observations, context)
        elif mode == "recent":
            observations = observations[::-1]
        elif mode != "all":
            raise ValueError("mode must be one of recent, best, all.")
        limit = max(1, int(limit))
        offset = max(0, int(offset))
        page = observations[offset : offset + limit]
        return {
            "mode": mode,
            "total": len(context.history),
            "offset": offset,
            "limit": limit,
            "trials": [_observation_summary(observation) for observation in page],
        }


class GetIncumbentTool(BaseBBOTool):
    name = "get_incumbent"
    description = "Return the current best known BBO configuration and objectives."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, context: BBOToolContext, **_: Any) -> dict[str, Any]:
        incumbent = context.incumbent
        if incumbent is None:
            return {"incumbent": None}
        return {
            "incumbent": {
                "config": incumbent.config,
                "score": incumbent.score,
                "objectives": incumbent.objectives,
                "trial_id": incumbent.trial_id,
                "metadata": incumbent.metadata,
            }
        }


class ValidateCandidatesTool(BaseBBOTool):
    name = "validate_candidates"
    description = "Validate candidate configurations against the BBO search space and duplicate history."
    parameters_schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Candidate objects, either raw configs or objects with a `config` field.",
            }
        },
        "required": ["candidates"],
    }

    async def execute(self, context: BBOToolContext, candidates: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list.")
        seen_history = {
            stable_config_identity(observation.suggestion.config)
            for observation in context.history
        }
        seen_payload: set[str] = set()
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                invalid.append({"index": index, "error": "candidate is not an object"})
                continue
            raw = item.get("config", item)
            if not isinstance(raw, dict):
                invalid.append({"index": index, "error": "`config` is not an object"})
                continue
            try:
                config = context.task_spec.search_space.coerce_config(raw, use_defaults=False)
            except Exception as exc:
                invalid.append({"index": index, "error": str(exc)})
                continue
            identity = stable_config_identity(config)
            duplicate = identity in seen_history or identity in seen_payload
            seen_payload.add(identity)
            valid.append({"index": index, "config": config, "duplicate": duplicate, "identity": identity})
        return {"valid": valid, "invalid": invalid, "valid_count": len(valid), "invalid_count": len(invalid)}


class SampleCandidatesTool(BaseBBOTool):
    name = "sample_candidates"
    description = "Sample valid BBO candidates without evaluating them."
    parameters_schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 128, "default": 4},
            "seed": {"type": "integer"},
            "strategy": {"type": "string", "enum": ["random", "around_incumbent"], "default": "random"},
            "jitter_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.1},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        n: int = 4,
        seed: int | None = None,
        strategy: str = "random",
        jitter_fraction: float = 0.1,
        **_: Any,
    ) -> dict[str, Any]:
        rng = random.Random(context.seed if seed is None else int(seed))
        target = min(max(1, int(n)), 128)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _attempt in range(target * 100):
            if strategy == "around_incumbent" and context.incumbent is not None:
                config = _sample_around(context.task_spec.search_space, context.incumbent.config, rng, float(jitter_fraction))
            elif strategy == "random":
                config = context.task_spec.search_space.sample(rng)
            else:
                raise ValueError("strategy must be random or around_incumbent.")
            identity = stable_config_identity(config)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append({"config": config, "identity": identity})
            if len(candidates) >= target:
                break
        return {"strategy": strategy, "candidates": candidates, "count": len(candidates)}


class AnalyzeHistoryTool(BaseBBOTool):
    name = "analyze_history"
    description = "Compute lightweight BBO history statistics for agent reasoning."
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 100},
        },
    }

    async def execute(self, context: BBOToolContext, limit: int = 100, **_: Any) -> dict[str, Any]:
        observations = [obs for obs in context.history if obs.success]
        if limit > 0:
            observations = observations[-int(limit) :]
        primary = context.task_spec.primary_objective.name
        scored = [obs for obs in observations if primary in obs.objectives]
        if not scored:
            return {"history_size": len(context.history), "success_count": 0, "primary_objective": primary}
        scores = np.asarray([float(obs.objectives[primary]) for obs in scored], dtype=float)
        direction = context.task_spec.primary_objective.direction
        best_index = int(np.argmin(scores) if direction == ObjectiveDirection.MINIMIZE else np.argmax(scores))
        analysis: dict[str, Any] = {
            "history_size": len(context.history),
            "success_count": len(scored),
            "primary_objective": primary,
            "direction": direction.value,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "best_trial": _observation_summary(scored[best_index]),
            "numeric_correlations": {},
            "categorical_groups": {},
        }
        for param in context.task_spec.search_space:
            values = [obs.suggestion.config.get(param.name) for obs in scored]
            if isinstance(param, (FloatParam, IntParam)):
                xs = np.asarray([float(value) for value in values], dtype=float)
                if len(xs) > 1 and float(np.std(xs)) > 0.0 and float(np.std(scores)) > 0.0:
                    analysis["numeric_correlations"][param.name] = float(np.corrcoef(xs, scores)[0, 1])
            elif isinstance(param, CategoricalParam):
                groups: dict[str, list[float]] = {}
                for value, score in zip(values, scores, strict=True):
                    groups.setdefault(str(value), []).append(float(score))
                analysis["categorical_groups"][param.name] = {
                    key: {"count": len(vals), "mean": float(np.mean(vals))}
                    for key, vals in groups.items()
                }
        return analysis


class MemoryReadTool(BaseBBOTool):
    name = "memory_read"
    description = "Read append-only BBO agent memory records."
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "minimum": 1, "default": 20},
        },
    }

    async def execute(
        self,
        context: BBOToolContext,
        kind: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> dict[str, Any]:
        if context.memory_store is None:
            return {"enabled": False, "records": []}
        records = context.memory_store.read(kind=kind, tags=tags, limit=int(limit))
        return {"enabled": True, "records": records, "count": len(records)}


class MemoryWriteTool(BaseBBOTool):
    name = "memory_write"
    description = "Append hypotheses, priors, failure notes, or strategy notes to BBO memory."
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "source_call_id": {"type": "string"},
            "trial_range": {"type": "array", "items": {"type": "integer"}},
            "metadata": {"type": "object"},
        },
        "required": ["kind", "content"],
    }

    async def execute(
        self,
        context: BBOToolContext,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        source_call_id: str | None = None,
        trial_range: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if context.memory_store is None:
            return {"enabled": False, "written": False}
        record = context.memory_store.append(
            kind=kind,
            content=content,
            tags=tags or (),
            source_call_id=source_call_id,
            trial_range=trial_range,
            metadata=metadata,
        )
        return {"enabled": True, "written": True, "record": record}


def create_core_BBO_tools(*, enable_memory: bool = True) -> list[BaseBBOTool]:
    tools: list[BaseBBOTool] = [
        GetTaskContextTool(),
        GetSearchSpaceTool(),
        GetTrialHistoryTool(),
        GetIncumbentTool(),
        ValidateCandidatesTool(),
        SampleCandidatesTool(),
        AnalyzeHistoryTool(),
    ]
    if enable_memory:
        tools.extend([MemoryReadTool(), MemoryWriteTool()])
    return tools


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
            raise TypeError(f"Unsupported parameter type for BBO tool schema: {type(param).__name__}")
    return schema


def _sort_by_primary(observations: list[TrialObservation], context: BBOToolContext) -> list[TrialObservation]:
    primary = context.task_spec.primary_objective.name
    direction = context.task_spec.primary_objective.direction
    scored = [obs for obs in observations if obs.success and primary in obs.objectives]
    reverse = direction == ObjectiveDirection.MAXIMIZE
    return sorted(scored, key=lambda obs: float(obs.objectives[primary]), reverse=reverse)


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
        "metadata": observation.metadata,
    }


def _sample_around(
    search_space: SearchSpace,
    incumbent: dict[str, Any],
    rng: random.Random,
    jitter_fraction: float,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    fraction = min(max(float(jitter_fraction), 0.0), 1.0)
    for param in search_space:
        current = incumbent.get(param.name, param.effective_default())
        if isinstance(param, FloatParam):
            span = float(param.high) - float(param.low)
            value = float(current) + rng.uniform(-span * fraction, span * fraction)
            config[param.name] = param.coerce(min(max(value, param.low), param.high))
        elif isinstance(param, IntParam):
            span = int(param.high) - int(param.low)
            step = max(1, int(round(span * fraction)))
            value = int(current) + rng.randint(-step, step)
            config[param.name] = param.coerce(min(max(value, param.low), param.high))
        elif isinstance(param, CategoricalParam):
            config[param.name] = current if rng.random() > fraction else param.sample(rng)
        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")
    return search_space.coerce_config(config, use_defaults=False)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


__all__ = [
    "AnalyzeHistoryTool",
    "GetIncumbentTool",
    "GetSearchSpaceTool",
    "GetTaskContextTool",
    "GetTrialHistoryTool",
    "MemoryReadTool",
    "MemoryWriteTool",
    "SampleCandidatesTool",
    "ValidateCandidatesTool",
    "create_core_BBO_tools",
    "search_space_schema",
]
