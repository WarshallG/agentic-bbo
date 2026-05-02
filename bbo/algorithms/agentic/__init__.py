"""Agentic algorithm implementations and compatibility re-exports."""

from ..llm_based import (
    HeuristicLlamboBackend,
    LlamboAlgorithm,
    LlamboBackend,
    OpenAICompatibleLlamboBackend,
)
from .general_agent import (
    ClaudeCodeBBOAlgorithm,
    GeneralAgentBBOAlgorithm,
    GeneralAgentConfig,
    GeneralAgentValidationError,
    NanobotBBOAlgorithm,
    OpenAICompatibleBBOAlgorithm,
    parse_agent_candidate_payload,
    search_space_schema,
)
from .general_agent_engines import (
    AgentResult,
    AgentWorkCopy,
    ClaudeCodeEngine,
    GeneralAgentEngine,
    MockAgentEngine,
    NanobotEngine,
    OpenAICompatibleToolEngine,
)
from .llm_client import PabloProviderConfig, create_llm_client
from .model_routing import PabloModelRoutingConfig, build_routing_table, resolve_role_model
from ..llm_based import HeuristicOproBackend, OpenAICompatibleOproBackend, OproAlgorithm, OproBackend
from .pablo import PabloAlgorithm
from .prompts import build_explorer_prompt, build_planner_prompt, build_worker_prompt
from .task_registry import TaskCard, TaskRegistry
from .validation import PabloValidationError

__all__ = [
    "AgentResult",
    "AgentWorkCopy",
    "ClaudeCodeBBOAlgorithm",
    "ClaudeCodeEngine",
    "GeneralAgentBBOAlgorithm",
    "GeneralAgentConfig",
    "GeneralAgentEngine",
    "GeneralAgentValidationError",
    "HeuristicLlamboBackend",
    "HeuristicOproBackend",
    "LlamboAlgorithm",
    "LlamboBackend",
    "MockAgentEngine",
    "NanobotBBOAlgorithm",
    "NanobotEngine",
    "OpenAICompatibleLlamboBackend",
    "OpenAICompatibleBBOAlgorithm",
    "OpenAICompatibleOproBackend",
    "OpenAICompatibleToolEngine",
    "OproAlgorithm",
    "OproBackend",
    "PabloAlgorithm",
    "PabloModelRoutingConfig",
    "PabloProviderConfig",
    "PabloValidationError",
    "TaskCard",
    "TaskRegistry",
    "build_explorer_prompt",
    "build_planner_prompt",
    "build_routing_table",
    "build_worker_prompt",
    "create_llm_client",
    "parse_agent_candidate_payload",
    "resolve_role_model",
    "search_space_schema",
]
