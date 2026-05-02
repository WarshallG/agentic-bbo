"""Algorithm packages and registry."""

from .agentic import (
    ClaudeCodeBBOAlgorithm,
    GeneralAgentBBOAlgorithm,
    NanobotBBOAlgorithm,
    OpenAICompatibleBBOAlgorithm,
    PabloAlgorithm,
)
from .llm_based import (
    HeuristicLlamboBackend,
    HeuristicOproBackend,
    LlamboAlgorithm,
    LlamboBackend,
    OpenAICompatibleLlamboBackend,
    OpenAICompatibleOproBackend,
    OproAlgorithm,
    OproBackend,
)
from .llm_based.skydiscover_interleaved import SkydiscoverInterleavedAlgorithm
from .model_based import CustomPfnsBoAlgorithm, OptunaTpeAlgorithm, Pfns4BoAlgorithm, TabPfnV2BoAlgorithm
from .registry import ALGORITHM_REGISTRY, AlgorithmSpec, algorithms_by_family, create_algorithm
from .traditional import PyCmaAlgorithm, RandomSearchAlgorithm

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmSpec",
    "ClaudeCodeBBOAlgorithm",
    "CustomPfnsBoAlgorithm",
    "GeneralAgentBBOAlgorithm",
    "HeuristicLlamboBackend",
    "HeuristicOproBackend",
    "LlamboAlgorithm",
    "LlamboBackend",
    "OpenAICompatibleLlamboBackend",
    "OpenAICompatibleBBOAlgorithm",
    "OpenAICompatibleOproBackend",
    "OptunaTpeAlgorithm",
    "OproAlgorithm",
    "OproBackend",
    "NanobotBBOAlgorithm",
    "PabloAlgorithm",
    "Pfns4BoAlgorithm",
    "PyCmaAlgorithm",
    "RandomSearchAlgorithm",
    "SkydiscoverInterleavedAlgorithm",
    "TabPfnV2BoAlgorithm",
    "algorithms_by_family",
    "create_algorithm",
]
