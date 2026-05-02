"""Runtime context passed to BBO tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ....core import BBOBenchmarkManifest, Incumbent, TaskDescriptionBundle, TaskSpec, TrialObservation
from .memory import BBOMemoryStore


@dataclass
class BBOToolContext:
    """Mutable per-run state visible to BBO tools."""

    task_spec: TaskSpec
    description: TaskDescriptionBundle
    manifest: BBOBenchmarkManifest
    workspace_dir: Path
    state_dir: Path
    history: list[TrialObservation]
    incumbent: Incumbent | None
    memory_store: BBOMemoryStore | None = None
    seed: int = 0


__all__ = ["BBOToolContext"]
