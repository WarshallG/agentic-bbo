"""Append-only BBO agent memory store."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BBOMemoryStore:
    """Append-only JSONL memory for agent hypotheses and task priors."""

    path: Path
    summary_path: Path | None = None

    def append(
        self,
        *,
        kind: str,
        content: str,
        tags: list[str] | tuple[str, ...] = (),
        source_call_id: str | None = None,
        trial_range: list[int] | tuple[int, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not kind.strip():
            raise ValueError("Memory `kind` must be non-empty.")
        if not content.strip():
            raise ValueError("Memory `content` must be non-empty.")
        record = {
            "timestamp": time.time(),
            "kind": kind.strip(),
            "content": content.strip(),
            "tags": [str(tag) for tag in tags],
            "source_call_id": source_call_id,
            "trial_range": list(trial_range) if trial_range is not None else None,
            "metadata": dict(metadata or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.write_summary()
        return record

    def read(
        self,
        *,
        kind: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        requested_tags = {str(tag) for tag in tags or ()}
        records: list[dict[str, Any]] = []
        for record in self._iter_records():
            if kind is not None and record.get("kind") != kind:
                continue
            if requested_tags and not requested_tags <= set(record.get("tags", [])):
                continue
            records.append(record)
        if limit <= 0:
            return records
        return records[-limit:]

    def write_summary(self) -> dict[str, Any]:
        records = self.read(limit=0)
        by_kind: dict[str, int] = {}
        for record in records:
            kind = str(record.get("kind", "unknown"))
            by_kind[kind] = by_kind.get(kind, 0) + 1
        summary = {
            "path": str(self.path),
            "record_count": len(records),
            "by_kind": by_kind,
            "latest": records[-5:],
        }
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    def _iter_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


__all__ = ["BBOMemoryStore"]
