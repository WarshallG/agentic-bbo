"""Base class for BBO function-calling tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .context import BBOToolContext


class BaseBBOTool(ABC):
    """A function-callable tool exposed to a BBO agent."""

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def function_spec(self) -> dict[str, Any]:
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define a non-empty name.")
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @abstractmethod
    async def execute(self, context: BBOToolContext, **kwargs: Any) -> Any:
        """Execute the tool and return a JSON-serializable result."""


__all__ = ["BaseBBOTool"]
