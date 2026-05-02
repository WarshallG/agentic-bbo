"""BBO web research tools with source logging."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serialization import append_jsonl
from .base import BaseBBOTool
from .context import BBOToolContext


@dataclass
class BBOWebSourceLogger:
    """Append-only logger for web search and fetch sources."""

    path: Path

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        source_id = f"src_{int(time.time() * 1000)}_{abs(hash(json.dumps(record, sort_keys=True, default=str))) % 1000000:06d}"
        payload = {"source_id": source_id, "timestamp": time.time(), **record}
        append_jsonl(self.path, payload)
        return payload


class DisabledBBOWebSearchProvider:
    async def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        del query, limit
        raise RuntimeError("BBO web search is disabled for this run.")


@dataclass
class MockBBOWebSearchProvider:
    """Deterministic web provider for tests."""

    results: list[dict[str, Any]] | None = None

    async def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        base = self.results or [
            {
                "title": "Mock BBO prior",
                "url": "https://example.test/bbo-prior",
                "snippet": f"Mock search result for {query}",
            }
        ]
        return base[: max(1, int(limit))]


@dataclass
class TavilyBBOWebSearchProvider:
    api_key: str
    endpoint: str = "https://api.tavily.com/search"
    timeout_seconds: float = 30.0

    async def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        payload = {"api_key": self.api_key, "query": query, "max_results": max(1, int(limit))}
        data = await asyncio.to_thread(_post_json, self.endpoint, payload, self.timeout_seconds)
        items = data.get("results", []) if isinstance(data, dict) else []
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", item.get("snippet", ""))),
            }
            for item in items
            if isinstance(item, dict)
        ]


@dataclass
class SerpApiBBOWebSearchProvider:
    api_key: str
    endpoint: str = "https://serpapi.com/search.json"
    timeout_seconds: float = 30.0

    async def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"engine": "google", "q": query, "api_key": self.api_key, "num": int(limit)})
        data = await asyncio.to_thread(_get_json, f"{self.endpoint}?{params}", self.timeout_seconds)
        items = data.get("organic_results", []) if isinstance(data, dict) else []
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("link", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in items
            if isinstance(item, dict)
        ][: max(1, int(limit))]


@dataclass
class BingBBOWebSearchProvider:
    api_key: str
    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"
    timeout_seconds: float = 30.0

    async def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"q": query, "count": int(limit)})
        data = await asyncio.to_thread(
            _get_json,
            f"{self.endpoint}?{params}",
            self.timeout_seconds,
            {"Ocp-Apim-Subscription-Key": self.api_key},
        )
        items = ((data.get("webPages") or {}).get("value") or []) if isinstance(data, dict) else []
        return [
            {
                "title": str(item.get("name", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in items
            if isinstance(item, dict)
        ]


def create_BBO_web_search_provider(provider: str, *, api_key_env: str | None = None) -> object:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"", "disabled", "none"}:
        return DisabledBBOWebSearchProvider()
    if normalized == "mock":
        return MockBBOWebSearchProvider()
    env_name = api_key_env or {
        "tavily": "TAVILY_API_KEY",
        "serpapi": "SERPAPI_API_KEY",
        "bing": "BING_SEARCH_API_KEY",
    }.get(normalized)
    api_key = os.environ.get(env_name or "")
    if not api_key:
        raise ValueError(f"BBO web search provider `{provider}` requires API key env `{env_name}`.")
    if normalized == "tavily":
        return TavilyBBOWebSearchProvider(api_key=api_key)
    if normalized == "serpapi":
        return SerpApiBBOWebSearchProvider(api_key=api_key)
    if normalized == "bing":
        return BingBBOWebSearchProvider(api_key=api_key)
    raise ValueError(f"Unknown BBO web search provider `{provider}`.")


class WebSearchTool(BaseBBOTool):
    name = "web_search"
    description = "Search the web for public task priors and record source metadata."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
    }

    async def execute(self, context: BBOToolContext, query: str, limit: int = 5, **_: Any) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must be non-empty.")
        provider = context.web_search_provider or DisabledBBOWebSearchProvider()
        if not hasattr(provider, "search"):
            raise TypeError("context.web_search_provider must provide an async search(query=..., limit=...) method.")
        results = await provider.search(query=query, limit=min(max(1, int(limit)), 10))  # type: ignore[attr-defined]
        logged = [_log_source(context, {"kind": "search_result", "query": query, **item}) for item in results]
        return {"query": query, "results": logged, "count": len(logged)}


class FetchURLTool(BaseBBOTool):
    name = "fetch_url"
    description = "Fetch a public URL allowed by the BBO manifest and record the source."
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000, "default": 4000},
        },
        "required": ["url"],
    }

    async def execute(self, context: BBOToolContext, url: str, max_chars: int = 4000, **_: Any) -> dict[str, Any]:
        _validate_fetch_allowed(context, url)
        payload = await asyncio.to_thread(_fetch_text, url, min(max(200, int(max_chars)), 20000))
        return _log_source(context, {"kind": "fetched_url", "url": url, **payload})


def _log_source(context: BBOToolContext, record: dict[str, Any]) -> dict[str, Any]:
    logger = context.source_logger
    if logger is not None and hasattr(logger, "log"):
        return logger.log(record)  # type: ignore[attr-defined]
    return {"source_id": None, "timestamp": time.time(), **record}


def _validate_fetch_allowed(context: BBOToolContext, url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("fetch_url requires an http(s) URL.")
    policy = context.manifest.research_policy or {}
    allowed = set(policy.get("allowed_fetch_domains", []) or [])
    if allowed and parsed.netloc not in allowed:
        raise ValueError(f"Domain `{parsed.netloc}` is not allowed by the BBO manifest.")
    if policy.get("allow_external_research") is False and not allowed:
        raise ValueError("External URL fetching is disabled by the BBO manifest.")


def _fetch_text(url: str, max_chars: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "agentic-bbo/0.1"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            raw = resp.read(max_chars + 1)
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc), "content": "", "content_type": ""}
    text = raw.decode("utf-8", errors="replace")
    return {
        "ok": True,
        "content_type": content_type,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout_seconds: float, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


__all__ = [
    "BBOWebSourceLogger",
    "BingBBOWebSearchProvider",
    "DisabledBBOWebSearchProvider",
    "FetchURLTool",
    "MockBBOWebSearchProvider",
    "SerpApiBBOWebSearchProvider",
    "TavilyBBOWebSearchProvider",
    "WebSearchTool",
    "create_BBO_web_search_provider",
]
