"""Minimal JSON-over-HTTP client for task evaluators (stdlib only, no requests)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

_DEFAULT_TIMEOUT_SEC = 300.0


def post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """POST JSON and parse JSON response."""
    url = path if path.startswith("http") else urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc
    return json.loads(raw)


def get_json(
    base_url: str,
    path: str,
    *,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """GET JSON (e.g. health). On HTTPError, read body and surface server message."""
    url = path if path.startswith("http") else urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # NB: HTTPError is a URLError subclass; do not only catch URLError or body is lost.
        err_raw = exc.read().decode("utf-8", errors="replace")
        detail = err_raw
        if err_raw.strip():
            try:
                parsed = json.loads(err_raw)
                if isinstance(parsed, dict) and "message" in parsed:
                    detail = str(parsed["message"])
            except (json.JSONDecodeError, TypeError):
                pass
        raise RuntimeError(
            f"GET HTTP {exc.code} {url!r}{': ' + detail if detail else ''}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc
    return json.loads(raw)


__all__ = ["get_json", "post_json"]
