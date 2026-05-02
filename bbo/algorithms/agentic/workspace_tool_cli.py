"""Workspace-local BBO tool bridge for shell/file based agents."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def main(argv: list[str] | None = None, *, default_config_path: str | None = None) -> int:
    """Run one workspace BBO tool and print a JSON result."""

    parser = argparse.ArgumentParser(description="Call a BBO workspace tool.")
    parser.add_argument("tool_name", help="Tool name from tool_specs.json.")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON object with tool arguments.")
    parser.add_argument("--config", default=default_config_path or "bbo_tool_config.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = _read_json(config_path)
    started = time.monotonic()
    timestamp = time.time()
    try:
        arguments = json.loads(args.arguments)
        if not isinstance(arguments, dict):
            raise TypeError("arguments must decode to a JSON object.")
        result = _execute(args.tool_name, arguments, config)
        payload = {"ok": True, "result": result}
        success = True
    except Exception as exc:
        payload = {"ok": False, "error": "exception", "message": str(exc)}
        success = False
    _log_call(config, args.tool_name, args.arguments, payload, started, timestamp, success, interface="workspace_cli")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if success else 2


def _execute(tool_name: str, arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "get_history": "get_trial_history",
        "get_space": "get_search_space",
        "get_objective": "get_objective",
        "get_tool_specs": "get_tool_specs",
        "get_manifest": "get_manifest",
    }
    tool_name = aliases.get(tool_name, tool_name)
    handlers: dict[str, ToolHandler] = {
        "get_task_context": _get_task_context,
        "get_manifest": _get_manifest,
        "get_search_space": _get_search_space,
        "get_objective": _get_objective,
        "get_tool_specs": _get_tool_specs,
        "get_trial_history": _get_trial_history,
        "get_incumbent": _get_incumbent,
        "validate_candidates": _validate_candidates,
        "sample_candidates": _sample_candidates,
        "analyze_history": _analyze_history,
        "memory_read": _memory_read,
        "memory_write": _memory_write,
        "code_interpreter": _code_interpreter,
        "web_search": _web_search,
        "fetch_url": _fetch_url,
    }
    if tool_name not in handlers:
        raise ValueError(f"Unknown BBO workspace tool `{tool_name}`.")
    return handlers[tool_name](arguments, config)


def _get_task_context(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    workspace = _workspace(config)
    max_chars = int(arguments.get("max_chars_per_section", 4000))
    include_manifest = bool(arguments.get("include_manifest", True))
    task_md = (workspace / "task.md").read_text(encoding="utf-8")
    requested = arguments.get("sections")
    section_map = _markdown_sections(task_md)
    if requested:
        wanted = {str(item) for item in requested}
        sections = {name: text for name, text in section_map.items() if name in wanted}
    else:
        sections = section_map
    return {
        "task_id": _manifest(config).get("task_id"),
        "objective": _read_json(workspace / "objective.json"),
        "sections": {key: _truncate(value, max_chars) for key, value in sections.items()},
        "manifest": _manifest(config) if include_manifest else None,
    }


def _get_search_space(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    parameters = _parameters(config)
    return {
        "parameters": parameters,
        "defaults": {param["name"]: param.get("default") for param in parameters},
        "dimension": len(parameters),
    }


def _get_manifest(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return {"manifest": _manifest(config)}


def _get_objective(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return {"objective": _objective(config)}


def _get_tool_specs(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return _read_json(_workspace(config) / "tool_specs.json")


def _get_trial_history(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    mode = str(arguments.get("mode", "recent"))
    limit = max(1, int(arguments.get("limit", 20)))
    offset = max(0, int(arguments.get("offset", 0)))
    trials = _history(config)
    if mode == "recent":
        ordered = list(reversed(trials))
    elif mode == "best":
        objective = _objective(config)
        name = objective["name"]
        reverse = objective.get("direction") == "maximize"
        ordered = sorted(
            [trial for trial in trials if trial.get("status") == "success" and name in trial.get("objectives", {})],
            key=lambda trial: float(trial["objectives"][name]),
            reverse=reverse,
        )
    elif mode == "all":
        ordered = trials
    else:
        raise ValueError("mode must be one of recent, best, all.")
    page = ordered[offset : offset + limit]
    return {"mode": mode, "total": len(trials), "offset": offset, "limit": limit, "trials": page}


def _get_incumbent(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    del arguments
    data = _read_json(_workspace(config) / "incumbent.json")
    if data.get("config") is None:
        return {"incumbent": None}
    return {"incumbent": data}


def _validate_candidates(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    candidates = arguments.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list.")
    history_ids = {_identity(trial.get("config", {})) for trial in _history(config)}
    payload_ids: set[str] = set()
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
            coerced = _coerce_config(raw, _parameters(config))
        except Exception as exc:
            invalid.append({"index": index, "error": str(exc)})
            continue
        identity = _identity(coerced)
        duplicate = identity in history_ids or identity in payload_ids
        payload_ids.add(identity)
        valid.append({"index": index, "config": coerced, "duplicate": duplicate, "identity": identity})
    return {"valid": valid, "invalid": invalid, "valid_count": len(valid), "invalid_count": len(invalid)}


def _sample_candidates(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    n = min(max(1, int(arguments.get("n", 4))), 128)
    seed = int(arguments.get("seed", config.get("seed", 0)))
    strategy = str(arguments.get("strategy", "random"))
    jitter_fraction = min(max(float(arguments.get("jitter_fraction", 0.1)), 0.0), 1.0)
    rng = random.Random(seed)
    parameters = _parameters(config)
    incumbent = _read_json(_workspace(config) / "incumbent.json").get("config")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(n * 100):
        if strategy == "random":
            candidate = _sample_random(parameters, rng)
        elif strategy == "around_incumbent" and isinstance(incumbent, dict):
            candidate = _sample_around(parameters, incumbent, rng, jitter_fraction)
        elif strategy == "around_incumbent":
            candidate = _sample_random(parameters, rng)
        else:
            raise ValueError("strategy must be random or around_incumbent.")
        identity = _identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append({"config": candidate, "identity": identity})
        if len(candidates) >= n:
            break
    return {"strategy": strategy, "candidates": candidates, "count": len(candidates)}


def _analyze_history(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    limit = int(arguments.get("limit", 100))
    objective = _objective(config)
    objective_name = objective["name"]
    trials = [trial for trial in _history(config) if trial.get("status") == "success" and objective_name in trial.get("objectives", {})]
    if limit > 0:
        trials = trials[-limit:]
    if not trials:
        return {"history_size": len(_history(config)), "success_count": 0, "primary_objective": objective_name}
    scores = [float(trial["objectives"][objective_name]) for trial in trials]
    best_index = min(range(len(scores)), key=scores.__getitem__)
    if objective.get("direction") == "maximize":
        best_index = max(range(len(scores)), key=scores.__getitem__)
    analysis: dict[str, Any] = {
        "history_size": len(_history(config)),
        "success_count": len(trials),
        "primary_objective": objective_name,
        "direction": objective.get("direction"),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": statistics.fmean(scores),
        "best_trial": trials[best_index],
        "numeric_correlations": {},
        "categorical_groups": {},
    }
    for param in _parameters(config):
        name = param["name"]
        values = [trial.get("config", {}).get(name) for trial in trials]
        if param["type"] in {"float", "int"}:
            xs = [float(value) for value in values]
            corr = _correlation(xs, scores)
            if corr is not None:
                analysis["numeric_correlations"][name] = corr
        elif param["type"] == "categorical":
            groups: dict[str, list[float]] = {}
            for value, score in zip(values, scores, strict=True):
                groups.setdefault(str(value), []).append(score)
            analysis["categorical_groups"][name] = {
                key: {"count": len(vals), "mean": statistics.fmean(vals)}
                for key, vals in groups.items()
            }
    return analysis


def _memory_read(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    path = _memory_path(config)
    kind = arguments.get("kind")
    tags = set(str(tag) for tag in arguments.get("tags", []) or [])
    limit = max(1, int(arguments.get("limit", 20)))
    if not path.exists():
        return {"enabled": True, "records": [], "count": 0}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if kind is not None and record.get("kind") != kind:
            continue
        if tags and not tags.issubset(set(str(tag) for tag in record.get("tags", []))):
            continue
        records.append(record)
    records = records[-limit:]
    return {"enabled": True, "records": records, "count": len(records)}


def _memory_write(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    kind = str(arguments.get("kind", "")).strip()
    content = str(arguments.get("content", "")).strip()
    if not kind or not content:
        raise ValueError("memory_write requires non-empty kind and content.")
    path = _memory_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": f"mem_{int(time.time() * 1000)}",
        "timestamp": time.time(),
        "kind": kind,
        "content": content,
        "tags": list(arguments.get("tags", []) or []),
        "source_call_id": arguments.get("source_call_id"),
        "trial_range": arguments.get("trial_range"),
        "metadata": arguments.get("metadata") or {},
    }
    _append_jsonl(path, record)
    summary_path = Path(config["memory_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"updated_at": time.time(), "record_count": _line_count(path)}, indent=2), encoding="utf-8")
    return {"enabled": True, "written": True, "record": record}


def _code_interpreter(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    code = str(arguments.get("code", ""))
    language = str(arguments.get("language", "python"))
    if not code.strip():
        raise ValueError("code must be non-empty.")
    backend = str(config.get("code_backend", "disabled")).strip().lower().replace("-", "_")
    manifest_policy = (_manifest(config).get("tool_policy") or {}).get("code_interpreter") or {}
    if manifest_policy.get("enabled") is False and backend not in {"mock"}:
        sandbox_result = {
            "status": "Disabled",
            "message": "The BBO manifest disables code_interpreter for this benchmark.",
            "run_result": None,
        }
    elif backend == "mock":
        sandbox_result = {
            "status": "Success",
            "message": "",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "execution_time": 0.0,
                "return_code": 0,
                "stdout": f"mock {language}: {len(code)} chars\n",
                "stderr": "",
            },
            "files": {},
        }
    elif backend == "sandboxfusion" and config.get("sandbox_fusion_base_url"):
        sandbox_result = _sandboxfusion_run(str(config["sandbox_fusion_base_url"]), code, language)
    else:
        sandbox_result = {
            "status": "Disabled",
            "message": "BBO code execution is disabled. Configure SandboxFusion to enable this tool.",
            "run_result": None,
        }
    return {
        "backend": backend,
        "language": language,
        "sandbox_result": sandbox_result,
        "budget_consumed": False,
    }


def _web_search(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query must be non-empty.")
    limit = min(max(1, int(arguments.get("limit", 5))), 10)
    provider = str(config.get("web_search_provider", "disabled")).strip().lower().replace("-", "_")
    policy = (_manifest(config).get("tool_policy") or {}).get("web_search") or {}
    if policy.get("enabled") is False and provider != "mock":
        return {"enabled": False, "query": query, "results": [], "count": 0}
    if provider in {"", "disabled", "none"}:
        return {"enabled": False, "query": query, "results": [], "count": 0}
    if provider == "mock":
        raw_results = [
            {
                "title": "Mock BBO prior",
                "url": "https://example.test/bbo-prior",
                "snippet": f"Mock search result for {query}",
            }
        ][:limit]
    elif provider == "tavily":
        raw_results = _tavily_search(query, limit, str(config.get("web_search_api_key_env") or "TAVILY_API_KEY"))
    elif provider == "serpapi":
        raw_results = _serpapi_search(query, limit, str(config.get("web_search_api_key_env") or "SERPAPI_API_KEY"))
    elif provider == "bing":
        raw_results = _bing_search(query, limit, str(config.get("web_search_api_key_env") or "BING_SEARCH_API_KEY"))
    elif provider == "search_r1":
        raw_results = _search_r1_search(query, limit, str(config.get("search_r1_base_url") or ""))
    else:
        raise ValueError(f"Unknown BBO web search provider `{provider}`.")
    logged = [_log_source(config, {"kind": "search_result", "query": query, **item}) for item in raw_results]
    return {"enabled": True, "query": query, "results": logged, "count": len(logged)}


def _fetch_url(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url", "")).strip()
    max_chars = min(max(200, int(arguments.get("max_chars", 4000))), 20000)
    if not url:
        raise ValueError("url must be non-empty.")
    allowed = _fetch_allowed(url, _manifest(config).get("research_policy") or {})
    if not allowed["ok"]:
        return {"enabled": False, "url": url, "reason": allowed["reason"], "content": ""}
    fetched = _fetch_text(url, max_chars)
    return _log_source(config, {"kind": "fetched_url", "url": url, **fetched})


def _workspace(config: dict[str, Any]) -> Path:
    return Path(config["workspace_dir"])


def _parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_read_json(_workspace(config) / "space.json").get("parameters", []))


def _objective(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_workspace(config) / "objective.json")


def _manifest(config: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_workspace(config) / "manifest.json")


def _history(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = _workspace(config) / "history.jsonl"
    if not path.exists():
        return []
    trials = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            trials.append(json.loads(line))
    return trials


def _memory_path(config: dict[str, Any]) -> Path:
    return Path(config["memory_path"])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _log_call(
    config: dict[str, Any],
    tool_name: str,
    raw_arguments: str,
    payload: dict[str, Any],
    started: float,
    timestamp: float,
    success: bool,
    *,
    interface: str,
) -> None:
    path = Path(config["tool_calls_path"])
    try:
        arguments: Any = json.loads(raw_arguments)
    except Exception:
        arguments = raw_arguments
    _append_jsonl(
        path,
        {
            "timestamp": timestamp,
            "call_id": "workspace_cli",
            "tool_call_id": f"workspace_cli_{int(timestamp * 1000)}",
            "tool_name": tool_name,
            "arguments": arguments,
            "success": success,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "result_preview": _truncate(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), 1200),
            "interface": interface,
        },
    )


def _log_source(config: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "source_id": f"src_{int(time.time() * 1000)}_{abs(hash(json.dumps(record, sort_keys=True, default=str))) % 1000000:06d}",
        "timestamp": time.time(),
        **record,
    }
    _append_jsonl(Path(config["sources_path"]), payload)
    return payload


def _markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"document": []}
    current = "document"
    for line in text.splitlines():
        if line.startswith("#"):
            name = line.lstrip("#").strip().lower().replace(" ", "_").replace("-", "_")
            if name:
                current = name
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if "\n".join(value).strip()}


def _coerce_config(raw: dict[str, Any], parameters: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {param["name"] for param in parameters}
    provided = set(raw)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        raise ValueError(f"config keys mismatch; missing={missing}, extra={extra}")
    result: dict[str, Any] = {}
    for param in parameters:
        name = param["name"]
        kind = param["type"]
        value = raw[name]
        if kind == "float":
            coerced = float(value)
            low = float(param["low"])
            high = float(param["high"])
            if not low <= coerced <= high:
                raise ValueError(f"{name}={coerced} outside [{low}, {high}]")
            result[name] = coerced
        elif kind == "int":
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, not bool")
            coerced = int(value)
            if coerced != value and not (isinstance(value, float) and value.is_integer()):
                raise ValueError(f"{name}={value!r} is not an integer")
            low = int(param["low"])
            high = int(param["high"])
            if not low <= coerced <= high:
                raise ValueError(f"{name}={coerced} outside [{low}, {high}]")
            result[name] = coerced
        elif kind == "categorical":
            choices = list(param["choices"])
            if value not in choices:
                raise ValueError(f"{name}={value!r} is not one of {choices!r}")
            result[name] = value
        else:
            raise ValueError(f"unsupported parameter type {kind!r}")
    return result


def _sample_random(parameters: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    for param in parameters:
        if param["type"] == "float":
            candidate[param["name"]] = rng.uniform(float(param["low"]), float(param["high"]))
        elif param["type"] == "int":
            candidate[param["name"]] = rng.randint(int(param["low"]), int(param["high"]))
        elif param["type"] == "categorical":
            candidate[param["name"]] = rng.choice(list(param["choices"]))
        else:
            raise ValueError(f"unsupported parameter type {param['type']!r}")
    return candidate


def _sample_around(parameters: list[dict[str, Any]], incumbent: dict[str, Any], rng: random.Random, fraction: float) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    for param in parameters:
        name = param["name"]
        current = incumbent.get(name, param.get("default"))
        if param["type"] == "float":
            low = float(param["low"])
            high = float(param["high"])
            span = high - low
            candidate[name] = min(max(float(current) + rng.uniform(-span * fraction, span * fraction), low), high)
        elif param["type"] == "int":
            low = int(param["low"])
            high = int(param["high"])
            span = high - low
            step = max(1, int(round(span * fraction)))
            candidate[name] = min(max(int(current) + rng.randint(-step, step), low), high)
        elif param["type"] == "categorical":
            candidate[name] = current if rng.random() > fraction else rng.choice(list(param["choices"]))
        else:
            raise ValueError(f"unsupported parameter type {param['type']!r}")
    return _coerce_config(candidate, parameters)


def _identity(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(var_x * var_y)


def _sandboxfusion_run(base_url: str, code: str, language: str) -> dict[str, Any]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "run_code")
    body = json.dumps({"code": code, "language": language}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {"status": "Error", "message": f"SandboxFusion request failed: {exc}", "run_result": None}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "Error", "message": "SandboxFusion returned non-JSON response.", "raw": raw}
    return payload if isinstance(payload, dict) else {"status": "Error", "message": "Unexpected response shape."}


def _tavily_search(query: str, limit: int, api_key_env: str) -> list[dict[str, Any]]:
    api_key = _required_env(api_key_env)
    data = _post_json("https://api.tavily.com/search", {"api_key": api_key, "query": query, "max_results": limit}, 30.0)
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "snippet": str(item.get("content", item.get("snippet", ""))),
        }
        for item in data.get("results", [])
        if isinstance(item, dict)
    ]


def _serpapi_search(query: str, limit: int, api_key_env: str) -> list[dict[str, Any]]:
    api_key = _required_env(api_key_env)
    params = urllib.parse.urlencode({"engine": "google", "q": query, "api_key": api_key, "num": limit})
    data = _get_json(f"https://serpapi.com/search.json?{params}", 30.0)
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("link", "")),
            "snippet": str(item.get("snippet", "")),
        }
        for item in data.get("organic_results", [])
        if isinstance(item, dict)
    ][:limit]


def _bing_search(query: str, limit: int, api_key_env: str) -> list[dict[str, Any]]:
    api_key = _required_env(api_key_env)
    params = urllib.parse.urlencode({"q": query, "count": limit})
    data = _get_json(
        f"https://api.bing.microsoft.com/v7.0/search?{params}",
        30.0,
        {"Ocp-Apim-Subscription-Key": api_key},
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


def _search_r1_search(query: str, limit: int, base_url: str) -> list[dict[str, Any]]:
    endpoint = _search_r1_retrieve_url(base_url or os.environ.get("AGENT_SEARCH_R1_BASE_URL", ""))
    data = _post_json(endpoint, {"queries": [query], "topk": limit, "return_scores": True}, 30.0)
    return _parse_search_r1_results(data, limit=limit)


def _search_r1_retrieve_url(base_url: str) -> str:
    stripped = base_url.strip().rstrip("/")
    if not stripped:
        raise RuntimeError("Search-R1 provider requires AGENT_SEARCH_R1_BASE_URL or `search_r1_base_url` in config.")
    if stripped.endswith("/retrieve"):
        return stripped
    return urllib.parse.urljoin(stripped + "/", "retrieve")


def _parse_search_r1_results(data: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    raw_results = data.get("result", data.get("results", []))
    if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
        raw_items = raw_results[0]
    elif isinstance(raw_results, list):
        raw_items = raw_results
    else:
        raw_items = []
    parsed = []
    for item in raw_items[:limit]:
        if not isinstance(item, dict):
            continue
        result = _normalize_search_r1_item(item)
        if result is not None:
            parsed.append(result)
    return parsed


def _normalize_search_r1_item(item: dict[str, Any]) -> dict[str, Any] | None:
    document = item.get("document") or item.get("doc") or item
    if not isinstance(document, dict):
        return None
    contents = str(document.get("contents", document.get("text", document.get("content", ""))))
    parsed_title, parsed_snippet = _split_search_r1_contents(contents)
    title = str(document.get("title") or item.get("title") or parsed_title or "Search-R1 result")
    url = str(document.get("url") or document.get("link") or item.get("url") or item.get("link") or "")
    snippet = str(document.get("snippet") or item.get("snippet") or parsed_snippet or contents)
    result: dict[str, Any] = {
        "title": title,
        "url": url,
        "snippet": snippet,
        "raw_document": document,
    }
    if "score" in item:
        result["score"] = item["score"]
    return result


def _split_search_r1_contents(contents: str) -> tuple[str, str]:
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0].strip().strip('"')
    snippet = "\n".join(lines[1:]).strip() if len(lines) > 1 else title
    return title, snippet


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable `{name}`.")
    return value


def _fetch_allowed(url: str, policy: dict[str, Any]) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "reason": "fetch_url requires an http(s) URL"}
    allowed = set(policy.get("allowed_fetch_domains", []) or [])
    if allowed and parsed.netloc not in allowed:
        return {"ok": False, "reason": f"Domain `{parsed.netloc}` is not allowed by the BBO manifest"}
    if policy.get("allow_external_research") is False and not allowed:
        return {"ok": False, "reason": "External URL fetching is disabled by the BBO manifest"}
    return {"ok": True}


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


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
