"""Health check for SandboxFusion-compatible BBO code execution."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SandboxCheck:
    name: str
    code: str
    expected_stdout: str


CHECKS = [
    SandboxCheck("python_basic", "print(2 + 2)", "4"),
    SandboxCheck("numpy", "import numpy as np\nprint(np.array([1, 2, 3]).sum())", "6"),
    SandboxCheck(
        "sklearn_gp",
        "\n".join(
            [
                "import numpy as np",
                "from sklearn.gaussian_process import GaussianProcessRegressor",
                "from sklearn.gaussian_process.kernels import RBF, WhiteKernel",
                "x = np.array([[0.0], [0.5], [1.0]])",
                "y = np.array([1.0, 0.0, 1.0])",
                "gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level=1e-5), random_state=0)",
                "gp.fit(x, y)",
                "mu, sigma = gp.predict(np.array([[0.25]]), return_std=True)",
                "print('gp-ok')",
            ]
        ),
        "gp-ok",
    ),
]


def run_healthcheck(base_url: str, *, timeout_seconds: float = 120.0) -> dict[str, Any]:
    """Run BBO's required SandboxFusion probes against ``base_url``."""

    if not base_url.strip():
        raise ValueError("SandboxFusion base URL is required.")
    results = []
    started = time.monotonic()
    for check in CHECKS:
        check_started = time.monotonic()
        payload = _run_code(base_url, check.code, timeout_seconds=timeout_seconds)
        stdout = _stdout(payload)
        return_code = _return_code(payload)
        ok = return_code == 0 and check.expected_stdout in stdout
        results.append(
            {
                "name": check.name,
                "ok": ok,
                "expected_stdout": check.expected_stdout,
                "stdout": stdout,
                "return_code": return_code,
                "status": payload.get("status"),
                "message": payload.get("message"),
                "duration_ms": round((time.monotonic() - check_started) * 1000.0, 3),
            }
        )
    return {
        "ok": all(item["ok"] for item in results),
        "base_url": base_url,
        "endpoint": urllib.parse.urljoin(base_url.rstrip("/") + "/", "run_code"),
        "checks": results,
        "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        "required_libraries": ["numpy", "scipy", "scikit-learn", "pandas", "joblib"],
    }


def _run_code(base_url: str, code: str, *, timeout_seconds: float) -> dict[str, Any]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "run_code")
    body = json.dumps({"code": code, "language": "python"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {"status": "Error", "message": f"SandboxFusion request failed: {exc}", "run_result": {"return_code": 1, "stdout": "", "stderr": str(exc)}}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "Error", "message": "SandboxFusion returned non-JSON response.", "raw": raw, "run_result": {"return_code": 1, "stdout": "", "stderr": raw[:1000]}}
    return payload if isinstance(payload, dict) else {"status": "Error", "message": "Unexpected response shape.", "run_result": {"return_code": 1, "stdout": "", "stderr": str(payload)}}


def _stdout(payload: dict[str, Any]) -> str:
    run_result = payload.get("run_result")
    if isinstance(run_result, dict):
        return str(run_result.get("stdout") or "")
    return str(payload.get("stdout") or "")


def _return_code(payload: dict[str, Any]) -> int:
    run_result = payload.get("run_result")
    if isinstance(run_result, dict):
        try:
            return int(run_result.get("return_code", 0))
        except Exception:
            return 1
    if str(payload.get("status", "")).lower() in {"success", "finished"}:
        return 0
    return 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a SandboxFusion /run_code endpoint for BBO agent use.")
    parser.add_argument("--base-url", default=os.environ.get("SANDBOX_FUSION_BASE_URL"))
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.base_url:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "missing_base_url",
                    "message": "Pass --base-url or set SANDBOX_FUSION_BASE_URL.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    result = run_healthcheck(str(args.base_url), timeout_seconds=float(args.timeout_seconds))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
