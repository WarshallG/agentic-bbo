from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bbo.algorithms.agentic.sandbox_healthcheck import run_healthcheck


class SandboxFusionMockHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, str]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.requests.append(body)
        code = body.get("code", "")
        if "GaussianProcessRegressor" in code:
            stdout = "gp-ok\n"
        elif "numpy" in code:
            stdout = "6\n"
        else:
            stdout = "4\n"
        payload = {
            "status": "Success",
            "run_result": {
                "status": "Finished",
                "return_code": 0,
                "stdout": stdout,
                "stderr": "",
            },
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def test_sandbox_healthcheck_uses_run_code_api() -> None:
    SandboxFusionMockHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SandboxFusionMockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_healthcheck(f"http://127.0.0.1:{server.server_port}", timeout_seconds=5)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result["ok"] is True
    assert [item["name"] for item in result["checks"]] == ["python_basic", "numpy", "sklearn_gp"]
    assert len(SandboxFusionMockHandler.requests) == 3
    assert all(request["language"] == "python" for request in SandboxFusionMockHandler.requests)
