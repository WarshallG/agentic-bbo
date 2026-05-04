"""Local HTTP bridge for BBOPlace-Bench's Python evaluator.

This lets the benchmark keep its existing ``/evaluate`` HTTP contract even when
the user does not want to run the published Docker image. The bridge expects an
upstream BBOPlace-Bench checkout with datasets and evaluator dependencies
already prepared, then exposes a tiny JSON API compatible with
``bbo.tasks.bboplace.task.BBOPlaceTask``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class BBOPlaceEvaluatorKey:
    benchmark: str
    placer: str
    seed: int
    n_macro: int | None = None
    eval_gp_hpwl: bool = False


EvaluatorFactory = Callable[[Path, BBOPlaceEvaluatorKey], Any]


class _MiniBenchLocalEvaluator:
    """Minimal non-Ray wrapper around BBOPlace-miniBench's GG/SP placers."""

    def __init__(self, root: Path, key: BBOPlaceEvaluatorKey) -> None:
        root = root.expanduser().resolve()
        _import_module(root, "utils.args_parser", add_src=True)
        args_parser_module = sys.modules["utils.args_parser"]
        parse_args = getattr(args_parser_module, "parse_args")
        placedb_module = _import_module(root, "placedb", add_src=True)
        placer_module = _import_module(root, "placer", add_src=True)

        args = SimpleNamespace(
            placer=_resolve_minibench_placer(key.placer),
            benchmark=_resolve_minibench_benchmark(root, key.benchmark),
            seed=int(key.seed),
            n_cpu=1,
            **({"n_macro": int(key.n_macro)} if key.n_macro is not None else {}),
        )
        args = parse_args(args)
        self.args = args
        self.placedb = getattr(placedb_module, "PlaceDB")(args=args)
        self.placer = getattr(placer_module, "REGISTRY")[args.placer.lower()](args=args, placedb=self.placedb)

    @property
    def n_dim(self) -> int:
        return int(self.placedb.node_cnt) * 2

    @property
    def xl(self) -> np.ndarray:
        return np.zeros(self.n_dim, dtype=float)

    @property
    def xu(self) -> np.ndarray:
        node_cnt = int(self.placedb.node_cnt)
        if self.args.placer == "gg":
            return np.asarray(([self.args.n_grid_x] * node_cnt) + ([self.args.n_grid_y] * node_cnt), dtype=float)
        if self.args.placer == "sp":
            return np.asarray([node_cnt] * self.n_dim, dtype=float)
        raise ValueError(f"Unsupported miniBench placer `{self.args.placer}`.")

    def evaluate(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if isinstance(x, list):
            x = np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        hpwl_list = []
        for row in x:
            res, _macro_pos = self.placer.evaluate(row)
            hpwl_list.append(float(res["hpwl"]))
        return {"hpwl": np.asarray(hpwl_list, dtype=float)}


def _looks_like_minibench(root: Path) -> bool:
    return "minibench" in root.name.lower()


def _resolve_minibench_benchmark(root: Path, benchmark: str) -> str:
    if "/" in benchmark:
        return benchmark
    candidates = [
        root / "benchmarks" / "ispd2005" / benchmark,
        root / "benchmarks" / "iccad2015" / benchmark,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return f"{candidate.parent.name}/{benchmark}"
    return f"ispd2005/{benchmark}"


def _resolve_minibench_placer(placer: str) -> str:
    normalized = placer.strip().lower()
    if normalized == "mgo":
        return "gg"
    if normalized in {"gg", "sp"}:
        return normalized
    raise ValueError(f"BBOPlace-miniBench does not support placer `{placer}`.")


def _import_module(root: Path, module_name: str, *, add_src: bool = False) -> ModuleType:
    root_str = str(root.resolve())
    src_str = str((root / "src").resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if add_src and src_str not in sys.path:
        sys.path.insert(0, src_str)
    return __import__(module_name, fromlist=["*"])


def build_upstream_evaluator(upstream_root: Path, key: BBOPlaceEvaluatorKey) -> Any:
    """Import and instantiate an upstream BBOPlace evaluator on demand."""

    root = upstream_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BBOPlace upstream root does not exist: {root}")
    try:
        if _looks_like_minibench(root):
            return _MiniBenchLocalEvaluator(root, key)

        evaluator_module = _import_module(root, "src.evaluator", add_src=False)
        evaluator_cls = getattr(evaluator_module, "Evaluator")
        args = SimpleNamespace(
            placer=key.placer,
            benchmark=key.benchmark,
            eval_gp_hpwl=bool(key.eval_gp_hpwl),
            seed=int(key.seed),
            **({"n_macro": int(key.n_macro)} if key.n_macro is not None else {}),
        )
        return evaluator_cls(args)
    except ImportError as exc:  # pragma: no cover - depends on external checkout.
        raise ImportError(
            "Could not import an evaluator from the upstream BBOPlace checkout. "
            "Set `BBOPLACE_UPSTREAM_ROOT` to a prepared `BBOPlace-Bench` or `BBOPlace-miniBench` root."
        ) from exc


class BBOPlaceLocalBridge:
    """Stateful adapter from the benchmark JSON payload to upstream ``Evaluator``."""

    def __init__(
        self,
        upstream_root: Path,
        *,
        eval_gp_hpwl: bool = False,
        evaluator_factory: EvaluatorFactory | None = None,
    ) -> None:
        self.upstream_root = upstream_root.expanduser().resolve()
        self.eval_gp_hpwl = bool(eval_gp_hpwl)
        self._evaluator_factory = evaluator_factory or build_upstream_evaluator
        self._evaluators: dict[BBOPlaceEvaluatorKey, Any] = {}

    def _evaluator_for(self, *, benchmark: str, placer: str, seed: int, n_macro: int | None) -> Any:
        key = BBOPlaceEvaluatorKey(
            benchmark=str(benchmark),
            placer=str(placer),
            seed=int(seed),
            n_macro=int(n_macro) if n_macro is not None else None,
            eval_gp_hpwl=self.eval_gp_hpwl,
        )
        evaluator = self._evaluators.get(key)
        if evaluator is None:
            evaluator = self._evaluator_factory(self.upstream_root, key)
            self._evaluators[key] = evaluator
        return evaluator

    def evaluate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        benchmark = str(payload.get("benchmark", "")).strip()
        if not benchmark:
            raise ValueError("Payload must include non-empty `benchmark`.")
        placer = str(payload.get("placer", "mgo")).strip() or "mgo"
        seed = int(payload.get("seed", 0))
        n_macro = int(payload.get("n_macro")) if payload.get("n_macro") is not None else None
        evaluator = self._evaluator_for(benchmark=benchmark, placer=placer, seed=seed, n_macro=n_macro)

        raw_x = payload.get("x")
        if raw_x is None:
            raise ValueError("Payload must include `x`.")
        matrix = np.asarray(raw_x, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
            raise ValueError("Payload `x` must decode to a non-empty 2D float array.")

        expected_dim = int(getattr(evaluator, "n_dim"))
        if matrix.shape[1] != expected_dim:
            raise ValueError(f"Evaluator expects vectors of length {expected_dim}, got {matrix.shape[1]}.")

        if n_macro is not None and int(n_macro) * 2 != expected_dim:
            raise ValueError(f"Payload `n_macro={n_macro}` does not match evaluator dimension {expected_dim}.")

        hpwl_raw = evaluator.evaluate(matrix)
        if isinstance(hpwl_raw, tuple):
            if not hpwl_raw:
                raise ValueError("Upstream evaluator returned an empty tuple.")
            first = hpwl_raw[0]
            if isinstance(first, dict) and "hpwl" in first:
                hpwl_raw = first["hpwl"]
            else:
                hpwl_raw = first
        elif isinstance(hpwl_raw, dict) and "hpwl" in hpwl_raw:
            hpwl_raw = hpwl_raw["hpwl"]
        hpwl = np.asarray(hpwl_raw, dtype=float).reshape(-1)
        if hpwl.size == 0:
            raise ValueError("Upstream evaluator returned an empty `hpwl` array.")
        if not np.all(np.isfinite(hpwl)):
            raise ValueError("Upstream evaluator returned a non-finite `hpwl` value.")

        return {
            "status": "success",
            "hpwl": [float(value) for value in hpwl.tolist()],
        }


class _Handler(BaseHTTPRequestHandler):
    bridge: BBOPlaceLocalBridge

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "bboplace_local_bridge",
                    "upstream_root": str(self.bridge.upstream_root),
                },
            )
            return
        self._send_json(404, {"status": "error", "message": f"Unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.rstrip("/") != "/evaluate":
            self._send_json(404, {"status": "error", "message": f"Unknown path: {self.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            response = self.bridge.evaluate_payload(payload)
        except ValueError as exc:
            self._send_json(400, {"status": "error", "error_type": type(exc).__name__, "message": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - depends on upstream evaluator.
            self._send_json(500, {"status": "error", "error_type": type(exc).__name__, "message": str(exc)})
            return

        self._send_json(200, response)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local HTTP bridge for upstream BBOPlace-Bench.")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path.cwd(),
        help="Prepared upstream BBOPlace-Bench checkout root. Can also be set via BBOPLACE_UPSTREAM_ROOT.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--eval-gp-hpwl", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configured_root = Path(os.environ.get("BBOPLACE_UPSTREAM_ROOT", str(args.upstream_root)))
    bridge = BBOPlaceLocalBridge(
        configured_root,
        eval_gp_hpwl=bool(args.eval_gp_hpwl),
    )
    handler_cls = type("BBOPlaceLocalHandler", (_Handler,), {"bridge": bridge})
    server = ThreadingHTTPServer((args.host, int(args.port)), handler_cls)
    print(f"BBOPlace local bridge listening on http://{args.host}:{args.port}")
    print(f"Upstream root: {configured_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown.
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
