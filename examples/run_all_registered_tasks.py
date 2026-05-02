"""Run all registered `bbo` tasks in one batch, with a dedicated BBOPlace matrix.

需要访问评估器的任务**默认不盲跑**：会探测本机 TCP 端口，只有端口能连上才排进队列：

- **8070** → BBOPlace（`bboplace_bench` 矩阵，容器内 8080 映射到宿主机 8070）
- **8080** → `knob_http_mariadb_sysbench_*`
- **8090** → `knob_http_surrogate_*`

`--include-http` 可**跳过探测、强制**排入所有上述任务（你确信服务已就绪时用）。
`--skip-http` 可**不跑任何** `knob_http_*`（纯本地/CI，无 MariaDB/代理容器）。

BBOPlace 与 `run_bboplace_demo.py` 同路径，默认 **Optuna TPE**；矩阵 `adaptec1` / `bigblue1` × `n_macro` 128/256。

**图**：若未加 `--no-plots`，`run_single_experiment` 会在每次运行的 `run_dir/plots/` 下写 trace、distribution、
per-trial / cumulative 用时等；2D 任务可能另有 landscape、regret。若一个 trial 都未完成（`trials.jsonl` 空），
则不会生成图片。

批跑结束会在 `runs/demo/<results-subdir>/` 下写 **CSV + JSON**：CSV 行=算法，**列名=各任务/实验 id**（每任务一列）；格内为单个 Y 或多目标时一行 JSON。
**JSON** 含 `runs` 完整 objectives 与 `algorithm_cells`。

Usage:
    uv run python examples/run_all_registered_tasks.py
    uv run python examples/run_all_registered_tasks.py --include-http
    uv run python examples/run_all_registered_tasks.py --bboplace-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

# 在导入 bbo 之前把仓库根加入 path（支持 `python examples/...` 直跑）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bbo.run import run_single_experiment
from bbo.tasks import ALL_TASK_NAMES, BBOPLACE_TASK_KEY
from bbo.tasks.bboplace.task import default_bboplace_definition

# 默认 BBOPlace：两个 benchmark、两种 n_macro（共 4 次）：可按需改元组
BBOPLACE_CASES: tuple[tuple[str, int], ...] = (
    ("adaptec1", 128),
    ("bigblue1", 128),
)

# 与本仓库各 HTTP 任务默认约定一致（见 database.md / task env 文档）
PORT_BBOPLACE: int = 8070
PORT_MARIADB_HTTP: int = 8080
PORT_SURROGATE_HTTP: int = 8090
DEFAULT_PROBE_HOST: str = "127.0.0.1"
DEFAULT_PROBE_TIMEOUT_S: float = 0.4


@dataclass(frozen=True)
class PortProbe:
    """一次 TCP 探测结果，供日志与排程用。"""

    host: str
    bboplace: bool
    mariadb: bool
    surrogate: bool


@dataclass(frozen=True)
class RunOutcome:
    """一次实验的标签与结果/错误，便于批跑结束时汇总。"""

    label: str
    ok: bool
    summary: dict[str, Any] | None
    error: str | None


def _run_one(
    *,
    label: str,
    run: dict[str, Any],
) -> RunOutcome:
    try:
        summary = run_single_experiment(**run)
    except Exception as exc:  # noqa: BLE001 — 批跑需捕获并继续
        return RunOutcome(label=label, ok=False, summary=None, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(summary, dict):
        return RunOutcome(
            label=label,
            ok=False,
            summary=None,
            error=f"unexpected return type: {type(summary)}",
        )
    return RunOutcome(label=label, ok=True, summary=summary, error=None)


def _probe_tcp(*, host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def probe_evaluator_ports(
    host: str = DEFAULT_PROBE_HOST, *, timeout_s: float = DEFAULT_PROBE_TIMEOUT_S
) -> PortProbe:
    return PortProbe(
        host=host,
        bboplace=_probe_tcp(host=host, port=PORT_BBOPLACE, timeout_s=timeout_s),
        mariadb=_probe_tcp(host=host, port=PORT_MARIADB_HTTP, timeout_s=timeout_s),
        surrogate=_probe_tcp(host=host, port=PORT_SURROGATE_HTTP, timeout_s=timeout_s),
    )


def _knob_http_kind(name: str) -> Literal["mariadb", "surrogate"] | None:
    if not name.startswith("knob_http_"):
        return None
    if name.startswith("knob_http_mariadb"):
        return "mariadb"
    if name.startswith("knob_http_surrogate"):
        return "surrogate"
    return None


def _non_bboplace_tasks(
    *,
    include_http: bool,
    skip_http: bool,
    probe: PortProbe,
) -> tuple[str, ...]:
    out: list[str] = []
    for name in ALL_TASK_NAMES:
        if name == BBOPLACE_TASK_KEY:
            continue
        k = _knob_http_kind(name)
        if k is not None:
            if skip_http:
                continue
            if include_http:
                out.append(name)
                continue
            if (k == "mariadb" and not probe.mariadb) or (k == "surrogate" and not probe.surrogate):
                continue
        out.append(name)
    return tuple(sorted(out))


def _should_run_bboplace(
    *,
    include_http: bool,
    skip_bboplace: bool,
    probe: PortProbe,
) -> bool:
    if skip_bboplace:
        return False
    if include_http:
        return True
    return bool(probe.bboplace)


_BBO_RE = re.compile(
    r"^bboplace\s+(?P<bench>\S+)\s+n_macro=(?P<nm>\d+)",
    re.IGNORECASE,
)


def experiment_id(*, label: str, task_name: str) -> str:
    """Stable column key for an experiment: task name, or BBOPlace (benchmark, n_macro)."""
    if task_name == BBOPLACE_TASK_KEY:
        m = _BBO_RE.match(label.strip())
        if m:
            return f"bboplace:{m.group('bench')}:n{m.group('nm')}"
        return f"{BBOPLACE_TASK_KEY}:default"
    return task_name


def _objectives_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """提取各目标 Y；优先用 incumbent 的 objectives，否则用 best_primary_objective 单列。"""
    out: dict[str, Any] = {}
    incumbents = summary.get("incumbents") or []
    if incumbents and isinstance(incumbents[0], dict):
        obj = incumbents[0].get("objectives")
        if isinstance(obj, dict) and obj:
            out.update(obj)
    bpo = summary.get("best_primary_objective")
    if bpo is not None and "best_primary_objective" not in out:
        out["best_primary_objective"] = bpo
    return out


def _format_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (not (v == v) or v in (float("inf"), float("-inf"))):
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def objective_cell_value(summary: dict[str, Any]) -> str:
    """CSV 单列 per 任务：优先 best_primary_objective；多目标时写入一行 JSON（objectives）。"""
    bpo = summary.get("best_primary_objective")
    if isinstance(bpo, (int, float)) and bpo == bpo and bpo not in (float("inf"), float("-inf")):
        return _format_cell(bpo)
    objs = _objectives_from_summary(summary)
    if not objs:
        return _format_cell(bpo) if bpo is not None else ""
    if len(objs) == 1:
        return _format_cell(next(iter(objs.values())))
    return json.dumps(objs, ensure_ascii=False, sort_keys=True)


def write_batch_objectives_table(
    *,
    runs: Sequence[tuple[str, dict[str, Any], RunOutcome]],
    output_dir: Path,
    table_basename: str = "batch_objectives_table",
) -> tuple[Path, Path]:
    """Write CSV: one column per task (experiment_id), one cell = single Y or JSON of multiple Y; JSON has full runs."""
    by_alg_cells: dict[str, dict[str, str]] = {}
    detail_rows: list[dict[str, Any]] = []
    eid_order: list[str] = []
    seen_eid: set[str] = set()

    for label, run_kw, out in runs:
        task_n = str(run_kw.get("task_name", ""))
        algo = str(run_kw.get("algorithm_name", "unknown"))
        eid = experiment_id(label=label, task_name=task_n)
        if eid not in seen_eid:
            seen_eid.add(eid)
            eid_order.append(eid)
        if algo not in by_alg_cells:
            by_alg_cells[algo] = {}
        if out.ok and out.summary:
            s = out.summary
            by_alg_cells[algo][eid] = objective_cell_value(s)
            detail_rows.append(
                {
                    "algorithm": algo,
                    "experiment_id": eid,
                    "label": label,
                    "ok": True,
                    "objectives": _objectives_from_summary(s),
                    "run_dir": s.get("run_dir", ""),
                }
            )
        else:
            by_alg_cells[algo][eid] = ""
            detail_rows.append(
                {
                    "algorithm": algo,
                    "experiment_id": eid,
                    "label": label,
                    "ok": False,
                    "error": out.error,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{table_basename}.csv"
    fieldnames = ["algorithm", *eid_order]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for algo in sorted(by_alg_cells):
            row: dict[str, str] = {"algorithm": algo}
            for eid in eid_order:
                row[eid] = by_alg_cells[algo].get(eid, "")
            w.writerow(row)

    json_path = output_dir / f"{table_basename}.json"
    payload: dict[str, Any] = {
        "csv_columns": fieldnames,
        "algorithm_cells": {
            algo: {eid: by_alg_cells[algo].get(eid, "") for eid in eid_order}
            for algo in sorted(by_alg_cells)
        },
        "runs": detail_rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every registered task (optionally with HTTP services). "
        "All experiments use the same optimizer (default: Optuna TPE). "
        "BBOPlace uses BBOPLACE_CASES with `task_kwargs.definition`.",
    )
    http_mode = parser.add_mutually_exclusive_group()
    http_mode.add_argument(
        "--include-http",
        action="store_true",
        help="不探测端口，强制排入 BBOPlace + 所有 knob_http_*（假定 8070/8080/8090 已可用）。",
    )
    http_mode.add_argument(
        "--skip-http",
        action="store_true",
        help="不排入任何 knob_http_*（BBOPlace 仍由 8070 探测或 --include-http 控制）。",
    )
    parser.add_argument(
        "--http-host",
        default=DEFAULT_PROBE_HOST,
        help=f"探测与文档约定一致的服务地址（默认 {DEFAULT_PROBE_HOST}）。",
    )
    parser.add_argument(
        "--http-probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_S,
        help="TCP 连接探测超时（秒）。",
    )
    parser.add_argument(
        "--bboplace-only",
        action="store_true",
        help="Only run the BBOPlace matrix (skip all other task names).",
    )
    parser.add_argument(
        "--skip-bboplace",
        action="store_true",
        help="Run non-BBOPlace tasks only (no HTTP evaluator for placement).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for every experiment.")
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=10,
        help="Eval budget for **non-**BBOPlace tasks (BBOPlace uses --bboplace-max-evaluations).",
    )
    parser.add_argument(
        "--bboplace-max-evaluations",
        type=int,
        default=10,
        help="Eval budget for each BBOPlace run (same algorithm as --algorithm).",
    )
    parser.add_argument(
        "--algorithm",
        default="optuna_tpe",
        help="Optimizer for every experiment, including the BBOPlace matrix (public name: optuna_tpe).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip post-run plots for every experiment (faster batch).",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Do not write batch_objectives_table.csv / .json after the run.",
    )
    parser.add_argument(
        "--results-subdir",
        default="batch_all_tasks",
        help="Under runs/demo/<subdir> — keeps batch artifacts separate from one-off demos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run; do not call evaluators.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List task names that would be run (after filters) and exit.",
    )
    args = parser.parse_args()

    if args.skip_bboplace and args.bboplace_only:
        print("error: --skip-bboplace and --bboplace-only are mutually exclusive", file=sys.stderr)
        return 2

    project_root = _PROJECT_ROOT
    results_base = project_root / "runs" / "demo" / args.results_subdir

    probe = probe_evaluator_ports(args.http_host, timeout_s=args.http_probe_timeout)
    run_bboplace = _should_run_bboplace(
        include_http=args.include_http,
        skip_bboplace=bool(args.skip_bboplace),
        probe=probe,
    )
    others = _non_bboplace_tasks(
        include_http=bool(args.include_http),
        skip_http=bool(args.skip_http),
        probe=probe,
    )

    if args.list:
        print("TCP probe", probe.host, f"(timeout={args.http_probe_timeout}s):")
        print(
            f"  {PORT_BBOPLACE} BBOPlace ->",
            "ok" if probe.bboplace else "closed",
        )
        print(
            f"  {PORT_MARIADB_HTTP} MariaDB HTTP ->",
            "ok" if probe.mariadb else "closed",
        )
        print(
            f"  {PORT_SURROGATE_HTTP} surrogate ->",
            "ok" if probe.surrogate else "closed",
        )
        if args.include_http:
            print("  mode: --include-http (force all external HTTP task groups)")
        elif args.skip_http:
            print("  mode: --skip-http (no knob_http_*)")
        else:
            print("  mode: auto (按端口排 knob_http；BBOPlace 要", PORT_BBOPLACE, "可连)")
        print("non-BBOPlace tasks:", len(others))
        for t in others:
            print(" ", t)
        if not args.skip_bboplace and run_bboplace:
            print("BBOPlace cases:", len(BBOPLACE_CASES))
            for b, n in BBOPLACE_CASES:
                print(" ", f"{b} n_macro={n} algorithm={args.algorithm}")
        elif not args.skip_bboplace and not run_bboplace:
            print("BBOPlace: skipped (port not reachable; use --include-http to force)")
        return 0

    planned: list[tuple[str, dict[str, Any]]] = []

    if not args.bboplace_only:
        for task_name in others:
            label = f"{task_name} seed={args.seed} algorithm={args.algorithm}"
            planned.append(
                (
                    label,
                    {
                        "task_name": task_name,
                        "algorithm_name": args.algorithm,
                        "seed": args.seed,
                        "max_evaluations": args.max_evaluations,
                        "results_root": results_base,
                        "generate_plots": not args.no_plots,
                    },
                )
            )

    if not args.list:
        print(
            f"HTTP services @ {probe.host}: BBOPlace:{PORT_BBOPLACE}={probe.bboplace} "
            f"MariaDB:{PORT_MARIADB_HTTP}={probe.mariadb} "
            f"surrogate:{PORT_SURROGATE_HTTP}={probe.surrogate}",
            flush=True,
        )
    if not run_bboplace and not args.skip_bboplace and not args.bboplace_only:
        print(
            f"Note: {probe.host}:{PORT_BBOPLACE} 未接受 TCP 连接，已跳过 BBOPlace 矩阵；"
            f"要强制请用 --include-http 或等 evaluator 在 {PORT_BBOPLACE} 就绪。",
            file=sys.stderr,
        )
    if args.bboplace_only and not run_bboplace:
        print(
            f"error: --bboplace-only 但 {probe.host}:{PORT_BBOPLACE} 不可达；"
            f"请启动容器或加 --include-http。",
            file=sys.stderr,
        )
        return 1

    if run_bboplace:
        for benchmark, n_macro in BBOPLACE_CASES:
            definition = default_bboplace_definition(
                benchmark=benchmark,
                n_macro=n_macro,
                bench_seed=args.seed,
            )
            label = (
                f"bboplace {benchmark} n_macro={n_macro} "
                f"seed={args.seed} algorithm={args.algorithm}"
            )
            planned.append(
                (
                    label,
                    {
                        "task_name": BBOPLACE_TASK_KEY,
                        "algorithm_name": args.algorithm,
                        "seed": args.seed,
                        "max_evaluations": args.bboplace_max_evaluations,
                        "results_root": results_base,
                        "task_kwargs": {"definition": definition},
                        "generate_plots": not args.no_plots,
                    },
                )
            )

    if args.dry_run:
        print("Dry run — would execute", len(planned), "experiments under", results_base)
        for label, _ in planned:
            print(" -", label)
        return 0

    results: list[RunOutcome] = []
    for label, kwargs in planned:
        print("===", label, flush=True)
        results.append(_run_one(label=label, run=kwargs))

    runs_zip: list[tuple[str, dict[str, Any], RunOutcome]] = list(
        zip(
            [label for label, _ in planned],
            [kw for _, kw in planned],
            results,
            strict=True,
        )
    )

    if not args.no_table and runs_zip:
        csv_p, json_p = write_batch_objectives_table(
            runs=runs_zip,
            output_dir=results_base,
        )
        print("Wrote objective summary table:", csv_p)
        print("Wrote full JSON:", json_p)

    failures = [r for r in results if not r.ok]
    print(json.dumps([{"label": r.label, "ok": r.ok, "error": r.error} for r in results], indent=2))
    if failures:
        for r in failures:
            print("FAILED:", r.label, r.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
