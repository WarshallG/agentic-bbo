from __future__ import annotations

from pathlib import Path

from bbo.tasks.bboplace.local_service import (
    BBOPlaceEvaluatorKey,
    BBOPlaceLocalBridge,
    _resolve_minibench_benchmark,
    _resolve_minibench_placer,
)


class _FakeEvaluator:
    def __init__(self, n_dim: int = 4) -> None:
        self.n_dim = n_dim

    def evaluate(self, x):
        return x.sum(axis=1)


class _FakeMiniBenchEvaluator:
    def __init__(self, n_dim: int = 4) -> None:
        self.n_dim = n_dim

    def evaluate(self, x):
        return {"hpwl": x.sum(axis=1)}, [{"macro": (0, 0)} for _ in range(len(x))]


def test_bboplace_local_bridge_evaluates_payload_and_caches_evaluator(tmp_path: Path) -> None:
    calls: list[BBOPlaceEvaluatorKey] = []

    def factory(root: Path, key: BBOPlaceEvaluatorKey) -> _FakeEvaluator:
        assert root == tmp_path
        calls.append(key)
        return _FakeEvaluator()

    bridge = BBOPlaceLocalBridge(tmp_path, evaluator_factory=factory)
    payload = {
        "benchmark": "adaptec1",
        "placer": "mgo",
        "seed": 7,
        "n_macro": 2,
        "x": [[1.0, 2.0, 3.0, 4.0]],
    }

    first = bridge.evaluate_payload(payload)
    second = bridge.evaluate_payload(payload)

    assert first["status"] == "success"
    assert first["hpwl"] == [10.0]
    assert second["hpwl"] == [10.0]
    assert len(calls) == 1


def test_bboplace_local_bridge_rejects_n_macro_dimension_mismatch(tmp_path: Path) -> None:
    bridge = BBOPlaceLocalBridge(tmp_path, evaluator_factory=lambda *_args: _FakeEvaluator())
    payload = {
        "benchmark": "adaptec1",
        "placer": "mgo",
        "seed": 0,
        "n_macro": 3,
        "x": [[1.0, 2.0, 3.0, 4.0]],
    }

    try:
        bridge.evaluate_payload(payload)
    except ValueError as exc:
        assert "does not match evaluator dimension" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for mismatched n_macro.")


def test_bboplace_local_bridge_accepts_minibench_tuple_result(tmp_path: Path) -> None:
    bridge = BBOPlaceLocalBridge(tmp_path, evaluator_factory=lambda *_args: _FakeMiniBenchEvaluator())
    payload = {
        "benchmark": "adaptec1",
        "placer": "mgo",
        "seed": 7,
        "n_macro": 2,
        "x": [[1.0, 2.0, 3.0, 4.0]],
    }

    result = bridge.evaluate_payload(payload)

    assert result["status"] == "success"
    assert result["hpwl"] == [10.0]


def test_minibench_name_and_placer_resolution(tmp_path: Path) -> None:
    (tmp_path / "benchmarks" / "ispd2005" / "adaptec1").mkdir(parents=True)

    assert _resolve_minibench_benchmark(tmp_path, "adaptec1") == "ispd2005/adaptec1"
    assert _resolve_minibench_benchmark(tmp_path, "ispd2005/adaptec1") == "ispd2005/adaptec1"
    assert _resolve_minibench_placer("mgo") == "gg"
    assert _resolve_minibench_placer("sp") == "sp"
