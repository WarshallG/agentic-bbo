"""Interleave SkyDiscover meta-evolution with BBO ask/tell on dict configs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...core import ObjectiveDirection, SearchSpace, TaskDescriptionBundle, TaskSpec
from ...core.adapters import ExternalOptimizerAdapter
from ...core.trial import TrialObservation, TrialSuggestion
from .generated_solver import load_suggest_next_config, search_space_to_parameter_specs
from .skydiscover_bbo import initial_strategy_template_path, meta_evaluator_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from skydiscover.config import LLMConfig


def _ensure_skydiscover_llm_evaluator_guide(llm: "LLMConfig") -> None:
    """Align evaluator/guide LLM lists with main models after late ``models`` injection.

    ``LLMConfig.__post_init__`` copies empty ``evaluator_models``/``guide_models`` from
    ``models`` when ``load_config`` built the config with ``models=[]``. If we only
    append to ``llm.models`` afterward, the two pools stay empty and
    ``DiscoveryController`` fails when constructing ``LLMPool``.
    """

    if not llm.evaluator_models:
        llm.evaluator_models = llm.models.copy()
    if not llm.guide_models:
        llm.guide_models = llm.models.copy()


def _maybe_apply_bbo_full_rewrite_template_dir(config: Any) -> None:
    """Prefer BBO-tailored full-rewrite templates when diff mode is off and YAML left ``template_dir`` unset."""

    if getattr(config, "diff_based_generation", True):
        return
    if getattr(getattr(config, "search", None), "type", "") == "evox":
        return
    cb = getattr(config, "context_builder", None)
    if cb is None:
        cb = getattr(config, "prompt", None)
    if cb is None or getattr(cb, "template_dir", None):
        return
    tpl_dir = Path(__file__).resolve().parent / "skydiscover_bbo" / "prompt_templates"
    if (tpl_dir / "full_rewrite_user_message.txt").is_file():
        cb.template_dir = str(tpl_dir)


def _apply_skydiscover_search_type(config: Any, search_type: str) -> None:
    """Apply a SkyDiscover search override across supported API shapes."""

    try:
        from skydiscover.config import apply_overrides
    except ImportError:
        apply_overrides = None

    if apply_overrides is not None:
        apply_overrides(config, search=search_type)
        return

    search = getattr(config, "search", None)
    if search is None:
        return

    current_type = getattr(search, "type", None)
    search.type = search_type
    if current_type == search_type:
        return

    try:
        from skydiscover.config import _DB_CONFIG_BY_TYPE
    except ImportError:
        return

    db_cls = _DB_CONFIG_BY_TYPE.get(search_type)
    if db_cls is not None and not isinstance(getattr(search, "database", None), db_cls):
        search.database = db_cls()


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class SkydiscoverInterleavedAlgorithm(ExternalOptimizerAdapter):
    """Use SkyDiscover to evolve ``suggest_next_config`` while optimizing task configs.

    Every ``interleave_every`` completed trials (and once before the first suggestion if
    no strategy file exists), optionally runs a short SkyDiscover round that writes
    ``<run_dir>/generated/strategy.py``. Between rounds, ``ask()`` loads that module
    and calls ``suggest_next_config`` with successful trial history.

    When ``skydiscover_runner_enabled`` is False, refreshes copy the bundled
    ``initial_strategy.py`` instead of invoking the SkyDiscover ``Runner`` (CI / no API key).

    Replay skips SkyDiscover calls and relies on the persisted ``strategy.py`` plus
    deterministic ``(seed, trial_index)`` inside ``suggest_next_config``.
    """

    def __init__(
        self,
        *,
        run_dir: Path | str | None = None,
        interleave_every: int = 5,
        skydiscover_round_iterations: int = 3,
        skydiscover_config_path: str | Path | None = None,
        skydiscover_runner_enabled: bool = False,
        skydiscover_search_type: str = "topk",
        skydiscover_model: str | None = None,
        max_meta_history: int = 32,
    ) -> None:
        super().__init__()
        if interleave_every <= 0:
            raise ValueError("interleave_every must be positive.")
        if skydiscover_round_iterations <= 0:
            raise ValueError("skydiscover_round_iterations must be positive.")
        if max_meta_history <= 0:
            raise ValueError("max_meta_history must be positive.")

        self._run_dir_arg = Path(run_dir) if run_dir is not None else None
        self.interleave_every = interleave_every
        self.skydiscover_round_iterations = skydiscover_round_iterations
        self._skydiscover_config_path = (
            Path(skydiscover_config_path) if skydiscover_config_path else None
        )
        self.skydiscover_runner_enabled = skydiscover_runner_enabled
        self.skydiscover_search_type = skydiscover_search_type
        self.skydiscover_model = skydiscover_model
        self.max_meta_history = max_meta_history

        self._run_dir: Path | None = None
        self._generated_dir: Path | None = None
        self._strategy_path: Path | None = None
        self._meta_context_path: Path | None = None
        self._seed = 0
        self._description: TaskDescriptionBundle = TaskDescriptionBundle.empty(task_id="uninitialized")
        self._parameter_specs: list[dict[str, Any]] = []
        self._observations: list[TrialObservation] = []
        self._trials_since_refresh = 0
        self._replay_mode = False
        self._suggest_fn: Callable[..., dict[str, Any]] | None = None
        self._round_counter = 0

    @property
    def name(self) -> str:
        return "skydiscover_interleaved"

    def setup(self, task_spec: TaskSpec, seed: int = 0, **kwargs: Any) -> None:
        if len(task_spec.objectives) != 1:
            raise ValueError("SkydiscoverInterleavedAlgorithm supports exactly one objective.")
        self.bind_task_spec(task_spec)
        self._seed = int(seed)
        description = kwargs.get("task_description")
        if isinstance(description, TaskDescriptionBundle):
            self._description = description
        else:
            self._description = TaskDescriptionBundle.empty(task_id=task_spec.name)

        run_dir_kw = kwargs.get("run_dir")
        base = self._run_dir_arg or (Path(run_dir_kw) if run_dir_kw else Path.cwd())
        self._run_dir = Path(base).resolve()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._generated_dir = self._run_dir / "generated"
        self._generated_dir.mkdir(parents=True, exist_ok=True)
        self._strategy_path = self._generated_dir / "strategy.py"
        self._meta_context_path = self._generated_dir / "meta_context.json"

        space = task_spec.search_space
        self._parameter_specs = search_space_to_parameter_specs(space)
        self._observations = []
        self._trials_since_refresh = 0
        self._suggest_fn = None
        self._round_counter = 0

        if not self._strategy_path.exists():
            shutil.copy2(initial_strategy_template_path(), self._strategy_path)
        self._load_strategy_module()

    def ask(self) -> TrialSuggestion:
        space = self.require_search_space()
        task_spec = self.require_task_spec()
        # 中文：非 replay 时按交错节奏触发 SkyDiscover，刷新 generated/strategy.py
        if not self._replay_mode:
            self._maybe_refresh_strategy()

        trial_index = len(self._observations)
        raw_config: dict[str, Any] | None = None
        meta: dict[str, Any] = {
            "skydiscover_interleaved": True,
            "trial_index": trial_index,
            "trials_since_refresh": self._trials_since_refresh,
            "runner_enabled": self.skydiscover_runner_enabled,
        }

        fn = self._require_suggest_fn()
        try:
            raw_config = fn(
                history=self._successful_history_tuples(),
                parameter_specs=list(self._parameter_specs),
                objective_direction=task_spec.primary_objective.direction.value,
                seed=self._seed,
                trial_index=trial_index,
            )
        except Exception as exc:
            logger.warning("suggest_next_config failed, falling back to random sample: %s", exc)
            meta["fallback"] = "random_sample"
            meta["error"] = str(exc)
            raw_config = space.sample()

        assert raw_config is not None
        meta["strategy_path"] = str(self._strategy_path) if self._strategy_path else ""

        return TrialSuggestion(config=dict(raw_config), metadata=meta)

    def tell(self, observation: TrialObservation) -> None:
        self._observations.append(observation)
        self.update_best_incumbent(observation)
        if not self._replay_mode:
            self._trials_since_refresh += 1

    def replay(self, history: list[TrialObservation]) -> None:
        if self._strategy_path is None:
            raise RuntimeError("SkydiscoverInterleavedAlgorithm.setup() must run before replay().")
        self._replay_mode = True
        try:
            self._load_strategy_module()
            for observation in history:
                expected = self.ask()
                self.assert_matching_config(expected.config, observation.suggestion.config)
                self.tell(self.make_replay_observation(expected, observation))
        finally:
            self._replay_mode = False

    def _successful_history_tuples(self) -> list[tuple[dict[str, Any], float]]:
        primary = self._primary_name
        assert primary is not None
        out: list[tuple[dict[str, Any], float]] = []
        for obs in self._observations:
            if not obs.success or primary not in obs.objectives:
                continue
            out.append((dict(obs.suggestion.config), float(obs.objectives[primary])))
        return out[-self.max_meta_history :]

    def _maybe_refresh_strategy(self) -> None:
        # 中文：在线交错——每 interleave_every 次 tell 后（或缺失策略文件）刷新内层求解器
        assert self._strategy_path is not None
        assert self._generated_dir is not None
        assert self._meta_context_path is not None

        need = self._trials_since_refresh >= self.interleave_every or not self._strategy_path.exists()
        if not need:
            return

        self._write_meta_context()
        if self.skydiscover_runner_enabled:
            try:
                asyncio.run(self._run_skydiscover_round())
            except Exception as exc:
                logger.warning("SkyDiscover round failed; refreshing template: %s", exc)
                shutil.copy2(initial_strategy_template_path(), self._strategy_path)
        else:
            shutil.copy2(initial_strategy_template_path(), self._strategy_path)

        self._load_strategy_module()
        self._trials_since_refresh = 0

    def _write_meta_context(self) -> None:
        assert self._meta_context_path is not None
        task_spec = self.require_task_spec()
        meta = task_spec.metadata
        problem_key = str(meta.get("problem_key", task_spec.name))
        payload: dict[str, object] = {
            "parameter_specs": self._parameter_specs,
            "objective_direction": task_spec.primary_objective.direction.value,
            "recent_history": [
                [cfg, score] for cfg, score in self._successful_history_tuples()
            ],
            "seed": self._seed,
            "task_name": task_spec.name,
            "problem_key": problem_key,
            "description_fingerprint": self._description.fingerprint,
        }
        # 中文：合成 BBO 任务在 metadata 中带有 known_optima，启用内层与主任务一致的到最优的距离评分
        known_optima = meta.get("known_optima")
        if known_optima:
            payload["meta_combined_score_mode"] = "distance_to_known_optimum"
            payload["known_optima"] = known_optima
        else:
            payload["meta_combined_score_mode"] = "contract"
        self._meta_context_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    async def _run_skydiscover_round(self) -> None:
        try:
            import skydiscover
            from skydiscover import config as skydiscover_config
        except ImportError as exc:
            raise RuntimeError(
                "SkyDiscover is not installed. Install optional extra `skydiscover` or set "
                "skydiscover_runner_enabled=False."
            ) from exc

        assert self._generated_dir is not None
        assert self._strategy_path is not None

        cfg_path = str(self._skydiscover_config_path) if self._skydiscover_config_path else None
        config = skydiscover_config.load_config(cfg_path)
        _apply_skydiscover_search_type(config, self.skydiscover_search_type)
        # 中文：YAML 可能未传 / 默认可仍为 diff；整文件解析显著更稳（仅当显式允许环境变量时才保留 diff）
        if os.environ.get("BBO_ALLOW_SKYDISCOVER_DIFF_PATCHES", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        ):
            config.diff_based_generation = False
        _maybe_apply_bbo_full_rewrite_template_dir(config)
        monitor = getattr(config, "monitor", None)
        if monitor is not None and hasattr(monitor, "enabled"):
            monitor.enabled = False
        config.max_parallel_iterations = 1
        if not config.llm.models:
            model_name = self.skydiscover_model or "gpt-4o-mini"
            config.llm.models = [skydiscover_config.LLMModelConfig(name=model_name)]

        # 中文：仅补全 models 时，需同步 evaluator/guide，否则 LLMPool 收到空列表
        _ensure_skydiscover_llm_evaluator_guide(config.llm)
        bridge_env = getattr(skydiscover_config, "bridge_provider_env", None)
        if callable(bridge_env):
            bridge_env(config)

        self._round_counter += 1
        round_dir = self._generated_dir / f"skydiscover_round_{self._round_counter:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        initial = (
            str(self._strategy_path)
            if self._strategy_path.exists()
            else str(initial_strategy_template_path())
        )
        env = {"BBO_SKYDISCOVER_META_CONTEXT": str(self._meta_context_path)}

        best_solution: str | None = None
        runner_cls = getattr(skydiscover, "Runner", None)
        if runner_cls is not None:
            runner = runner_cls(
                evaluation_file=str(meta_evaluator_path()),
                initial_program_path=initial,
                config=config,
                output_dir=str(round_dir),
                evaluator_env_vars=env,
            )
            result = await runner.run(iterations=self.skydiscover_round_iterations)
            best_solution = getattr(result, "best_solution", None)
        else:
            controller_cls = getattr(skydiscover, "MainController", None)
            if controller_cls is None:
                raise RuntimeError("Installed SkyDiscover exposes neither Runner nor MainController.")
            previous_env = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                controller = controller_cls(
                    initial_program_path=initial,
                    evaluation_file=str(meta_evaluator_path()),
                    config=config,
                    output_dir=str(round_dir),
                )
                best_program = await controller.run(iterations=self.skydiscover_round_iterations)
                best_solution = getattr(best_program, "solution", None)
            finally:
                _restore_env(previous_env)

        best_py = round_dir / "best" / "best_program.py"
        if best_py.is_file():
            shutil.copy2(best_py, self._strategy_path)
        elif best_solution:
            self._strategy_path.write_text(best_solution, encoding="utf-8")
        else:
            logger.warning(
                "SkyDiscover produced no best_program.py; reverting to bundled seed strategy."
            )
            shutil.copy2(initial_strategy_template_path(), self._strategy_path)

    def _load_strategy_module(self) -> None:
        if self._strategy_path is None or not self._strategy_path.is_file():
            self._suggest_fn = None
            return
        self._suggest_fn = load_suggest_next_config(self._strategy_path)

    def _require_suggest_fn(self) -> Callable[..., dict[str, Any]]:
        if self._suggest_fn is None:
            self._load_strategy_module()
        if self._suggest_fn is None:
            raise RuntimeError("No strategy module loaded; call setup() first.")
        return self._suggest_fn
