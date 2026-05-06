# 评估协议

- 当前 adapter 评估的是 evaluator service 返回的 `hpwl`，它对应 MGO 风格的 macro-placement wirelength 信号，而不是论文中完整讨论的全部指标。
- `hpwl` 越小越好。一次成功的响应必须在 `hpwl` 键下返回一个非空 JSON 列表，本任务使用其中第一个元素作为目标值。
- 在本仓库的 experiment loop 中，一次评估请求记作一次 evaluation。
- 可复现性依赖于固定 container / image 版本、benchmark 名称、placer 和 benchmark seed。当前打包任务默认使用 benchmark `adaptec1`、placer `mgo`、`bench_seed = 1`。
- 完整的 BBOPlace-Bench 还讨论了 GP-HPWL 和下游 PPA 指标，但这些指标并没有在当前 `bbo/tasks/bboplace/task.py` adapter 中暴露出来。
- 报告结果时，至少应记录 benchmark、placer、container tag、evaluation budget，以及 evaluator service 是本地还是远端。
