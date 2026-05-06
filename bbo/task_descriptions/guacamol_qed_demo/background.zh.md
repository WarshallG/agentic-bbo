# 背景

`guacamol_qed_demo` 是这个仓库中集成的第一个面向 GuacaMol 的任务。
它刻意比原始的 GuacaMol 目标导向基准工作流更窄：这个 demo 不是要求生成器发明新分子，而是暴露一个仓库内固定的候选 SMILES 字符串池，并使用与 GuacaMol `qed_benchmark` 相同的 QED 目标对每个候选进行打分。

这种简化是有意为之的。
第一阶段的集成目标是一个稳定、离线友好的 smoke task，用来在 `agentic-bbo` 内部先验证打分、任务封装、日志记录和 replay 行为，然后再尝试更广泛的生成器集成。
