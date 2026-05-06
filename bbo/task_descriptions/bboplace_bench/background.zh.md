# 背景

BBOPlace-Bench 是一个面向 chip placement 的 black-box optimization benchmark，而 chip placement 会显著影响后续的 routing、timing、power 和 area。
论文将 macro placement 视为一个昂贵的黑盒问题：候选解需要通过 placement 工具链评估，而不是通过闭式目标函数直接计算；即使只使用 proxy metric，在真实设计上也可能有明显开销。

本仓库封装的任务对应 BBOPlace-Bench 中的 mask-guided optimization（MGO）形式，并通过已发布的 evaluator service 暴露出来。
在 MGO 里，优化器先给出二维画布上的连续 macro 坐标提案，随后 evaluator 再通过 wire-mask-guided decoding，把这个提案转换成一个合法的 macro placement，同时尽量减小增量 HPWL。
因此，这个任务更准确地说是一个“带结构的连续黑盒优化问题”，而不是“把每个 macro 直接放到指定位置”的简单 API。

完整的 BBOPlace-Bench 论文研究了多种问题建模、优化算法和评估指标，并覆盖 ISPD 2005、ICCAD 2015 等工业 benchmark。
而本仓库里的 adapter 主要聚焦于 evaluator service 暴露出来的 MGO 风格 macro-placement 目标，适合作为一个现实但易接入的入口，用来比较通用 BBO 算法在芯片设计工作负载上的表现。
