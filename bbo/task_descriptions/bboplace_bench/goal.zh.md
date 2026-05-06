# 目标

- 主指标：最小化 evaluator 返回的 `hpwl`，数值越小越好。
- 当前封装：本仓库中的 task 对应 BBOPlace-Bench 的 MGO 形式，即搜索向量先给出 macro 坐标提案，再由 evaluator 解码成合法 placement。
- 搜索向量：长度为 `2 * n_macro` 的连续向量。`x_0` 到 `x_{n_macro-1}` 表示横坐标，随后 `y_0` 到 `y_{n_macro-1}` 表示纵坐标。
- 默认实例：当前代码里的定义默认使用 benchmark `adaptec1`、placer `mgo`、`n_macro = 32`、网格大小 `224 x 224`；如果需要可在代码里覆盖这些定义。
- 评估计数：每次向 evaluator 发起一次 POST 请求，记作一次 evaluation。本仓库里的默认 budget 是 40 次，除非调用方显式覆盖 `max_evaluations`。
- 最优值：把该任务视为未知最优解的黑盒问题。论文表明 MGO 表现很强，但这里不假设某个芯片 case 的全局最优值已知。
