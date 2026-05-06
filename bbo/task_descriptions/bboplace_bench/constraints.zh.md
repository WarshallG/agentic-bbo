# 约束

- 坐标边界：封装后的 `SearchSpace` 要求每个 macro 都满足 `0 <= x_i <= n_grid_x` 且 `0 <= y_i <= n_grid_y`。从 placement 语义上说，合法画布位置位于 `[0, n_grid_x) x [0, n_grid_y)`。
- 合法性不是由原始向量单独决定的：在 MGO 中，evaluator 会先对坐标提案做 wire-mask-guided decoding，再计算目标值。
- Overlap 与越界处理：会导致 overlap 或超出画布边界的候选网格会被 decoder 排除，因此即使原始提案较差，最终被评估的 placement 仍应保持合法。
- 反馈昂贵：evaluation 可能需要明显时间，不能把它当作 synthetic task 那样频繁随意调用；应避免无意义的重复查询。
- 接口稳定性：本仓库假设存在 `/evaluate` evaluator endpoint，能接收文档里描述的 JSON payload，并返回非空的 `hpwl` 列表。若 schema 不匹配，会被视为 task failure。
- 不要假设闭式结构：不要假设存在梯度、凸性或已知最优值，只能依赖 evaluator 返回的反馈。
