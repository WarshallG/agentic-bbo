# 提交接口

- 参数：对默认的 `adaptec1` MGO 实例，本任务暴露 64 个浮点参数，对应 32 个 macro：`x_0` ... `x_31`，随后是 `y_0` ... `y_31`。
- 语义：这些值表示 macro 坐标提案，而不是保证最终合法的 placement。evaluator 会先解码，再计算 `hpwl`。
- 本仓库使用的请求格式：

```json
{
  "benchmark": "adaptec1",
  "seed": 1,
  "n_macro": 32,
  "placer": "mgo",
  "x": [[x_0, x_1, "...", x_31, y_0, y_1, "...", y_31]]
}
```

- 期望响应格式：

```json
{
  "hpwl": [123456.0]
}
```

- 本仓库的运行日志：每个 trial 都会 append 到 JSONL 日志中，并记录目标值 `hpwl`、`dimension`、`n_macro` 等标量指标，以及 `coord::x_i` / `coord::y_i` 形式的逐坐标指标。
- 暴露给算法的 task metadata 包括 benchmark 名称、placer、benchmark seed、网格大小、macro 数量以及 evaluator base URL。
- Batch 说明：虽然上游 evaluator 的 payload 里 `x` 是一个候选向量 batch，但当前封装任务每次请求只发送一个候选解。
