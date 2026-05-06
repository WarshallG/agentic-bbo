# 目标

**最大化**主目标 `throughput`（即评估服务返回的 TPS/吞吐，字段 `y` 或 `tps`）。

- 搜索域：与 knob JSON 中每个键对应的一维 `[0,1]` 连续坐标，再解码到物理整型/枚举值。
- 一次有效评估 = 一次成功的 `POST /evaluate`，`workload` 固定为 **``read_write``**，服务端映射到 **``oltp_read_write``**。
- 若 `status` 非 `success` 或出现超时，该 trial 记为失败（见 `trials.jsonl`）。

同组对比不同算法时，应固定**镜像、`server.py`、压测参数与 seed**；否则绝对 TPS 不可比。

**Knob 表：**5 个旋钮（`knobs_SYSBENCH_top5.json`，与离线 `knob_surrogate_sysbench_5` 语义一致）。
