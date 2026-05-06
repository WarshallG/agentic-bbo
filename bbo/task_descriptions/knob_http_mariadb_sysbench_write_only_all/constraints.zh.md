# 约束

- 同一评估器实例上**不要并发**发多个重评估；服务端已加锁，但仍建议工作流上串行化。
- 单次评估时间可达数分钟，客户端超时（默认 300s）可能不够，应按需调大。
- 本 `task_id` 下 **sysbench 子命令固定为 ``oltp_write_only``**，搜索空间**不可**在运行时改 workload。
- 旋钮表默认来自 `bbo/tasks/dbtune/assets/knobs_mysql_all_197.json`；`bbo.run` 默认不暴露自定义 JSON 路径。
- 本基准面向**实验/调参**环境，请勿连接生产库。

与 `constraints.md` 配套；以英文为权威说明。
