# 目标

最小化主目标 `guacamol_qed_loss`，其中：

- `guacamol_qed_loss = 1.0 - guacamol_qed_score`
- `guacamol_qed_score` 是用与 GuacaMol `qed_benchmark` 相同描述符选择计算得到的 RDKit QED 值

因此，更低的 loss 等价于从固定候选池中选出 GuacaMol 风格 QED 分数更高的分子。
