# 评估协议

- 数据来源：任务实现内部定义的、仓库本地固定候选 SMILES 池。
- 对每次评估，任务都会先用 RDKit 解析提交的 SMILES，然后计算 `Descriptors.qed`，与 GuacaMol `qed_benchmark` 使用的目标保持一致。
- 任务会报告：
  - 主目标：`guacamol_qed_loss = 1.0 - guacamol_qed_score`
  - 指标：原始 `guacamol_qed_score` 和 `qed`
  - 元数据：所选 `smiles`、有效性标记，以及候选池大小
- 评估器在任务侧是确定性的，不依赖 benchmark seed。
