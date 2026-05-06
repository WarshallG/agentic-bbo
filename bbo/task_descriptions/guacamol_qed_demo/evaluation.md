# Evaluation Protocol

- Data source: a repository-local fixed pool of candidate SMILES strings defined inside the task implementation.
- For each evaluation, the task parses the proposed SMILES with RDKit and computes `Descriptors.qed`, matching the objective used by GuacaMol's `qed_benchmark`.
- The task reports:
  - primary objective: `guacamol_qed_loss = 1.0 - guacamol_qed_score`
  - metrics: raw `guacamol_qed_score` and `qed`
  - metadata: selected `smiles`, validity flag, and candidate-pool size
- The evaluator is deterministic on the task side and does not depend on the benchmark seed.
