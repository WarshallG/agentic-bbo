# Goal

Minimize the primary objective `guacamol_qed_loss`, where:

- `guacamol_qed_loss = 1.0 - guacamol_qed_score`
- `guacamol_qed_score` is the RDKit QED value computed with the same descriptor choice used by GuacaMol's `qed_benchmark`

Lower loss is therefore equivalent to selecting molecules with higher GuacaMol-style QED scores from the fixed candidate pool.
