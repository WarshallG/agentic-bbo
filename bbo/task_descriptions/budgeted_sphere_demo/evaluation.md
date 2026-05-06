# Evaluation Protocol

- The reported objective is `true_loss + fidelity_gap`.
- `true_loss` is the exact sphere value.
- `fidelity_gap` is deterministic and shrinks monotonically as budget approaches `1.0`.
- Metrics record both `evaluation_budget` and `true_loss` so downstream analysis can distinguish fidelity from the observed objective.
