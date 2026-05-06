# Submission Interface

- Optimizers may change only one parameter: `SMILES`.
- The parameter value must be one of the task's bundled candidate-pool entries.
- Each evaluation corresponds to one QED computation for one selected candidate molecule.
- The default evaluation budget is `40`, while smoke tests use smaller budgets such as `3`.
