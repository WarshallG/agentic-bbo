# Constraints

- Search domain is the **unit hypercube** `[0, 1]^d` with one coordinate per tuned knob, in fixed order matching the surrogate `X-name`.
- Evaluations are **noise-free** with respect to the bundled surrogate (deterministic RF).
- Replacing the bundled small RF with a larger externally trained model is allowed if `X-name` and knob JSON stay consistent.
