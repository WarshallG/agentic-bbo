# Goal

Minimize `loss` over a 2D box.
The true objective is the sphere function `x1^2 + x2^2`, but this task intentionally exposes a budget-controlled fidelity level.

- One evaluation counts as one call with a concrete `budget` in `[0.25, 1.0]`.
- Higher budgets are more faithful to the true sphere loss.
- The optimum is attained at the origin with true loss `0.0`.
