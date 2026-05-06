# Constraints

- Coordinate bounds: the packaged `SearchSpace` enforces `0 <= x_i <= n_grid_x` and `0 <= y_i <= n_grid_y` for every macro coordinate. Conceptually, valid canvas locations are on the placement grid `[0, n_grid_x) x [0, n_grid_y)`.
- Legality is enforced by decoding, not by the raw vector alone: under MGO, the evaluator may adjust macro positions through wire-mask-guided decoding before computing the objective.
- Overlap and boundary handling: candidate grids that would overlap existing macros or exceed the canvas boundary are excluded by the decoder, so the final evaluated placement is expected to be legal even if the raw proposal is poor.
- Expensive black-box feedback: evaluation may require noticeable service time. Avoid unnecessary duplicate queries, and expect much slower turnaround than lightweight synthetic tasks.
- Stable interface assumption: this repo expects an evaluator endpoint at `/evaluate` that accepts the documented JSON payload and returns a non-empty `hpwl` list. Endpoint/schema mismatches are treated as task failures.
- No closed-form shortcuts: do not assume gradients, convexity, or a known optimum. Use only the observed evaluator feedback.
