# Goal

- Primary metric: minimize the evaluator's returned `hpwl` value; lower is better.
- Current formulation: the packaged task targets the MGO formulation from BBOPlace-Bench, where the search vector proposes macro coordinates and the evaluator decodes them into a legal placement.
- Search vector: a continuous vector of length `2 * n_macro`. Entries `x_0` through `x_{n_macro-1}` are horizontal coordinates, followed by `y_0` through `y_{n_macro-1}` for vertical coordinates.
- Default instance: the task definition in this repo uses benchmark `adaptec1`, placer `mgo`, `n_macro = 32`, and a `224 x 224` grid unless you override the definition in code.
- Evaluation counting: one POST to the evaluator counts as one task evaluation. The default task budget in this repo is 40 evaluations unless the caller overrides `max_evaluations`.
- Optimum: treat the objective as unknown and black-box. The paper reports competitive performance for MGO, but the true global optimum for a given chip case is not assumed to be known.
