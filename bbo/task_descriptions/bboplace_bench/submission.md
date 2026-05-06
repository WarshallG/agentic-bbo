# Submission Interface

- Parameters: the task exposes 64 floating-point parameters by default for the `adaptec1` MGO instance with 32 macros: `x_0` ... `x_31`, followed by `y_0` ... `y_31`.
- Semantics: these values are proposed macro coordinates, not guaranteed final legal positions. The evaluator decodes them into a placement before computing `hpwl`.
- Request contract used by this repo:

```json
{
  "benchmark": "adaptec1",
  "seed": 1,
  "n_macro": 32,
  "placer": "mgo",
  "x": [[x_0, x_1, "...", x_31, y_0, y_1, "...", y_31]]
}
```

- Expected response shape:

```json
{
  "hpwl": [123456.0]
}
```

- Runtime logging in this repo: each trial is appended to JSONL logs and includes the objective `hpwl`, scalar metrics such as `dimension` and `n_macro`, and per-coordinate metrics stored as `coord::x_i` / `coord::y_i`.
- Task metadata exposed to algorithms includes the benchmark name, placer, benchmark seed, grid size, macro count, and evaluator base URL.
- Batch note: the upstream evaluator payload uses `x` as a batch of candidate vectors, but the packaged task currently sends one candidate per request.
