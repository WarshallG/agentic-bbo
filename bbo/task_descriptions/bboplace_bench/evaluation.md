# Evaluation Protocol

- This adapter currently evaluates the macro-placement objective returned by the evaluator service as `hpwl`, which corresponds to an MGO-style macro-placement wirelength signal rather than the full set of metrics discussed in the paper.
- Lower `hpwl` is better. A successful response must contain a non-empty JSON list under the `hpwl` key, and this task uses the first element of that list as the objective value.
- One evaluator request counts as one evaluation in this repo's experiment loop.
- Reproducibility depends on fixing the container/image version, benchmark name, placer, and benchmark seed. The packaged task defaults to benchmark `adaptec1`, placer `mgo`, and `bench_seed = 1`.
- The full BBOPlace-Bench framework also discusses GP-HPWL and downstream PPA metrics, but those metrics are not exposed by the current `bbo/tasks/bboplace/task.py` adapter.
- When reporting results, include at least the benchmark, placer, container tag, task budget, and whether the evaluator service was local or remote.
