# Background

`budgeted_sphere_demo` is a lightweight synthetic benchmark used to exercise the benchmark stack's explicit budget semantics.
It keeps the same simple sphere geometry as a smoke-test task, but the returned loss is fidelity-dependent: low budgets use a biased approximation, while budget `1.0` returns the true sphere value.
