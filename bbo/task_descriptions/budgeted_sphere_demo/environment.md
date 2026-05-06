# Environment Setup

This task shares the standard repository environment used by the other synthetic benchmarks.

```bash
uv sync --extra dev
```

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task budgeted_sphere_demo --max-evaluations 3
```
