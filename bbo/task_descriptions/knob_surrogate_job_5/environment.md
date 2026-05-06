# Environment Setup

```bash
uv sync --extra dev --extra surrogate
```

Copy the matching ``.joblib`` into ``bbo/tasks/dbtune/assets/`` (see catalog for filename). Optional: set the env var listed in ``catalog.py`` for `knob_surrogate_job_5`.
