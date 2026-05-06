# Environment Setup

This task is designed to run inside the repository's managed Python environment without online downloads.
The recommended setup is:

```bash
uv sync --extra dev --extra bo-tutorial
```

This is an already-validated environment, not necessarily the only viable one.
The hard requirement for this task is a working RDKit installation compatible with the repository's Python environment.

Minimal smoke test:

```bash
uv run python -m bbo.run --algorithm random_search --task guacamol_qed_demo --max-evaluations 3
```

The local sibling `guacamol/` checkout is used only as a reference implementation during development.
This task does not require the external repository to be importable at runtime.
