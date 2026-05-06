# Background

`guacamol_qed_demo` is the first GuacaMol-oriented task integrated into this repository.
It is intentionally narrower than the original GuacaMol goal-directed benchmark workflow: instead of asking a generator to invent molecules, this demo exposes a fixed repository-local pool of candidate SMILES strings and scores each candidate with the same QED objective used by GuacaMol's `qed_benchmark`.

This simplification is deliberate.
The first integration target is a stable, offline-friendly smoke task that validates scoring, task packaging, logging, and replay behavior inside `agentic-bbo` before any broader generator-facing integration is attempted.
