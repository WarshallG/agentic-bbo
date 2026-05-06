# Constraints

- The exposed parameter `SMILES` is categorical and may take values only from the bundled fixed candidate pool shipped with the task code.
- This task is intentionally offline-friendly: it must not require online dataset downloads, online model downloads, or remote services during evaluation.
- The first integration version is a fixed-pool selection task, not the full GuacaMol generator benchmark interface.
- Preserve append-only logs and replay-based resume behavior.
