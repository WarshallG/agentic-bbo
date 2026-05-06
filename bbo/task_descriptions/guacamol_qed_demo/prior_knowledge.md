# Domain Prior Knowledge

- For valid molecules, GuacaMol-style QED scores are bounded in `[0, 1]`.
- This task converts that score to a minimization objective by reporting `1.0 - score`.
- The bundled pool is curated to contain valid candidates, but if an invalid molecule ever appears, the implementation follows GuacaMol's corrupt-score convention and treats it as `-1.0`, which corresponds to a loss of `2.0`.
- Because the candidate pool is fixed and local, search quality depends entirely on which pool members are proposed, not on generative chemistry capabilities.
