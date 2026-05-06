# Domain Prior Knowledge

- For **5-knob** tasks, the set was chosen for interpretability; `innodb_doublewrite` is an **enum (ON/OFF)**, the others are integers with wide ranges in the JSON.
- For **~197D** tasks, each evaluation is expensive: favor algorithms that make strong use of a **small** trial budget, or do staged dimensionality reduction outside this core if your research allows.
- Workload **``oltp_read_write``** stresses different subsystems: read-heavy vs write-heavy optima need not match.
- Expect **noise** between trials even for repeated configurations unless you add explicit replication in your experiment driver.

These notes are *hints*, not formal invariants of the search space.
