# Background

BBOPlace-Bench is a benchmark for black-box optimization on chip placement, a stage of VLSI design that strongly affects routing, timing, power, and area.
The paper frames macro placement as an expensive black-box problem: a candidate placement is evaluated through placement tooling rather than a closed-form objective, and even proxy metrics can be costly on realistic designs.

The task packaged in this repository wraps the benchmark's mask-guided optimization (MGO) formulation behind a published evaluator service.
In MGO, the optimizer proposes continuous macro coordinates on a 2D canvas, and the evaluator applies wire-mask-guided decoding to convert that proposal into a legal macro placement while trying to keep the incremental HPWL small.
This makes the task a structured continuous black-box problem rather than a direct "place every macro exactly here" API.

The full BBOPlace-Bench paper studies multiple formulations, algorithms, and evaluation metrics on industrial suites such as ISPD 2005 and ICCAD 2015.
The adapter in this repo focuses on the MGO-style macro-placement objective exposed by the evaluator service, which is a practical entry point for comparing generic BBO algorithms on a realistic chip-design workload.
