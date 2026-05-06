# Goal

**Maximize** the primary objective `throughput` (larger is better), implemented as the parsed **per-second throughput** in the JSON response field `y` (alias `tps`).

- **Search space:** one float in `[0,1]` per exposed knob, decoded to physical values with the same rules as the surrogate `KnobSpaceFromJson` helper.
- **One evaluation** = one successful `POST /evaluate` with `{"knobs":{...},"workload":"point_select"}` that runs **``oltp_point_select``** under the fixed `server.py` parameters.
- **Valid run:** evaluator `status` is `success` and the returned objective is finite.

Comparative benchmarks should keep **image version, `server.py` timing, and hardware** fixed.
