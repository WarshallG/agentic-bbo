# Evaluation

- **Objective:** maximize `throughput`.
- **Implementation:** `y = model.predict(decode(x))` where `x` is the normalized suggestion vector and `decode` maps `[0,1]^d` to integer/enum indices and floats per knob spec JSON.
