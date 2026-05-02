# SandboxFusion for BBO Agents

This guide configures the BBO `code_interpreter` tool for agentic BBO runs.
BBO does not implement a local security sandbox. Real code execution should go
through a SandboxFusion-compatible `/run_code` service.

## 1. Start SandboxFusion

Official local Docker deployment:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

Mainland China mirror:

```bash
docker run -it -p 8080:8080 \
  vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

The service exposes:

```text
POST http://localhost:8080/run_code
```

with request body:

```json
{"code": "print('Hello, world!')", "language": "python"}
```

## 2. Preset Python Packages

For the default BBO agent workflow, the SandboxFusion Python runtime should
include:

```text
numpy
scipy
scikit-learn
pandas
joblib
```

These are enough for the workspace GP example and lightweight analysis scripts.
The default `agent_workspace/gp_expected_improvement.py` requires `numpy` and
`scikit-learn`; `scipy`, `pandas`, and `joblib` are recommended for common BBO
analysis tasks and future examples.

Optional heavy packages:

```text
torch
gpytorch
botorch
```

Do not make these heavy packages mandatory for the default BBO agent image.
They are useful for a separate "full BO" image, but they increase image size and
create version conflicts with lightweight benchmark environments.

## 3. Point BBO at SandboxFusion

Either export the base URL:

```bash
export SANDBOX_FUSION_BASE_URL="http://localhost:8080"
```

or pass it through the runner:

```bash
uv run python -m bbo.run \
  --task branin_demo \
  --algorithm nanobot \
  --agent-code-backend sandboxfusion \
  --sandbox-fusion-base-url "http://localhost:8080"
```

If no base URL is configured, BBO returns a disabled `code_interpreter` result
instead of executing arbitrary local code.

## 4. Health Check

Run the BBO healthcheck before a real agent experiment:

```bash
uv run python -m bbo.algorithms.agentic.sandbox_healthcheck \
  --base-url "$SANDBOX_FUSION_BASE_URL"
```

The healthcheck calls `/run_code` three times:

```text
python_basic  -> print(2 + 2)
numpy         -> import numpy as np
sklearn_gp    -> fit sklearn.gaussian_process.GaussianProcessRegressor
```

The endpoint is ready for the default BBO agent workflow only when all checks
return `ok: true`.

## 5. Real Nanobot Example

This keeps the `run.sh` LLM proxy style and enables SandboxFusion for code:

```bash
export NANOBOT_API_KEY="..."
export SANDBOX_FUSION_BASE_URL="http://localhost:8080"

uv run --extra nanobot python -m bbo.run \
  --task branin_demo \
  --algorithm nanobot \
  --seed 7 \
  --max-evaluations 8 \
  --agent-initial-random 2 \
  --agent-provider openai \
  --agent-model deepseek-reasoner \
  --agent-api-base "http://35.220.164.252:3888/v1/" \
  --agent-api-key-env NANOBOT_API_KEY \
  --agent-tool-mode workspace_json \
  --agent-code-backend sandboxfusion \
  --sandbox-fusion-base-url "$SANDBOX_FUSION_BASE_URL" \
  --agent-web-search-provider mock \
  --agent-require-visible-cot \
  --no-agent-allow-fallback \
  --results-root runs/nanobot_deepseek_reasoner_real \
  --no-plots
```

Expected artifacts include:

```text
agent_tool_calls.jsonl
agent_optimization_trace.jsonl
llm_logs/
reasoning_traces/
agent_reasoning_metadata.jsonl
```

`agent_tool_calls.jsonl` should show successful `code_interpreter` calls with
`backend: sandboxfusion` when the agent chooses to run sandboxed analysis code.

## 6. References

- SandboxFusion get-started docs: https://bytedance.github.io/SandboxFusion/docs/docs/get-started/
- SandboxFusion `/run_code` API: https://bytedance.github.io/SandboxFusion/zh-Hans/docs/api/run-code-run-code-post/
