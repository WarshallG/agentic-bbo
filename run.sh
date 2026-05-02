#!/usr/bin/env bash
set -euo pipefail

: "${NANOBOT_API_KEY:?Set NANOBOT_API_KEY before running this script.}"
export SANDBOX_FUSION_BASE_URL="${SANDBOX_FUSION_BASE_URL:-http://localhost:3211}"
export AGENT_WEB_SEARCH_PROVIDER="${AGENT_WEB_SEARCH_PROVIDER:-mock}"
export AGENT_WEB_SEARCH_API_KEY_ENV="${AGENT_WEB_SEARCH_API_KEY_ENV:-}"

WEB_SEARCH_ARGS=(--agent-web-search-provider "$AGENT_WEB_SEARCH_PROVIDER")
if [ -n "$AGENT_WEB_SEARCH_API_KEY_ENV" ]; then
    WEB_SEARCH_ARGS+=(--agent-web-search-api-key-env "$AGENT_WEB_SEARCH_API_KEY_ENV")
fi

uv run --extra nanobot python -m bbo.run \
    --task branin_demo \
    --algorithm nanobot \
    --seed 7 \
    --max-evaluations 20 \
    --agent-initial-random 2 \
    --agent-provider openai \
    --agent-model deepseek-reasoner \
    --agent-api-base "http://35.220.164.252:3888/v1/" \
    --agent-api-key-env NANOBOT_API_KEY \
    --agent-tool-mode workspace_json \
    --agent-code-backend sandboxfusion \
    --sandbox-fusion-base-url "$SANDBOX_FUSION_BASE_URL" \
    --agent-require-visible-cot \
    --no-agent-allow-fallback \
    --agent-timeout-seconds 240 \
    "${WEB_SEARCH_ARGS[@]}" \
    --results-root runs/nanobot_deepseek_sandboxfusion_full \
    --no-plots
