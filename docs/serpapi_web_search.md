# SerpAPI Web Search Setup

BBO's agentic `web_search` tool can use SerpAPI directly. This is the
recommended setup for public runs because users only need one API key and do
not need to run a separate search server.

## Configure the API Key

Create a SerpAPI key at https://serpapi.com/ and export it before running BBO:

```bash
export SERPAPI_API_KEY=...
```

The default `run.sh` uses this key automatically.

```bash
./run.sh
```

For direct CLI use:

```bash
uv run --extra nanobot python -m bbo.run \
  --algorithm nanobot \
  --task branin_demo \
  --agent-tool-mode workspace_json \
  --agent-web-search-provider serpapi
```

## Custom Key Environment Variable

If you do not want to name the key `SERPAPI_API_KEY`, pass the environment
variable name explicitly:

```bash
export MY_SERPAPI_KEY=...

uv run --extra nanobot python -m bbo.run \
  --algorithm nanobot \
  --task branin_demo \
  --agent-tool-mode workspace_json \
  --agent-web-search-provider serpapi \
  --agent-web-search-api-key-env MY_SERPAPI_KEY
```

## Offline Smoke Test

Use the mock provider when you want to test the agent plumbing without making
network calls:

```bash
export AGENT_WEB_SEARCH_PROVIDER=mock
./run.sh
```

`mock` returns deterministic placeholder results and still exercises BBO's tool
logging path.

## Outputs

Web-search calls are logged under each run directory:

- `agent_tool_calls.jsonl`: every BBO tool call
- `agent_web_sources.jsonl`: normalized SerpAPI search results with source IDs

The agent can access the same tool in either runtime mode:

- `workspace_json`: `from bbo_tools import BBO; BBO().web_search(...)`
- `function_calling`: native `web_search` tool calls when the endpoint supports tools
