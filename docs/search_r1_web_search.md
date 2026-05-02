# Search-R1 Web Search Setup

This guide configures BBO's `web_search` tool to call a Search-R1-compatible
retrieval server. BBO does not start or manage Search-R1; it only sends
`POST /retrieve` requests to a server that is already running.

Reference implementation and setup examples:

- Search-R1 repository: https://github.com/PeterGriffinJin/Search-R1
- Search-R1 retriever guide: https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md

## BBO Configuration

Set the BBO provider and the Search-R1 server URL:

```bash
export AGENT_WEB_SEARCH_PROVIDER=search_r1
export AGENT_SEARCH_R1_BASE_URL=http://127.0.0.1:8000
```

Then run the normal agent script:

```bash
./run.sh
```

Or pass the same settings directly to `bbo.run`:

```bash
uv run --extra nanobot python -m bbo.run \
  --algorithm nanobot \
  --task branin_demo \
  --agent-tool-mode workspace_json \
  --agent-web-search-provider search_r1 \
  --agent-search-r1-base-url http://127.0.0.1:8000
```

The base URL may point either to the server root (`http://127.0.0.1:8000`) or
directly to the endpoint (`http://127.0.0.1:8000/retrieve`).

## Verify the Search-R1 Server

Before running a benchmark, check that the server responds:

```bash
curl -sS -X POST "$AGENT_SEARCH_R1_BASE_URL/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"queries":["branin optimization prior"],"topk":2,"return_scores":true}'
```

BBO expects a Search-R1-style response:

```json
{
  "result": [
    [
      {
        "document": {
          "contents": "\"Title\"\nSnippet or passage text"
        },
        "score": 1.0
      }
    ]
  ]
}
```

Results are normalized into BBO's `title`, `url`, `snippet`, and optional
`score` fields, then logged to `agent_web_sources.jsonl`.

## Option 1: Local Search-R1 Retriever

Use this path for private, domain-specific, or offline corpora. No online
search API key is required, but you must prepare a Search-R1 corpus and index.

Example BM25 launch, adapted from Search-R1's retriever guide:

```bash
cd /path/to/Search-R1

save_path=/your/path/to/wiki18_bm25
huggingface-cli download PeterJinGo/wiki-18-bm25-index \
  --repo-type dataset \
  --local-dir "$save_path"

python search_r1/search/retrieval_server.py \
  --index_path "$save_path/bm25" \
  --corpus_path "$save_path/wiki-18.jsonl" \
  --topk 5 \
  --retriever_name bm25
```

Dense e5 retrievers use the same `/retrieve` API. Follow Search-R1's guide for
the flat GPU or HNSW64 CPU index, then set BBO's `AGENT_SEARCH_R1_BASE_URL` to
that server.

## Option 2: Search-R1 SerpAPI Server

Use this path when you want online web search through SerpAPI. You need a
SerpAPI account and API key.

```bash
export SERPAPI_API_KEY=...

cd /path/to/Search-R1
python search_r1/search/serp_search_server.py \
  --search_url https://serpapi.com/search \
  --topk 5 \
  --serp_api_key "$SERPAPI_API_KEY"
```

BBO-side configuration:

```bash
export AGENT_WEB_SEARCH_PROVIDER=search_r1
export AGENT_SEARCH_R1_BASE_URL=http://127.0.0.1:8000
```

## Option 3: Search-R1 Google CSE Server

Use this path when you prefer Google Custom Search. You need a Google Custom
Search API key and CSE ID.

```bash
export GOOGLE_SEARCH_API_KEY=...
export GOOGLE_CSE_ID=...

cd /path/to/Search-R1
python search_r1/search/google_search_server.py \
  --api_key "$GOOGLE_SEARCH_API_KEY" \
  --cse_id "$GOOGLE_CSE_ID" \
  --topk 5 \
  --snippet_only
```

BBO-side configuration is the same:

```bash
export AGENT_WEB_SEARCH_PROVIDER=search_r1
export AGENT_SEARCH_R1_BASE_URL=http://127.0.0.1:8000
```

## Notes

- `nanobot` with `--agent-tool-mode workspace_json` calls this through
  `BBO().web_search(...)` or `bbo_tool.py web_search ...`.
- `agentic_openai_compatible` with `--agent-tool-mode function_calling` exposes
  the same provider as a native function-calling tool.
- If you choose Search-R1 local retrieval, BBO does not need SerpAPI or Google
  credentials.
- If the Search-R1 server is remote, make sure the benchmark host can reach it
  and that the server is not exposed publicly without access controls.
