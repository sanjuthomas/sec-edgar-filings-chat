# Chat with SEC Filings

Conversational RAG web app for SEC EDGAR filings. Ask a natural-language question, retrieve matching chunks from **pgvector** or **Qdrant**, and get a cited answer from a local **Ollama** LLM.

![SEC EDGAR Semantic Search — cited answer for a Goldman Sachs 10-K question](docs/chat-screenshot.png)

*Example: ask about Goldman Sachs Q1 results with ticker `GS` and form `10-K`; the assistant returns a cited summary with expandable source cards (25 chunks from pgvector, `qwen3:30b`).*

## Features

- **Multi-turn chat** — message history, per-turn source cards, **New conversation** reset
- **Dual vector stores** — pgvector (psycopg) or Qdrant (REST); selectable in the UI
- **Configurable chunk count** — presets 10 / 25 / 50 / 100 or any value from 1–500
- **Cited RAG answers** — inline `[1]`, `[2]`, … citations with SEC EDGAR links
- **Hybrid search** — pgvector: vector + ParadeDB BM25 RRF; Qdrant: dense + sparse BM25 RRF (when enabled)
- **Search-in-progress UX** — submit button disables and shows “Searching…” until reload
- **Turn metadata** — retrieval/generation timing, vector store, model, source count

---

## Stack

| Layer | Technology |
|-------|------------|
| UI | FastAPI + Jinja2 |
| Retrieval | PostgreSQL + pgvector **or** Qdrant REST |
| Query embeddings | Ollama `bge-m3` (1024-dim) |
| Answer generation | Ollama HTTP API (user-selectable model; default `qwen3:30b`) |

> **Embedding model:** Indexes were built with `BAAI/bge-m3` (1024 dimensions). Query embeddings **must** use the same model (`ollama pull bge-m3`). The older `bge-small-en-v1.5` (384-dim) index is **not** compatible.

---

## Prerequisites

- Python **3.11+**
- **Ollama** on `localhost:11434` with chat models and **`bge-m3`** for query embeddings
- **PostgreSQL + pgvector** — [sec-edgar-filings-to-pgvector](https://github.com/sanjuthomas/sec-edgar-filings-to-pgvector) Docker (`edgar-pgvector`) on `localhost:5433`, database `edgar`
- **Qdrant** — [sec-edgar-filings-to-qdrant](https://github.com/sanjuthomas/sec-edgar-filings-to-qdrant) Docker (`edgar-qdrant`) on `localhost:6333`

Indexed data must exist before searching — run an ingest project first.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

ollama pull bge-m3
ollama list

uvicorn app.main:app --host 0.0.0.0 --port 8095 --reload
```

Open **http://localhost:8095**

Example questions:

> Do you know if the Adobe board approved a buyback program?

> Who are the elected directors in Goldman Sachs?

Optional filters: ticker (`GS`), form (`10-K`).

---

## Docker

Ollama must run on the **host** (`ollama pull bge-m3` plus a chat model). Vector stores run in the **ingest project** Docker stacks above.

```bash
docker compose up --build
```

Open **http://localhost:8095**

The chat container reaches `edgar-pgvector` and `edgar-qdrant` on the host via `host.docker.internal` (ports `5433` / `6333`).

### Manual image run

```bash
docker build -t sec-edgar-filings-chat:local .
docker run --rm -p 8095:8095 \
  -e PG_HOST=host.docker.internal \
  -e PG_PORT=5433 \
  -e PGPASSWORD=postgres \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  --add-host=host.docker.internal:host-gateway \
  sec-edgar-filings-chat:local
```

For crawler, ingest, and a full demo stack, use [sec-edgar-filings-rag-demo](https://github.com/sanjuthomas/sec-edgar-filings-rag-demo).

---

## Configuration

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | `8095` | HTTP port |
| `PGUSER` / `PGPASSWORD` | `postgres` / `postgres` | PostgreSQL credentials |
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5433` | PostgreSQL port |
| `PG_DATABASE` | `edgar` | Database name |
| `DATABASE_URL` | _(built from above)_ | Optional full connection string override |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_CHAT_MODEL` | `qwen3:30b` | Default chat model on page load |
| `OLLAMA_EMBEDDING_MODEL` | `bge-m3` | Query embedding model |
| `SEARCH_TOP_K` | `25` | Default chunk count on page load |
| `EMBEDDING_DIMENSIONS` | `1024` | Expected embedding size |
| `PGSEARCH_ENABLED` | `true` | pgvector hybrid search (vector + BM25 RRF) |
| `QDRANTSEARCH_ENABLED` | `true` | Qdrant hybrid search (dense + BM25 RRF) |
| `QDRANT_DENSE_VECTOR` | `dense` | Qdrant named vector for dense search |
| `QDRANT_BM25_VECTOR` | `content-bm25` | Qdrant sparse vector for BM25 search |
| `QDRANT_BM25_MODEL` | `Qdrant/bm25` | Qdrant document query model for BM25 leg |
| `HYBRID_RETRIEVAL_TOP_K` | `50` | Candidates per leg before hybrid fusion |
| `DEFAULT_VECTOR_STORE` | `pgvector` | Default store (`pgvector` or `qdrant`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST (`edgar-qdrant` from ingest compose) |
| `QDRANT_COLLECTION` | `filing_chunks` | Qdrant collection name |
| `SESSION_SECRET_KEY` | `dev-only-change-in-production` | Signs the browser session cookie |
| `CONVERSATION_MAX_TURNS` | `40` | Max user+assistant turns kept in memory |

---

## How it works

```mermaid
sequenceDiagram
    participant Browser
    participant Routes as search.py
    participant Conv as ConversationStore
    participant RAG as RagSearchService
    participant Embed as Ollama bge-m3
    participant VStore as pgvector or Qdrant
    participant LLM as Ollama
    participant View as Jinja2

    Browser->>Routes: POST /chat (message, model, store, chunkCount)
    Routes->>Conv: load or create conversation
    Routes->>RAG: continue_conversation(message)
    RAG->>Embed: embed retrieval query
    Embed-->>RAG: 1024-dim query vector
    RAG->>VStore: top-K similarity search (+ optional filters)
    VStore-->>RAG: filing chunks + metadata
    RAG->>LLM: prior turns + question + source excerpts
    LLM-->>RAG: answer with [1][2] citations
    RAG-->>Routes: updated conversation
    Routes->>Conv: save conversation
    Routes->>View: render index.html
    View-->>Browser: HTML (thread + source cards)
```

1. Browser submits a message via POST to `/chat` with model, vector store, chunk count, and optional filters.
2. `search.py` loads or creates a `Conversation` from the signed session cookie.
3. `RagSearchService` orchestrates the full RAG pipeline on each turn — chunks are **not** sent to the browser until the LLM finishes.
4. Ollama embeds the retrieval query with `bge-m3` (short follow-ups are expanded with the prior user message).
5. Hybrid retrieval when enabled: **pgvector** runs cosine + ParadeDB BM25; **Qdrant** runs dense + sparse BM25; both fuse with RRF (`HYBRID_RETRIEVAL_TOP_K` per leg, user `chunk_count` for final top-N).
6. Otherwise `ChunkSearchRouter` queries the selected store with vector search only.
7. Top-K chunks plus prior chat turns are passed to Ollama with a system prompt requiring inline citations.
8. Jinja2 renders the full thread with source cards and SEC EDGAR links.

Use **New conversation** (`POST /chat/new`) to clear context.

---

## Project layout

```
app/
├── main.py                 # App factory, DI wiring, lifespan (ticker metadata load)
├── config.py               # pydantic-settings
├── models.py               # Pydantic DTOs (ChatForm, ChunkMatch, Conversation, …)
├── routes/search.py        # GET /, POST /chat, POST /chat/new
├── repositories/           # pgvector, Qdrant, filing metadata
├── services/
│   ├── rag_search.py       # RAG orchestration (retrieve → generate)
│   ├── ollama_client.py    # embed + chat HTTP client
│   ├── chunk_search_router.py
│   ├── pg_hybrid_search.py # optional BM25 + vector fusion
│   ├── ticker_resolver.py
│   └── conversation_store.py
├── templates/index.html
└── static/
tests/                      # pytest unit tests
```

---

## Database requirements

When using **pgvector**, the app expects the schema from [sec-edgar-filings-to-pgvector](https://github.com/sanjuthomas/sec-edgar-filings-to-pgvector):

- **`filings`** — one row per accession
- **`filing_chunks`** — embedded text chunks with `vector(1024)` and HNSW index

When using **Qdrant**, the `filing_chunks` collection must exist (created by [sec-edgar-filings-to-qdrant](https://github.com/sanjuthomas/sec-edgar-filings-to-qdrant)).

```bash
psql postgresql://localhost:5433/edgar -c "SELECT COUNT(*) FROM filing_chunks;"
```

This app does **not** create tables or collections — ingest projects own the schema.

---

## Tests

```bash
pytest
```

Unit tests cover validation, routing, ticker resolution, and hybrid reranking without live Postgres, Qdrant, or Ollama.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `relation filing_chunks does not exist` | Run ingest in [sec-edgar-filings-to-pgvector](https://github.com/sanjuthomas/sec-edgar-filings-to-pgvector) |
| Connection refused on `5433` | Start pgvector in the ingest project |
| Qdrant connection errors | Start [sec-edgar-filings-to-qdrant](https://github.com/sanjuthomas/sec-edgar-filings-to-qdrant); check `QDRANT_URL` (`http://localhost:6333` on host, `host.docker.internal:6333` from chat container) |
| Port `8095` already in use | Stop the other process or change `SERVER_PORT` |
| Ollama timeout / slow answers | Large models (e.g. `qwen3:30b`) can take minutes; try a smaller model |
| Poor search quality | Ensure `bge-m3` is pulled; try ticker/form filters |
| Embedding errors / wrong dimensions | Run `ollama pull bge-m3`; indexes must be 1024-dim BGE-M3 |

---

## License

MIT — see [LICENSE](LICENSE).
