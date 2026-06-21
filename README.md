# sec-edgar-filings-chat

Conversational RAG web app for SEC EDGAR filings. Ask a natural-language question, retrieve matching chunks from **pgvector** or **Qdrant**, and get a cited answer from a local **Ollama** LLM.

Python port of the [Spring Boot semantic search UI](https://github.com/sanjuthomas/sec-edgar-filings-semantic-search-ui).

![SEC EDGAR Semantic Search — cited answer for a Goldman Sachs 10-K question](docs/chat-screenshot.png)

*Example: ask about Goldman Sachs Q1 results with ticker `GS` and form `10-K`; the assistant returns a cited summary with expandable source cards (25 chunks from pgvector, `qwen3:30b`).*

| | |
|---|---|
| **Ingest (pgvector)** | [sec-edgar-filings-to-pgvector](https://github.com/sanjuthomas/sec-edgar-filings-to-pgvector) |
| **Ingest (Qdrant)** | [sec-edgar-filings-to-qdrant](https://github.com/sanjuthomas/sec-edgar-filings-to-qdrant) |
| **Full Docker stack** | [sec-edgar-filings-rag-demo](https://github.com/sanjuthomas/sec-edgar-filings-rag-demo) |
| **Agent guidance** | [AGENTS.md](AGENTS.md) |
| **License** | [MIT](LICENSE) |

---

## Ecosystem mind map

This repo is the **search + answer UI only**. Filing download, chunking, and vector indexing live in sibling projects.

```mermaid
mindmap
  root((SEC EDGAR RAG))
    Data sources
      SEC EDGAR API
      10-K 10-Q 8-K filings
    Ingest pipelines
      sec-edgar-filings-to-pgvector
        PostgreSQL + pgvector
        filings + filing_chunks tables
        BGE-M3 1024-dim embeddings
      sec-edgar-filings-to-qdrant
        Qdrant collection filing_chunks
        same BGE-M3 vectors
    This app
      sec-edgar-filings-chat
        FastAPI + Jinja2 UI
        per-turn retrieval + chat
        cited answers + source cards
    Runtime services
      Ollama
        bge-m3 query embeddings
        user-selected chat model
      PostgreSQL :5433
      Qdrant :16333
    Demo
      sec-edgar-filings-rag-demo
        Docker Compose all-in-one
```

---

## Request dataflow

Each chat turn **re-retrieves** fresh chunks before generating an answer. Conversation history is kept in memory (session cookie → `ConversationStore`).

```mermaid
flowchart TB
    subgraph Browser["Browser (Jinja2 UI)"]
        Form["ChatForm<br/>message · model · store · top-K · ticker · form"]
    end

    subgraph FastAPI["FastAPI app"]
        Routes["routes/search.py<br/>GET / · POST /chat · POST /chat/new"]
        Store["ConversationStore<br/>in-memory turns + settings"]
        RAG["RagSearchService"]
        Ticker["TickerResolver<br/>infer ticker from query"]
        RQuery["build_retrieval_query<br/>expand short follow-ups"]
        Router["ChunkSearchRouter"]
        Hybrid["PgHybridSearchService<br/>optional vector + BM25 RRF"]
    end

    subgraph Ollama["Ollama :11434"]
        Embed["/api/embeddings · bge-m3 · 1024-dim"]
        Chat["/api/chat · user-selected model"]
    end

    subgraph VectorStores["Vector stores (pre-indexed by ingest)"]
        PG["pgvector<br/>cosine on filing_chunks"]
        QD["Qdrant REST<br/>/collections/filing_chunks/points/query"]
        BM25["pg BM25 keyword search<br/>when PGSEARCH_ENABLED"]
    end

    Form -->|POST /chat| Routes
    Routes --> Store
    Routes --> RAG
    RAG --> RQuery
    RQuery --> Ticker
    Ticker --> Embed
    Embed --> Router
    Embed --> Hybrid
    Router --> PG
    Router --> QD
    Hybrid --> PG
    Hybrid --> BM25
    PG --> RAG
    QD --> RAG
    BM25 --> RAG
    RAG -->|source excerpts + prior turns| Chat
    Chat --> RAG
    RAG --> Store
    Store -->|HTML thread + source cards| Form
```

### Turn-by-turn pipeline

| Step | Component | What happens |
|------|-----------|--------------|
| 1 | `search.py` | Validate `ChatForm`; load or create `Conversation` from signed session cookie |
| 2 | `build_retrieval_query` | Expand terse follow-ups (e.g. “how much was it?”) with the prior user message |
| 3 | `TickerResolver` | Apply explicit ticker filter or infer one from the retrieval query |
| 4 | `OllamaClient.embed` | Embed retrieval query with `bge-m3` (must match ingest index) |
| 5 | Retrieval | **pgvector**: cosine search, or hybrid vector + BM25 RRF when enabled · **Qdrant**: REST vector query |
| 6 | `OllamaClient.chat_messages` | Send system prompt, prior Q&A, and fresh excerpts; get cited answer |
| 7 | Jinja2 | Render full thread with collapsible source cards linking to SEC EDGAR |

Use **New conversation** (`POST /chat/new`) to clear context.

---

## Features

- **Multi-turn chat** — message history, per-turn source cards, **New conversation** reset
- **Dual vector stores** — pgvector (psycopg) or Qdrant (REST); selectable in the UI
- **Configurable chunk count** — presets 10 / 25 / 50 / 100 or any value from 1–500
- **Cited RAG answers** — inline `[1]`, `[2]`, … citations with SEC EDGAR links
- **Optional pgvector hybrid search** — vector + BM25 with reciprocal rank fusion when `PGSEARCH_ENABLED=true`
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
- **PostgreSQL + pgvector** on `localhost:5433`, database `edgar` (when using pgvector)
- **Qdrant** on `localhost:16333` (when using Qdrant)

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

### Compose

Requires **pgvector**, **Qdrant** (if selected), and **Ollama** reachable from the container:

```bash
docker compose up --build
```

### Manual run

```bash
docker run --rm -p 8095:8095 \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5433/edgar \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e QDRANT_URL=http://host.docker.internal:16333 \
  --add-host=host.docker.internal:host-gateway \
  sec-edgar-filings-chat:local
```

For a full stack (Postgres, Qdrant, Ollama, ingest, UI), use [sec-edgar-filings-rag-demo](https://github.com/sanjuthomas/sec-edgar-filings-rag-demo).

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
| `HYBRID_RETRIEVAL_TOP_K` | `50` | Candidates per leg before hybrid fusion |
| `DEFAULT_VECTOR_STORE` | `pgvector` | Default store (`pgvector` or `qdrant`) |
| `QDRANT_URL` | `http://localhost:16333` | Qdrant REST base URL |
| `QDRANT_COLLECTION` | `filing_chunks` | Qdrant collection name |
| `SESSION_SECRET_KEY` | `dev-only-change-in-production` | Signs the browser session cookie |
| `CONVERSATION_MAX_TURNS` | `40` | Max user+assistant turns kept in memory |

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
| Qdrant connection errors | Start Qdrant; check `QDRANT_URL` (host `16333`, Compose internal `6333`) |
| Port `8095` already in use | Stop the other process or change `SERVER_PORT` |
| Ollama timeout / slow answers | Large models (e.g. `qwen3:30b`) can take minutes; try a smaller model |
| Poor search quality | Ensure `bge-m3` is pulled; try ticker/form filters |
| Embedding errors / wrong dimensions | Run `ollama pull bge-m3`; indexes must be 1024-dim BGE-M3 |

---

## License

MIT — see [LICENSE](LICENSE).
