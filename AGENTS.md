# AGENTS.md

Guidance for AI coding agents working in **sec-edgar-filings-chat**.

## Project summary

FastAPI RAG web app for SEC EDGAR filings. Users ask a natural-language question in a Jinja2 UI; the server embeds the query, retrieves similar chunks from **pgvector** or **Qdrant**, and generates a cited answer with a local **Ollama** LLM.

This repo is the **search + answer UI only**. Filing download and vector indexing live in sibling projects:

- [sec-edgar-filings-to-pgvector](https://github.com/sanjuthomas/sec-edgar-filings-to-pgvector)
- [sec-edgar-filings-to-qdrant](https://github.com/sanjuthomas/sec-edgar-filings-to-qdrant)
- [sec-edgar-filings-rag-demo](https://github.com/sanjuthomas/sec-edgar-filings-rag-demo) — full Docker Compose stack

Java reference implementation: [sec-edgar-filings-semantic-search-ui](https://github.com/sanjuthomas/sec-edgar-filings-semantic-search-ui)

Stack: Python **3.11+**, FastAPI, Jinja2, psycopg + pgvector, httpx, Ollama (chat + query embeddings).

---

## Embedding model policy (required)

Query embeddings **must** match the ingest pipeline.

| Setting | Value |
|---------|--------|
| Model | `bge-m3` (Ollama) |
| Dimensions | **1024** |
| Runtime | Ollama `/api/embeddings` via `OLLAMA_EMBEDDING_MODEL` |
| Chat | Separate Ollama model (`OLLAMA_CHAT_MODEL`) |

Requires `ollama pull bge-m3` before search. Query embeddings **must** match the ingest pipeline (`BAAI/bge-m3`, 1024-dim).

Agents **must not** switch to `bge-small-en-v1.5` (384-dim) or other models without re-indexing both pgvector and Qdrant collections.

---

## Vector store architecture

Retrieval is routed by `ChunkSearchRouter` to repository implementations:

| Store | Class | Access |
|-------|-------|--------|
| `pgvector` | `PgVectorChunkRepository` | psycopg cosine search on `filing_chunks` + `filings` |
| `qdrant` | `QdrantChunkRepository` | REST query to `/collections/{collection}/points/query` |

- Config: `DEFAULT_VECTOR_STORE`, `QDRANT_URL`, `QDRANT_COLLECTION`
- Default vector store: **qdrant**
- Qdrant URL default: `http://localhost:16333` (host); Compose demo uses `http://qdrant:6333` internally.
- Schema is owned by ingest projects — this app does **not** create tables or Qdrant collections.

---

## RAG pipeline (server-side, conversational)

All orchestration happens in `RagSearchService` — each turn re-retrieves before answering.

1. `search` routes validate `ChatForm` and load/create a `Conversation` from the session cookie.
2. Optional `TickerResolver` infers ticker from the retrieval query when no ticker filter is set.
3. `build_retrieval_query` expands short follow-ups with the prior user message for embedding search.
4. `OllamaClient.embed` embeds the retrieval query via Ollama `bge-m3` (1024-dim).
5. `ChunkSearchRouter` retrieves top-K chunks from the selected vector store.
6. `OllamaClient.chat_messages` sends prior turns plus the latest user message with fresh excerpts.
7. Jinja2 renders the full chat thread with per-turn source cards.

**UI form fields** (`ChatForm`):

| Field | Required | Notes |
|-------|----------|-------|
| `message` | Yes | Max 2000 chars; cleared after each send |
| `chat_model` | Yes | Populated from Ollama `/api/tags`; validated server-side |
| `vector_store` | Yes | `pgvector` or `qdrant` |
| `chunk_count` | Yes | 1–500; presets 10/25/50/100 in UI; default from `SEARCH_TOP_K` |
| `ticker` | No | Uppercased filter |
| `form` | No | e.g. `10-K` |

Conversation state lives in an in-memory `ConversationStore` keyed by session cookie (`conversation_id`). For multi-instance deployments, replace with Redis or similar.

---

## Testing conventions

- Run `pytest` after code changes (same as CI).
- Unit tests live under `tests/`; no integration tests against live pgvector/Qdrant/Ollama today.
- Prefer testing pure logic (`TickerResolver`, `VectorStoreType`, form normalization) without spinning up external services.

Agents **should** add tests when changing validation, routing, or ticker-resolution logic.

---

## Code conventions

- Package root: `app/`
- DTOs: Pydantic models in `app/models.py`
- Config: `app/config.py` via `pydantic-settings`
- Repositories: `app/repositories/` — one implementation per vector store
- Routes: `app/routes/search.py` — GET `/`, POST `/chat`, POST `/chat/new`
- Match existing style; avoid unrelated refactors.
- Keep behavior aligned with the Java reference implementation unless explicitly changing the Python port.

---

## Commands

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8095 --reload   # dev server
pytest                                                       # unit tests (CI)
docker compose up --build                                    # local image
```

UI: http://localhost:8095/

**Local prerequisites** (not started by this repo):

- PostgreSQL + pgvector on `localhost:5433`, database `edgar`
- Qdrant on `localhost:16333` (if using Qdrant)
- Ollama on `localhost:11434` with chat models (default config: `qwen3:30b`)

---

## Do not

- Commit secrets, `.env`, or credentials.
- Change embedding model or dimensions without coordinating with ingest repos.
- Use a generic vector-store abstraction that hides ingest-project schema — this app queries ingest tables/collections directly.
- Remove user-selectable vector store or Ollama model dropdowns without maintainer approval.
- Edit unrelated files or expand scope beyond the task.
- Commit unless explicitly requested by the user.

---

## References

- [README.md](README.md) — features, configuration, troubleshooting
- [sec-edgar-filings-rag-demo](https://github.com/sanjuthomas/sec-edgar-filings-rag-demo) — full-stack Compose architecture
- [LICENSE](LICENSE) — MIT
