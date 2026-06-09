# Agentic RAG Chatbot

A production-minded, **multi-tenant Agentic RAG chatbot** that answers questions
**strictly from a client's own uploaded documents** — with full tenant isolation,
source citations, a lightweight decision-making agent, and an honest fallback when
the answer isn't in the documents.

Built with FastAPI, Qdrant, PostgreSQL, Celery, and Claude.

## What it does

```
Client uploads documents
  → chunked + embedded + stored in Qdrant (one isolated collection per tenant)

End user asks a question (web widget or API)
  → question embedded → semantic search filtered by tenant_id
  → agent decides: answer  /  re-query once  /  escalate
  → Claude answers ONLY from the retrieved chunks, with sources
  → nothing relevant? → "I don't have that information" (no hallucination)
```

The agent never lets Claude answer without retrieved context, never crosses tenant
boundaries, and logs every Q&A.

## Architecture

```
                         ┌──────────────┐
  Web widget / API  ───► │  FastAPI app │
                         └──────┬───────┘
        X-API-Key → tenant      │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  agent  → retrieval → embeddings → Qdrant       │
        │            │                                    │
        │            └─► generation (Claude, grounded)    │
        └───────┬───────────────────────┬────────────────┘
                │                        │
          PostgreSQL                 Celery + Redis
        (tenants, docs,           (background ingestion:
         chat logs)                chunk → embed → Qdrant)
```

Each domain is a self-contained package under `app/`: `tenants`, `embeddings`,
`documents`, `retrieval`, `chat`, `agent`, plus `clients/` (the only place the
Qdrant / Claude / Redis / embedding SDKs are touched) and `security/`.

## Stack

| Layer        | Choice |
|--------------|--------|
| Language     | Python 3.12+ (`mypy --strict`, `ruff`) |
| Framework    | FastAPI (async, package per domain) |
| Validation   | Pydantic v2 + pydantic-settings |
| Vector DB    | Qdrant — one collection per tenant |
| Relational   | PostgreSQL + async SQLAlchemy + Alembic |
| Queue        | Celery + Redis (background ingestion) |
| Embeddings   | FastEmbed (local, `BAAI/bge-small-en-v1.5`, 384-dim) — no API key |
| LLM          | Anthropic Claude (`claude-sonnet-4-6` by default) |
| Tests        | pytest + httpx (every endpoint tested) |

## Getting started

Prerequisites: [Docker](https://www.docker.com/), [uv](https://docs.astral.sh/uv/),
and an Anthropic API key.

```bash
docker compose up -d                      # start Postgres, Redis, Qdrant
uv sync                                    # install deps (uv manages Python + venv)
cp .env.example .env                       # then set ANTHROPIC_API_KEY in .env
uv run alembic upgrade head                # create the database schema
uv run uvicorn app.main:app --reload       # API + web widget on :8000
uv run celery -A app.celery_app:celery_app worker --loglevel=info   # in another terminal
```

Check health:

```bash
curl localhost:8000/health
# {"status":"ok","postgres":"ok","redis":"ok","qdrant":"ok"}
```

## Try it

**Web widget:** open <http://localhost:8000/widget/>, paste a tenant API key, and chat.

**Scripted demo (≈2 minutes):**

```bash
./scripts/demo.sh
```

It creates a tenant, uploads `samples/acme_handbook.txt`, waits for ingestion, then
asks grounded questions and one out-of-scope question (which is honestly escalated).

**By hand:**

```bash
# 1. create a tenant (returns an api_key, shown once)
curl -s -X POST localhost:8000/tenants -H 'Content-Type: application/json' \
  -d '{"name":"Acme"}'

# 2. upload a document (ingested in the background)
curl -s -X POST localhost:8000/documents -H "X-API-Key: <key>" \
  -F "file=@samples/acme_handbook.txt"

# 3. ask a question
curl -s -X POST localhost:8000/chat -H "X-API-Key: <key>" \
  -H 'Content-Type: application/json' -d '{"question":"What is the return policy?"}'
```

## API

| Method | Path                | Auth        | Purpose |
|--------|---------------------|-------------|---------|
| GET    | `/health`           | —           | Postgres/Redis/Qdrant connectivity |
| POST   | `/tenants`          | admin*      | Create a tenant, returns `api_key` |
| GET    | `/tenants/me`       | X-API-Key   | Current tenant |
| POST   | `/documents`        | X-API-Key   | Upload a text document (ingested async) |
| GET    | `/documents/{id}`   | X-API-Key   | Ingestion status |
| POST   | `/chat`             | X-API-Key   | Ask a question → grounded answer + sources |
| GET    | `/widget/`          | —           | Web chat UI |

\* `POST /tenants` requires `X-Admin-Key` only when `ADMIN_API_KEY` is set (open in dev).
`/chat` and `/documents` are rate-limited (`RATE_LIMIT_PER_MINUTE`, per API key).

## Configuration

All settings come from `.env` (see `.env.example`). Key variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` / `REDIS_URL` / `QDRANT_URL` | see `.env.example` | match `docker-compose.yml` |
| `ANTHROPIC_API_KEY` | — | required for `/chat` |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | swappable; no re-index needed |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `BAAI/bge-small-en-v1.5` / `384` | changing requires re-ingestion |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | deterministic chunking |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_SCORE_THRESHOLD` | `5` / `0.5` | below threshold → no context |
| `ADMIN_API_KEY` | empty | gates tenant creation in production |
| `RATE_LIMIT_PER_MINUTE` | `60` | per-client limit on public endpoints |

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy app tests        # strict type check
uv run pytest                # all tests
uv run pytest -m "not integration"   # unit tests only (no running stack needed)
```

Integration tests need the Docker stack up; they are marked `@pytest.mark.integration`.

## Deploy (Docker)

A full containerized stack (API + worker + Postgres + Redis + Qdrant) is defined in
`docker-compose.prod.yml`. Only the API port is published; the datastores stay on the
internal network, and memory limits keep the stack bounded.

```bash
git clone https://github.com/gayratjon-02/agenticRag && cd agenticRag
cp .env.example .env            # set ANTHROPIC_API_KEY, ADMIN_API_KEY; use service-name URLs:
                                #   DATABASE_URL=postgresql+asyncpg://rag:rag@postgres:5432/rag
                                #   REDIS_URL=redis://redis:6379/0
                                #   QDRANT_URL=http://qdrant:6333
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head
```

The API and web widget are then served on port `8090` (`http://<host>:8090/widget/`).

## License

MIT.
