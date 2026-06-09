# Agentic RAG Chatbot

A multi-tenant, agentic Retrieval-Augmented Generation chatbot that answers questions **strictly from a client's own uploaded documents** — with full tenant isolation, source citations, and an honest fallback when the answer isn't in the documents.

## How it works

```
Client uploads documents
  → chunk + embed + store in Qdrant (one collection per tenant)

User asks a question
  → embed the question
  → semantic search in Qdrant (filtered by tenant)
  → retrieve top-k relevant chunks
  → agent decides: answer / re-query once / escalate
  → Claude answers ONLY from the retrieved chunks, with sources
  → nothing relevant? → "I don't have that information"
```

## Stack

- **Python 3.12+ / FastAPI** — async API, package per domain
- **Pydantic v2** — request/response validation and settings
- **Qdrant** — vector store, isolated per tenant
- **PostgreSQL + SQLAlchemy (async) + Alembic** — metadata and chat logs
- **Celery + Redis** — background document ingestion
- **Anthropic Claude** — grounded answer generation
- **pytest** — every endpoint is tested

## Status

Under active development, built in phases (foundation → tenants → embeddings → ingestion → retrieval → generation → agent → channel → hardening).

## Getting started

Setup and run instructions will be documented here as the foundation lands.
