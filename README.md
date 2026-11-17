# Enterprise Knowledge Intelligence Platform

Retrieval-augmented generation (RAG) over enterprise documents: ingestion, chunking, embeddings,
hybrid retrieval, reranking and cited answers from Claude, with document-level access control,
an evaluation harness, and latency/token-cost tracing.

## Architecture

```
Upload ─► Parse ─► Chunk ─► Embed ─► Postgres (pgvector HNSW + full-text GIN)
                                             │
Query ─► Auth (JWT → groups) ─► Hybrid search (vector + keyword, RRF fusion)
       ─► ACL filter in SQL ─► Cross-encoder rerank ─► Claude with citations
       ─► SSE stream of answer + sources
              ↘ Redis cache   ↘ Trace (latency, tokens, $)   ↘ Eval harness
```

| Layer        | Choice                                   | Why |
|--------------|------------------------------------------|-----|
| API          | FastAPI + Pydantic                       | Async, typed, native streaming |
| Store        | PostgreSQL 16 + pgvector                 | Vectors, keyword search and ACLs in one transactional DB |
| Cache        | Redis 7                                  | Query-embedding and answer caching |
| Embeddings   | bge-small-en-v1.5 (local, 384-d)         | No extra API dependency |
| Generation   | Claude (Anthropic API)                   | Native citations, streaming |
| Packaging    | Docker / docker compose                  | Same stack locally and in CI |

## Roadmap

- [x] **Phase 1 – Foundation:** project layout, Docker Compose, config, schema + migrations, health probes, structured logs, CI
- [ ] **Phase 2 – Ingestion:** upload, PDF/DOCX/MD/HTML parsing, heading-aware chunking, embeddings
- [ ] **Phase 3 – Retrieval:** hybrid search (RRF), ACL filtering, reranking
- [ ] **Phase 4 – Answering:** streamed, cited answers; structured extraction endpoint
- [ ] **Phase 5 – Auth & caching:** JWT, users/groups, admin grants, Redis caching
- [ ] **Phase 6 – Observability & evals:** per-request traces and cost, retrieval/answer eval harness in CI
- [ ] **Phase 7 – UI & deploy:** chat UI with citations, Azure Container Apps

## Quick start

Requires Docker Desktop.

```bash
docker compose up -d --build
```

- API docs: http://localhost:8000/docs
- Liveness: http://localhost:8000/health/live
- Readiness (Postgres, Redis, schema): http://localhost:8000/health/ready

Migrations run automatically when the API container starts.

## Development

Run lint and tests inside the container (tests use the compose Postgres and Redis):

```bash
docker compose run --rm -v "${PWD}:/code" api sh -c "ruff check . && ruff format --check . && pytest"
```

Create a new migration after changing `app/db/models.py`:

```bash
docker compose run --rm -v "${PWD}:/code" api alembic revision --autogenerate -m "describe change"
```

`tests/test_schema.py::test_migrations_match_models` fails if models and migrations drift apart.

Host ports: Postgres `5433`, Redis `6380`, API `8000` (chosen to avoid clashing with local installs).

## Project layout

```
app/
  main.py            app factory, lifespan (DB + Redis pools)
  core/              settings, JSON logging, request-id middleware
  db/                SQLAlchemy models, async session
  api/               routers (health, …)
migrations/          Alembic migrations
tests/               integration tests
.github/workflows/   CI: lint, migrate, test, downgrade check, prod image build
```

## Data model

- `users`, `groups`, `user_groups`: who can see what
- `documents`: one row per uploaded file (deduplicated by SHA-256), with ingestion status
- `document_access`: which groups can read a document
- `chunks`: text chunks with page/section, a 384-d `embedding` (HNSW, cosine) and a generated `tsv` column (GIN) for keyword search
