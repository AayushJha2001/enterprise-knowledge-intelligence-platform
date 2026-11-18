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
- [x] **Phase 2 – Ingestion:** upload, PDF/DOCX/MD/HTML parsing, heading-aware chunking, embeddings, background worker
- [x] **Phase 3 – Retrieval:** hybrid search (RRF), ACL filtering, reranking
- [x] **Phase 4 – Answering:** streamed, cited answers; structured extraction; Claude or free offline generator
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

A one-shot `migrate` container applies migrations before the API and worker start. The
first worker start downloads the embedding model (~70 MB) into the `models` volume.

## Ingesting documents

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/documents          # 202, status "pending"
curl http://localhost:8000/documents/<id>                             # poll until "ready"
curl http://localhost:8000/documents/<id>/chunks                      # inspect chunking
```

Supported: PDF, DOCX, Markdown, HTML, plain text (max 25 MB). Optional form fields:
`title`, and `group_ids` (repeatable) for access control. Re-uploading identical bytes returns
409 with the existing id. `POST /documents/<id>/reindex` re-runs ingestion; `DELETE` removes the
document, its chunks and the file.

How it works:

1. **Upload** (API) validates type by extension *and* file signature, deduplicates by SHA-256,
   stores the file and inserts a `pending` row. It never parses in the request.
2. **Worker** (`python -m app.worker`) claims the oldest pending document with
   `SELECT … FOR UPDATE SKIP LOCKED`; the documents table is the queue, so several workers can
   run with no broker and no double-processing.
3. **Parse** into blocks that keep page numbers (PDF) and heading paths (DOCX/Markdown/HTML),
   including tables and code blocks.
4. **Chunk** per section (never across headings), ~350 target / 450 max estimated tokens,
   ~50-token overlap; oversized paragraphs split on sentences.
5. **Embed** `title > section path` + chunk text with `BAAI/bge-small-en-v1.5` (local ONNX, CPU).
6. **Store** all chunks in one transaction; Postgres fills the full-text `tsv` column itself.

Failures: unreadable files (corrupt, encrypted, scanned PDFs without text) fail immediately with
a reason; transient errors are retried up to 3 times; a document stuck in `processing` for 15
minutes (crashed worker) is requeued.

## Searching

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "how much parental leave do I get?", "top_k": 5}'
```

Options: `mode` (`hybrid` default, or `dense` / `keyword` for comparison), `rerank`
(default `true`), `document_ids` (restrict to some documents). Each result carries its
document, section path, page range, the final `score`, and the per-stage ranks
(`dense_rank`, `keyword_rank`, `rrf_score`, `rerank_score`); `timings_ms` breaks down latency.

Pipeline:

1. **Dense:** embed the query (bge-small, query encoding) and take the 40 nearest chunks from
   the HNSW index (`ef_search=100`, iterative scan so ACL filtering can't starve results).
2. **Keyword:** Postgres full-text search with OR semantics, ranked by `ts_rank_cd`; catches
   exact identifiers like `VPN-809` or `ORD-5003` that embeddings blur.
3. **Fuse** with Reciprocal Rank Fusion (k=60): rank-based, so no score calibration needed.
4. **Rerank** the top 20 with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), which reads the
   query and passage together. Scores are mapped to 0–1 relevance.

**Access control** is a SQL predicate inside both retrieval queries: a chunk is a candidate
only if the caller is an admin or in a group its document is shared with. Unshared documents
are admin-only. `document_ids` can narrow results but never widen access.

> Until Phase 5 adds login, callers act as a local admin, or as a specific user via an
> `X-User-Id` header. This shortcut is refused when `ENVIRONMENT=production`.

Sample documents for trying it out are in `samples/corpus/`.

## Asking questions

Works out of the box with **no API key and no cost**:

```bash
python scripts/ask.py "How much parental leave does the secondary caregiver get?"
```

### Two answer generators, one interface

| `LLM_PROVIDER` | Used when | What it does |
|---|---|---|
| `extractive` | default without a key | Offline. Picks the sentences from the retrieved passages that best match the question (local embeddings) and cites each one. Free, ~1 s on a laptop CPU. |
| `anthropic` | default when `ANTHROPIC_API_KEY` is set | Claude writes the answer from the passages, with native citations. |

Both receive the same `search_result` messages and emit the same events, so the API,
streaming format and citations are identical; switching is a config change. The extractive
generator can't combine facts, apply conditions ("220 EUR in Paris" vs "150 EUR elsewhere")
or rephrase. It quotes. `/extract` in offline mode is a heuristic (best-matching sentence;
first number in it for numeric fields; booleans left null).

### Using Claude (optional, paid)

Put a key in a `.env` file next to `docker-compose.yml` (git ignores it), then restart the
API:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env        # bash / zsh
docker compose up -d api
python scripts/ask.py "How much parental leave do I get?"
```

On Windows PowerShell, `>` writes UTF-16, which Docker Compose can't read. Use:

```powershell
Set-Content -Path .env -Value "ANTHROPIC_API_KEY=sk-ant-..." -Encoding ascii
```

`POST /ask` with `{"question": "..."}` streams server-sent events: `sources` (the numbered
passages), `text` and `citation` deltas, then `done` with the full answer (Markdown with
`[n]` markers), per-segment citations, token usage, estimated cost and timings. Send
`"stream": false` for a single JSON response.

`POST /extract` fills named fields as schema-validated JSON, each with the numbers of the
passages that support it:

```json
{"query": "hotel expense policy",
 "fields": [{"name": "cap_paris", "description": "Hotel cap per night in Paris", "type": "number"}]}
```

How answers stay grounded:

- Passages go to Claude as `search_result` blocks with citations enabled, so every claim
  comes back linked to the passage (and paragraph) it came from. `grounded: false` flags an
  answer with no citations at all.
- Only passages the caller may read are retrieved, so the model never sees anything else.
- If nothing relevant is found (no results, or the best reranked passage scores below
  `ANSWER_MIN_RELEVANCE`), the API answers "not found" **without calling the model**.
- The system prompt treats passages as data, not instructions (prompt-injection defense).
- Retrieval uses its own short DB session, released before the model call starts.

Model settings: `claude-opus-5-5`, effort `low` (configurable via `ANSWER_EFFORT`), with
server-side fallback enabled so a request declined by a safety classifier is retried on
another model. Refusals and truncated answers are reported in `status`.

## Development

Run lint and tests inside the container. Tests use a separate `ekip_test` database (created
and migrated automatically) and a hashing stand-in for the embedding model, so they never touch
your dev data and need no model download:

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
  main.py            app factory, lifespan (DB + Redis pools, file storage)
  worker.py          ingestion worker process
  core/              settings, JSON logging, request-id middleware, principal (who's asking)
  db/                SQLAlchemy models, async session
  api/               routers (health, documents, search)
  ingestion/         parsers, chunking, embeddings, storage, pipeline (the job)
  retrieval/         hybrid search, RRF fusion, cross-encoder reranking
  generation/        Claude gateway, offline extractive gateway, prompts, cited answers,
                     structured extraction, pricing
scripts/ask.py       command-line client for /ask
samples/corpus/      fictional company documents for demos and evals
migrations/          Alembic migrations
tests/               integration tests
.github/workflows/   CI: lint, migrate, test, downgrade check, prod image build
```

## Data model

- `users`, `groups`, `user_groups`: who can see what
- `documents`: one row per uploaded file (deduplicated by SHA-256), with ingestion status
- `document_access`: which groups can read a document
- `chunks`: text chunks with page/section, a 384-d `embedding` (HNSW, cosine) and a generated `tsv` column (GIN) for keyword search
