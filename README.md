# Enterprise Knowledge Intelligence Platform

Retrieval-augmented generation (RAG) over enterprise documents: ingestion, chunking, embeddings,
hybrid retrieval, reranking and cited answers from an LLM, with document-level access control,
an evaluation harness, and latency/token-cost tracing.

## Architecture

```
Upload ─► Parse ─► Chunk ─► Embed ─► Postgres (pgvector HNSW + full-text GIN)
                                             │
Query ─► Auth (JWT → groups) ─► Hybrid search (vector + keyword, RRF fusion)
       ─► ACL filter in SQL ─► Cross-encoder rerank ─► LLM with citations
       ─► SSE stream of answer + sources
              ↘ Redis cache   ↘ Trace (latency, tokens, $)   ↘ Eval harness
```

| Layer        | Choice                                   | Why |
|--------------|------------------------------------------|-----|
| API          | FastAPI + Pydantic                       | Async, typed, native streaming |
| Store        | PostgreSQL 16 + pgvector                 | Vectors, keyword search and ACLs in one transactional DB |
| Cache        | Redis 7                                  | Query-embedding and answer caching |
| Embeddings   | bge-small-en-v1.5 (local, 384-d)         | No extra API dependency |
| Generation   | Anthropic API (LLM)                      | Native citations, streaming |
| Packaging    | Docker / docker compose                  | Same stack locally and in CI |

## Roadmap

- [x] **Phase 1 – Foundation:** project layout, Docker Compose, config, schema + migrations, health probes, structured logs, CI
- [x] **Phase 2 – Ingestion:** upload, PDF/DOCX/MD/HTML parsing, heading-aware chunking, embeddings, background worker
- [x] **Phase 3 – Retrieval:** hybrid search (RRF), ACL filtering, reranking
- [x] **Phase 4 – Answering:** streamed, cited answers; structured extraction; LLM or free offline generator
- [x] **Phase 5 – Auth & caching:** JWT, users/groups, admin grants, Redis caching
- [x] **Phase 6 – Observability & evals:** per-request traces and cost, retrieval/answer eval harness in CI
- [x] **Phase 7 – UI & deploy:** chat UI with citations, production image and compose, Azure guide

## Quick start

Requires Docker Desktop.

```bash
docker compose up -d --build
```

- Chat UI: http://localhost:8000 (sign in with a demo user, see [Users and access](#users-and-access))
- API docs: http://localhost:8000/docs
- Liveness: http://localhost:8000/health/live
- Readiness (Postgres, Redis, schema): http://localhost:8000/health/ready

A one-shot `migrate` container applies migrations before the API and worker start. The
first worker start downloads the embedding model (~70 MB) into the `models` volume.

## Ingesting documents

Every endpoint except `/health/*` and `/auth/token` needs a bearer token (see
[Users and access](#users-and-access)); uploading, reindexing and deleting need an admin.

```bash
TOKEN=$(curl -s -d "username=admin@northwind.example&password=demo-admin-password" \
  http://localhost:8000/auth/token | jq -r .access_token)
curl -H "Authorization: Bearer $TOKEN" -F "file=@handbook.pdf" http://localhost:8000/documents
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/documents/<id>          # until "ready"
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/documents/<id>/chunks   # inspect chunks
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
curl -X POST http://localhost:8000/search -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query": "how much parental leave do I get?"}'
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

Sample documents for trying it out are in `samples/corpus/`.

## Users and access

There is no self sign-up. Create the first admin (prompts for a password, 12+ characters):

```bash
docker compose exec api python -m app.cli create-admin --email you@example.com --name "Your Name"
```

Or, for a demo, upload `samples/corpus/` and then create the demo groups and users and share
the sample documents with them (refused when `ENVIRONMENT=production`; the passwords in
`samples/demo-users.json` are public):

```bash
docker compose exec api python -m app.cli seed-demo
```

| User | Password | Groups | Can read |
|---|---|---|---|
| `admin@northwind.example` | `demo-admin-password` | admin | everything |
| `hana@northwind.example` | `demo-hana-password` | hr, all-staff | handbook, security policy |
| `eli@northwind.example` | `demo-eli-password` | engineering, all-staff | + order service runbook |

**Endpoints:** `POST /auth/token` (OAuth2 password form; email as `username`),
`GET /auth/me`, `POST /auth/password`, `POST /auth/logout-everywhere`. Admins:
`/admin/users` and `/admin/groups` (CRUD) and `PUT /admin/documents/{id}/groups` to set who
can read a document. The Swagger UI at `/docs` has an **Authorize** button.

**How it works:**

- Passwords are hashed with **argon2id**; login timing is the same for unknown emails.
- Access tokens are **HS256 JWTs** (60 min) holding only the user id and a token version.
  Admin rights and groups are **read from the database on every request**, so removing
  someone from a group takes effect immediately, not when their token expires.
- **Revocation:** changing a password, deactivating a user or "log out everywhere" bumps
  the user's token version, which invalidates every token issued before it.
- **Brute-force protection:** 5 failed logins per email and client address lock that pair
  for 15 minutes (counter in Redis).
- Non-admins see only documents shared with their groups; other ids return **404**, so the
  API doesn't confirm they exist. Admins can't demote, deactivate or delete themselves.
- The app **refuses to start in production** with the default or a short `JWT_SECRET`.

## Caching

Redis caches query embeddings (24 h), search results (10 min) and answers (1 h). A repeated
question goes from ~3 s to ~30 ms end to end on a laptop; responses carry `cached: true`.

- Keys include the caller's **access scope** (admin, or a hash of their sorted groups):
  users with identical access share entries; users with different access never do.
- Keys include an **index version** that the worker bumps after each ingestion, and the API
  bumps on delete or sharing changes, invalidating everything at once without scanning keys.
- Refused or truncated answers are never cached. If Redis is down, requests run uncached.

## Asking questions

Works out of the box with **no API key and no cost**:

```bash
python scripts/ask.py --email hana@northwind.example "How much parental leave does the secondary caregiver get?"
```

### Two answer generators, one interface

| `LLM_PROVIDER` | Used when | What it does |
|---|---|---|
| `extractive` | default without a key | Offline. Picks the sentences from the retrieved passages that best match the question (local embeddings) and cites each one. Free, ~1 s on a laptop CPU. |
| `anthropic` | default when `ANTHROPIC_API_KEY` is set | The LLM writes the answer from the passages, with native citations. |

Both receive the same `search_result` messages and emit the same events, so the API,
streaming format and citations are identical; switching is a config change. The extractive
generator can't combine facts, apply conditions ("220 EUR in Paris" vs "150 EUR elsewhere")
or rephrase. It quotes. `/extract` in offline mode is a heuristic (best-matching sentence;
first number in it for numeric fields; booleans left null).

### Using the Anthropic API (optional, paid)

Put a key in a `.env` file next to `docker-compose.yml` (git ignores it), then restart the
API:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env        # bash / zsh
docker compose up -d api
python scripts/ask.py --email hana@northwind.example "How much parental leave do I get?"
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

- Passages go to the LLM as `search_result` blocks with citations enabled, so every claim
  comes back linked to the passage (and paragraph) it came from. `grounded: false` flags an
  answer with no citations at all.
- Only passages the caller may read are retrieved, so the model never sees anything else.
- If nothing relevant is found (no results, or the best reranked passage scores below
  `ANSWER_MIN_RELEVANCE`), the API answers "not found" **without calling the model**.
- The system prompt treats passages as data, not instructions (prompt-injection defense).
- Retrieval uses its own short DB session, released before the model call starts.

Model settings: Anthropic's Opus model (`ANSWER_MODEL`), effort `low` (configurable via `ANSWER_EFFORT`), with
server-side fallback enabled so a request declined by a safety classifier is retried on
another model. Refusals and truncated answers are reported in `status`.

## Web UI

`http://localhost:8000` serves a chat UI (`app/web/`: plain HTML, CSS and JavaScript, no
build step): sign in, see the documents you can read, ask questions, and watch the answer
stream in. Each citation marker opens the exact passage it came from; the footer shows
latency, generator, cost and whether the answer came from the cache. Admins get an upload
button. It adapts to phone widths and to light/dark mode.

Security: the UI is served with a strict Content-Security-Policy (only its own scripts and
styles, no inline code, no framing), every API response carries `nosniff`,
`X-Frame-Options: DENY` and `Referrer-Policy: no-referrer`, and all server-provided text is
inserted with `textContent`, so document content can't inject script (a test enforces the
absence of HTML sinks). The token lives in `sessionStorage` (per tab, gone when it closes).
CORS is off unless `CORS_ALLOW_ORIGINS` lists a separate frontend domain.

## Production

```bash
export POSTGRES_PASSWORD=... JWT_SECRET=...        # 32+ random characters each
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec api python -m app.cli create-admin --email you@example.com --name You
```

The production image (`Dockerfile`, target `prod`) runs as a non-root user, has the
embedding and reranker models **baked in** (no download at start-up, works offline), a
container health check, and trusts forwarded headers from the load balancer so login
lockout sees real client IPs. Migrations run as a one-off `migrate` service, not on every
start, so several API replicas never race. The compose file publishes only the API port,
requires the secrets from the environment, and refuses demo seeding.
`create-admin --password-stdin` creates the first admin from scripts.

Cloud: [docs/deploy-azure.md](docs/deploy-azure.md) walks through Azure Container Apps
(API, worker and migration job from one image), PostgreSQL Flexible Server with pgvector,
Azure Cache for Redis and an Azure Files share for uploads. It has not been run against a
live subscription.

## Observability

Every search, answer and extraction writes one row to `request_traces`: user, question
(optional, `TRACE_STORE_QUERIES`), outcome, whether it was cached, per-stage timings,
tokens, estimated cost, result count, top relevance score and the chunk ids used. Writing a
trace never fails the request it describes.

- `GET /admin/metrics?hours=24`: per endpoint request count, errors, cache hit rate,
  no-answer rate, p50/p95/max latency, tokens and cost (SQL `percentile_cont` over traces).
- `GET /admin/traces?endpoint=ask&status=error`: recent traces for debugging.

Traces also feed evaluation work: the relevance-cutoff tuning below used the `top_score`
recorded per question.

## Evaluation

```bash
python -m evals.run --setup                        # upload + share samples/corpus, then evaluate
python -m evals.run --check evals/thresholds.json  # exit 1 on regression (runs in CI)
```

`evals/datasets/northwind.jsonl` has **68 labelled questions** over the sample corpus:
paraphrased questions, exact identifiers (`VPN-809`, `HR-100`), table lookups,
unanswerable questions, and access-control probes (a user asking about a document they
can't read). Each answerable question names the section that holds the answer and the facts
an answer must contain. The harness calls the real HTTP API as each question's user, with
`Cache-Control: no-cache`, and writes a Markdown + JSON report to `evals/reports/`.

**Retrieval** (top 10; a result is relevant if it comes from the expected section):

| config | hit@1 | hit@3 | MRR@10 | nDCG@10 | p50 latency | ACL leaks |
|---|---|---|---|---|---|---|
| dense (vector) | 0.95 | 1.00 | 0.972 | 0.979 | 25 ms | 0 |
| keyword (full text) | 0.97 | 0.97 | 0.970 | 0.973 | 2 ms | 0 |
| hybrid (RRF) | 0.93 | 0.98 | 0.961 | 0.970 | 5 ms | 0 |
| **hybrid + reranker** | **1.00** | **1.00** | **1.000** | **1.000** | 826 ms | 0 |

**Answers** (offline extractive generator, $0):

| metric | value |
|---|---|
| fact recall / fully correct | 0.93 / 0.93 |
| grounded (cites a source) | 0.98 |
| citation precision | 0.93 |
| false "not found" on answerable questions | 0.02 |
| correct "not found" on unanswerable questions | 0.89 |
| access-control leaks | 0 |
| latency p50 / p95 | 1.7 s / 3.0 s (laptop CPU) |

What the numbers showed:

- **The reranker is the only configuration with perfect ranking, at ~0.8 s per query** on a
  laptop CPU. Hybrid without it is slightly *below* dense alone here: OR-style keyword
  matches on common words add noise in a corpus this small.
- **The relevance cutoff was too strict.** At 0.02 the system said "not found" to 4 of 58
  answerable questions. A sweep over the reranker scores stored in the traces showed that
  any value from 0.0005 to 0.004 minimises errors; it is now 0.002 (false "not found" 7%
  → 2%, fact recall 0.88 → 0.93), with no new wrong answers.
- **The remaining misses are limits of the extractive generator**, not retrieval: it quoted
  the SEV3 line for a SEV1 question, the symptoms instead of the fix, and "answered" *Who is
  the CEO?* with a sentence mentioning Northwind. These are what an LLM generator fixes.

Caveats: the corpus is small (3 documents, ~20 sections), so retrieval sits near its
ceiling, and the cutoff was tuned on the same questions it is scored on. A larger corpus
and a held-out question set are the next step.

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
  cli.py             create-admin, seed-demo
  core/              settings, logging, middleware, security (passwords, JWT, principal), cache
  db/                SQLAlchemy models, async session
  api/               routers (health, auth, admin, documents, search, ask)
  ingestion/         parsers, chunking, embeddings, storage, pipeline (the job)
  web/               chat UI (HTML/CSS/JS, served at /)
  retrieval/         hybrid search, RRF fusion, cross-encoder reranking
  generation/        Anthropic gateway, offline extractive gateway, prompts, cited answers,
                     structured extraction, pricing
scripts/ask.py       command-line client for /ask
samples/corpus/      fictional company documents for demos and evals
evals/               labelled dataset, metrics, harness, CI thresholds
docs/deploy-azure.md Azure Container Apps deployment guide
docker-compose.prod.yml  production-style single-host stack
migrations/          Alembic migrations
tests/               integration tests
.github/workflows/   CI: lint, migrate, test, downgrade check, prod image build
```

## Data model

- `users`, `groups`, `user_groups`: who can see what
- `documents`: one row per uploaded file (deduplicated by SHA-256), with ingestion status
- `document_access`: which groups can read a document
- `chunks`: text chunks with page/section, a 384-d `embedding` (HNSW, cosine) and a generated `tsv` column (GIN) for keyword search
