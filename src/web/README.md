# Web UI (`src/web`)

A minimal, developer-oriented web interface for browsing the ingested
article dataset and querying it through the project's two existing
retrieval approaches — **TF-IDF** and **Semantic (pgvector)** — side by
side.

This did not exist in the original project — the only interfaces were
CLI scripts (`scripts/search.py`, `scripts/evaluate.py`). Everything
under `src/web` is new. No files under `src/search`, `src/db`,
`src/embeddings`, or `src/utils` were modified.

```
src/web/
├── README.md                 (this file)
├── backend/
│   ├── __init__.py
│   ├── main.py                FastAPI app: routes for articles, search, health
│   ├── schemas.py              Pydantic request/response models
│   ├── engines.py               Lazy singleton cache for the TF-IDF index
│   └── snippets.py              Query-term highlighting / excerpt extraction
└── frontend/
    ├── index.html               Page shell (Search tab + Browse Articles tab)
    ├── styles.css                Dark, dense, developer-console styling
    └── app.js                    All UI logic — no framework, no build step
```

---

## Scope

The approach selector in the UI offers exactly the two retrieval
methods that already existed in this project:

| Approach | Backed by                                                  |
| -------- | ---------------------------------------------------------- |
| TF-IDF   | `src/search/classical.py` (existing, untouched)          |
| Semantic | `src/search/semantic.py` (existing, untouched, pgvector) |

This is deliberately scoped to existing functionality only — no new
retrieval logic was introduced anywhere in `src/search`.

**A note on "date":** the brief asks for a `date` column in the article
listing. The `documents` table (`sql/schema.sql`) has no publish-date
field, and the AG News dataset itself doesn't ship one — `scripts/ingest.py`
only stores `title`, `content`, `source` (always `"ag_news"`), and a
`metadata.label` category. The UI shows `created_at` (the row's
*ingestion* timestamp) in the date column instead, labeled "Ingested" so
it isn't mistaken for an article publish date. Similarly, `category` is
not a real column — it's read out of `metadata->>'label'`.

---

## Running it

From the **project root** (the directory containing `src/`, `scripts/`, `sql/`):

* [ ]
  ```bash
  # 1. Make sure the existing project is already set up and ingested
  #    (see the main README — .env configured, Postgres running, schema
  #    applied, documents ingested). This UI is a presentation layer over
  #    that existing pipeline; it does not replace any setup step.

  # 2. Install the two extra packages the web UI needs
  pip install fastapi "uvicorn[standard]"
  # (or add them to your requirements.txt — see below)

  # 3. Start the API server (it also serves the frontend — see below)
  uvicorn src.web.backend.main:app --reload --port 8000

  # 4. Open the UI
  #    http://localhost:8000
  ```

`src/web/backend/main.py` mounts `src/web/frontend` as static files at
`/`, so step 3 alone serves both the API (`/api/...`) and the UI from
one process on one port — no separate frontend server, no build step,
no `npm install`. Opening `http://localhost:8000` loads `index.html`,
which calls back into the same origin's `/api/*` routes.

If you'd rather serve the frontend separately (e.g. a plain
`python -m http.server` in `src/web/frontend`), set `API_BASE` near the
top of `app.js` to the backend's full URL (e.g.
`http://localhost:8000`) so the two origins don't mismatch.

### Dependencies

Only two new packages are required on top of your existing
`requirements.txt`:

```
fastapi>=0.110
uvicorn[standard]>=0.29
```

### First TF-IDF search is slower than the rest

TF-IDF builds its index by scanning the full `documents` table in
memory (`TfidfVectorizer.fit_transform`). That happens once per
process, lazily, on the first search request that uses it
(`src/web/backend/engines.py`), and is cached for every request after
that. Semantic search has no such warm-up since the ANN index already
lives in Postgres. If you re-ingest documents while the server is
running, call `POST /api/admin/rebuild-indices` (or just restart the
server) so the in-memory TF-IDF index picks up the new corpus.

---

## API reference

All endpoints are also browsable/testable at `http://localhost:8000/docs`
(FastAPI's auto-generated Swagger UI).

### `GET /api/health`

```json
{ "status": "ok", "document_count": 30000, "indices_ready": {"tfidf": true, "semantic": true} }
```

### `GET /api/articles?page=1&page_size=25&category=Sports&q=match`

Server-side paginated listing. `category` and `q` are optional filters.

```json
{
  "items": [
    {
      "id": 42,
      "title": "...",
      "source": "ag_news",
      "category": "Sports",
      "created_at": "2026-06-20T14:32:00+00:00",
      "metadata": {"label": "Sports"},
      "content_preview": "first 200 chars..."
    }
  ],
  "page": 1, "page_size": 25, "total": 30000, "total_pages": 1200
}
```

### `GET /api/articles/{id}`

Full content + metadata for one article (used by the hover/expand preview).

### `GET /api/categories`

`["Business", "Sci/Tech", "Sports", "World"]` — distinct labels, for the filter dropdown.

### `POST /api/search`

```json
// request
{ "query": "renewable energy breakthroughs", "approach": "semantic", "top_k": 10 }
```

```json
// response
{
  "results": [
    {
      "doc_id": 42, "rank": 1, "title": "...", "score": 0.81,
      "snippet": "...**renewable** **energy** projects are...",
      "source": "ag_news", "category": "Sci/Tech", "approach": "semantic"
    }
  ],
  "stats": {
    "query": "renewable energy breakthroughs", "approach": "semantic",
    "latency_ms": 41.2, "num_results": 10,
    "indexing_method": "pgvector IVFFlat (cosine similarity)",
    "notes": null
  }
}
```

`approach` is one of `"tfidf" | "semantic"`.

### `POST /api/admin/rebuild-indices`

Forces a rebuild of the in-memory TF-IDF cache. Call this after
re-running ingestion without restarting the server.

---

## Frontend notes

- No build step, no framework, no npm dependency — plain HTML/CSS/JS.
  `app.js` uses `fetch` and vanilla DOM APIs only.
- **Search tab**: query box, approach selector (TF-IDF / Semantic),
  top-k selector, a stats strip (latency / result count / approach /
  indexing method), and a result list. Hovering or focusing a result
  title opens a popover with full metadata and content (fetched from
  `GET /api/articles/{id}`).
- **Browse Articles tab**: paginated table (id/title/source/category/
  ingested date), a text filter (debounced, 350ms) and a category
  dropdown, both filtering server-side via query params. Hovering a
  title opens the same metadata popover used in Search.
- Loading and empty states are explicit: a status line shows "Loading…"
  / "No results…" / a clear error message rather than a silently blank
  screen, for both tabs.
- Snippets: the backend wraps matched query terms in the excerpt with
  `**term**`; the frontend turns those into `<mark>` highlights after
  HTML-escaping the rest of the text, so article content can never
  inject markup.

---

## Known limitations / things to revisit if this grows past a dev tool

- The TF-IDF index is per-process and in-memory; running multiple API
  server workers means each worker rebuilds its own copy independently
  and they can drift after re-ingestion until each is rebuilt. Fine for
  a single-developer tool, not for a multi-instance deployment.
- CORS is wide open (`allow_origins=["*"]`) since this is meant for
  local development. Tighten this before exposing the API beyond
  localhost.
- No auth on `/api/admin/rebuild-indices` — anyone who can reach the
  API can trigger a rebuild. Acceptable for a local tool, not for a
  shared/public deployment.
