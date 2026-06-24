# Semantic Search Engine with pgvector

A document retrieval system that finds results by **semantic meaning** rather than exact keyword matching. Natural-language queries are encoded into dense vector representations and matched against a PostgreSQL vector database using cosine similarity, with a classical TF-IDF baseline for comparative evaluation.

---

## Overview

Traditional keyword search fails when a user's phrasing differs from the document's wording. This system addresses that limitation by embedding both documents and queries into a shared vector space using a pre-trained transformer model, enabling retrieval based on conceptual similarity rather than lexical overlap.

**Dataset:** AG News (Hugging Face) - approximately 120,000 English news articles across four categories: World, Sports, Business, and Sci/Tech.

**Embedding model:** `all-MiniLM-L6-v2` (Sentence-Transformers) - produces 384-dimensional unit vectors optimised for semantic textual similarity.

---

## Technology Stack

| Component          | Technology                           |
| ------------------ | ------------------------------------ |
| Language           | Python 3.10+                         |
| Vector database    | PostgreSQL 16 with pgvector          |
| Embeddings         | Sentence-Transformers (Hugging Face) |
| Baseline retrieval | TF-IDF with bigrams (scikit-learn)   |
| ORM                | SQLAlchemy 2                         |
| Visualisation      | matplotlib, seaborn                  |
| CLI output         | Rich                                 |

All dependencies are open-source.

---

## Repository Structure

```
semantic-search-engine/
|
|-- run_all.sh                    Master setup and pipeline script
|-- requirements.txt              Python dependencies
|-- .env.example                  Environment variable template
|-- report.md                     Structured technical report
|
|-- scripts/
|   |-- 01_install_deps.sh        Install system packages, PostgreSQL, pgvector, and Python deps
|   |-- 02_setup_postgres.sh      Create database and user
|   |-- 03_init_schema.sh         Apply SQL schema via setup_db.py
|   |-- 04_ingest.sh              Download dataset, generate embeddings, store in pgvector
|   |-- 05_search.sh              Interactive search CLI
|   |-- 06_evaluate.sh            Benchmark evaluation with charts
|   |-- setup_db.py               Database initialisation script
|   |-- ingest.py                 Ingestion pipeline
|   |-- search.py                 Search CLI entry point
|   `-- evaluate.py               Evaluation suite
|
|-- src/
|   |-- db/
|   |   |-- connection.py         SQLAlchemy engine, session context manager, psycopg2 helpers
|   |   `-- models.py             ORM models (Document, Embedding) and CRUD helpers
|   |-- embeddings/
|   |   `-- encoder.py            Sentence-Transformers wrapper with batching and caching
|   |-- search/
|   |   |-- semantic.py           pgvector cosine similarity search
|   |   `-- classical.py          In-memory TF-IDF search engine
|   `-- utils/
|       |-- preprocess.py         Text cleaning, chunking, and truncation
|       `-- display.py            Rich terminal output for search results
|
|-- sql/
|   |-- schema.sql                Extension, tables, IVFFlat index, and view definitions
|   `-- reset.sql                 Drop all tables (destructive)
|
|-- notebooks/
|   `-- analysis.ipynb            EDA, t-SNE visualisation, and comparative analysis
|
|-- tests/
|   |-- test_preprocess.py        Unit tests for preprocessing utilities
|   `-- test_validation.py        Validation tests for pipeline parameters and error handling
|
`-- data/                         Generated outputs (charts, CSVs) - not committed
```

---

## Setup and Usage

### Option A - Automated (recommended)

**Before running, you must create and configure `.env` manually.** The script reads it but will never create or overwrite it for you.

```bash
# 1. Create your environment file from the template
cp .env.example .env

# 2. Open it and set your values — at minimum, change DB_PASSWORD
nano .env
```

Then run the master script:

```bash
bash run_all.sh
```

This executes all five steps in sequence: dependency installation, database creation, schema setup, document ingestion, and evaluation. The script will exit immediately with a clear error message if `.env` is missing.

You can override `INGEST_LIMIT` and `BATCH_SIZE` at the command line without touching `.env`:

```bash
INGEST_LIMIT=20000 BATCH_SIZE=128 bash run_all.sh
```

### Option B - Step by step

```bash
# 1. Install system packages, PostgreSQL 16, pgvector, and Python packages
bash scripts/01_install_deps.sh

# 2. Create and configure the environment file
cp .env.example .env
nano .env   # set DB_PASSWORD; optionally add HF_TOKEN and CUDA_VISIBLE_DEVICES

# 3. Create the PostgreSQL database and user
bash scripts/02_setup_postgres.sh

# 4. Apply the schema (creates tables and the IVFFlat index)
bash scripts/03_init_schema.sh

# 5. Download AG News and ingest documents with embeddings
LIMIT=5000 bash scripts/04_ingest.sh

# 6. Run the evaluation suite
bash scripts/06_evaluate.sh

# 7. Search interactively (or pass a query directly)
bash scripts/05_search.sh
bash scripts/05_search.sh "machine learning breakthroughs"
```

---

## Configuration

All settings are read from a `.env` file at the project root. **This file must be created manually before running any script — it is never generated or overwritten automatically.** Copy the template and fill in your values:

```bash
cp .env.example .env
nano .env
```

| Variable               | Default          | Description                                                    |
| ---------------------- | ---------------- | -------------------------------------------------------------- |
| `DB_HOST`              | localhost        | PostgreSQL host                                                |
| `DB_PORT`              | 5432             | PostgreSQL port                                                |
| `DB_NAME`              | semantic_search  | Database name                                                  |
| `DB_USER`              | postgres         | Database user                                                  |
| `DB_PASSWORD`          | postgres         | Database password                                              |
| `EMBEDDING_MODEL`      | all-MiniLM-L6-v2 | Sentence-Transformers model identifier                         |
| `TOP_K`                | 10               | Default number of results returned                             |
| `BATCH_SIZE`           | 64               | Embedding batch size during ingestion                          |
| `HF_TOKEN`             | _(unset)_        | Hugging Face API token — see below                             |
| `CUDA_VISIBLE_DEVICES` | _(unset)_        | Set to `""` to force CPU and silence CUDA warnings — see below |

### Hugging Face token (optional)

Without a token, dataset and model downloads are unauthenticated and subject to rate limits. To enable higher limits:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token**, select **Read** scope, copy the value
4. Add it to `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### CUDA / GPU warning

If your NVIDIA driver is older than version 525, PyTorch will emit a warning on startup:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old...
```

The model falls back to CPU automatically — everything still works, just slower. To suppress the warning, add this to `.env`:

```
CUDA_VISIBLE_DEVICES=
```

This tells PyTorch not to probe for a GPU. Remove the line once the driver is updated (`sudo apt install nvidia-driver-535` then reboot).

---

## Search CLI

```bash
# Interactive mode (press Ctrl-C to quit)
bash scripts/05_search.sh

# Single query with semantic and TF-IDF results side by side
bash scripts/05_search.sh "climate change renewable energy"

# Semantic results only
bash scripts/05_search.sh "neural networks" --no-tfidf

# Return more results
python scripts/search.py --query "space mission" --top-k 20
```

---

## Technical Report

The technical report is available in `report.md`. It follows the required structure: introduction, state of the art, methodology, implementation details, results, critical discussion, and conclusion.

---

## Evaluation

The evaluation script (`scripts/06_evaluate.sh`) runs ten predefined queries through both retrieval methods and measures:

- Query latency in milliseconds
- Result overlap between semantic and TF-IDF at top-k
- Average similarity scores per method

Results are written to `data/evaluation_results.csv` and `data/evaluation_results.png`.

---

## Jupyter Notebook

The notebook at `notebooks/analysis.ipynb` provides:

- Document count and category distribution
- t-SNE projection of 1,000 document embeddings to visualise semantic clustering
- Side-by-side search result comparison across representative queries
- Score distribution histograms and latency charts

To open it:

```bash
source .venv/bin/activate
cd notebooks
jupyter notebook analysis.ipynb
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## Ingesting More Documents

The AG News dataset contains approximately 120,000 articles. To increase the ingestion volume:

```bash
LIMIT=20000 BATCH_SIZE=128 bash scripts/04_ingest.sh
```

Larger batch sizes improve throughput but require more RAM. A batch size of 64 is appropriate for machines with 8 GB of memory.

---

## Changing the Embedding Model

Update `EMBEDDING_MODEL` in `.env` to any model from the [Sentence-Transformers model hub](https://www.sbert.net/docs/pretrained_models.html), then re-run ingestion. Note that the embedding dimension is fixed to 384 in `sql/schema.sql` and `src/db/models.py`; if you switch to a higher-dimensional model such as `all-mpnet-base-v2` (768 dimensions), update both values and reset the database with `psql -f sql/reset.sql` before re-ingesting.

Multilingual support is available via `paraphrase-multilingual-MiniLM-L12-v2`, which also produces 384-dimensional embeddings and requires no schema changes.

---

## Notes on `requirements.txt`

The original `requirements.txt` was missing five packages required at runtime:

| Package           | Reason                                       |
| ----------------- | -------------------------------------------- |
| `torch`           | Required backend for `sentence-transformers` |
| `transformers`    | Model loading and tokenisation               |
| `huggingface-hub` | Model download from the Hugging Face Hub     |
| `rich`            | Coloured terminal output in the search CLI   |
| `tabulate`        | Table formatting in the evaluation report    |

All five are included in the corrected `requirements.txt`.
