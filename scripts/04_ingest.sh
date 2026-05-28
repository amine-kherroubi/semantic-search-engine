#!/usr/bin/env bash
# =============================================================================
# 04_ingest.sh
# Download the dataset, generate embeddings, and store in pgvector.
#
# Options (edit below or pass as env vars):
#   LIMIT      — how many documents to ingest  (default: 5000)
#   BATCH_SIZE — embedding batch size          (default: 64)
# =============================================================================
set -euo pipefail

LIMIT="${LIMIT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "════════════════════════════════════════"
echo "  Step 4 — Ingest documents"
echo "  Limit:      $LIMIT docs"
echo "  Batch size: $BATCH_SIZE"
echo "════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/scripts/ingest.py" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "✓ Ingestion complete."
echo "  Next: run  bash scripts/05_search.sh"
