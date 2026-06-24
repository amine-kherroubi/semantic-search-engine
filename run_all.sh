#!/usr/bin/env bash
# run_all.sh
# Master script - runs the full pipeline from scratch.
# Designed for a fresh Ubuntu machine.
# Edit the variables in the CONFIG section below before running.
set -euo pipefail

# CONFIG - edit these values
DB_NAME="semantic_search"
DB_USER="postgres"
DB_PASSWORD="postgres"       # change this!
INGEST_LIMIT=5000            # number of documents to ingest
BATCH_SIZE=64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Write .env from config
cat > "$SCRIPT_DIR/.env" <<EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=10
BATCH_SIZE=$BATCH_SIZE
EOF
echo "[run_all] .env written."

echo "Semantic Search Engine - Full Pipeline"

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/scripts/*.sh

echo "Step 1/6: Install dependencies"
bash "$SCRIPT_DIR/scripts/01_install_deps.sh"

echo ""
echo "Step 2/6: Configure PostgreSQL"
bash "$SCRIPT_DIR/scripts/02_setup_postgres.sh"

echo ""
echo "Step 3/6: Apply database schema"
bash "$SCRIPT_DIR/scripts/03_init_schema.sh"

echo ""
echo "Step 4/6: Ingest dataset (limit=$INGEST_LIMIT)"
LIMIT=$INGEST_LIMIT BATCH_SIZE=$BATCH_SIZE \
    bash "$SCRIPT_DIR/scripts/04_ingest.sh"

echo ""
echo "Step 5/6: Run evaluation"
bash "$SCRIPT_DIR/scripts/06_evaluate.sh"

echo ""
echo "Step 6/6: Pipeline complete."
echo "To search interactively: bash scripts/05_search.sh"
echo "Single query: bash scripts/05_search.sh \"your query\""
