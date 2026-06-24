#!/usr/bin/env bash
# run_all.sh
# Master script — runs the full pipeline from scratch on a fresh Ubuntu machine.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BEFORE RUNNING THIS SCRIPT — MANUAL SETUP REQUIRED                      ║
# ║                                                                          ║
# ║  This script reads configuration from a .env file.                       ║
# ║  It will NEVER create or overwrite .env for you.                         ║
# ║                                                                          ║
# ║  1. Copy the template:                                                   ║
# ║       cp .env.example .env                                               ║
# ║                                                                          ║
# ║  2. Open .env and fill in your values:                                   ║
# ║       nano .env                                                          ║
# ║                                                                          ║
# ║     Required:                                                            ║
# ║       DB_PASSWORD  — use a real password (default "yourpassword" is      ║
# ║                      insecure and will cause the setup to fail)          ║
# ║                                                                          ║
# ║     Optional:                                                            ║
# ║       HF_TOKEN     — Hugging Face read-only token for higher download    ║
# ║                      rate limits.  Get one free at:                      ║
# ║                      https://huggingface.co/settings/tokens              ║
# ║                      Leave commented-out to use unauthenticated access   ║
# ║                      (slower on busy days but always works).             ║
# ║                                                                          ║
# ║       INGEST_LIMIT — passed via the LIMIT env-var to 04_ingest.sh.       ║
# ║                      Default is 5000; the full dataset has ~120,000.     ║
# ║                                                                          ║
# ║  3. Then run:  bash run_all.sh                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# ── Pre-flight: require .env to exist ─────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "ERROR: .env file not found."
    echo ""
    echo "  Create it from the template and fill in your values before running:"
    echo "    cp .env.example .env"
    echo "    nano .env"
    echo ""
    exit 1
fi

# Load .env (export all variables so child scripts inherit them)
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ── Optional overrides via environment variables ───────────────────────────
# You can override INGEST_LIMIT and BATCH_SIZE at the command line:
#   INGEST_LIMIT=20000 BATCH_SIZE=128 bash run_all.sh
INGEST_LIMIT="${INGEST_LIMIT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Semantic Search Engine - Full Pipeline"
echo "  .env loaded from: $ENV_FILE"
echo "  INGEST_LIMIT: $INGEST_LIMIT"
echo "  BATCH_SIZE:   $BATCH_SIZE"
echo ""

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/scripts/*.sh

echo "Step 1/5: Install dependencies"
bash "$SCRIPT_DIR/scripts/01_install_deps.sh"

echo ""
echo "Step 2/5: Configure PostgreSQL"
bash "$SCRIPT_DIR/scripts/02_setup_postgres.sh"

echo ""
echo "Step 3/5: Apply database schema"
bash "$SCRIPT_DIR/scripts/03_init_schema.sh"

echo ""
echo "Step 4/5: Ingest dataset (limit=$INGEST_LIMIT)"
LIMIT=$INGEST_LIMIT BATCH_SIZE=$BATCH_SIZE \
    bash "$SCRIPT_DIR/scripts/04_ingest.sh"

echo ""
echo "Step 5/5: Run evaluation"
bash "$SCRIPT_DIR/scripts/06_evaluate.sh"

echo ""
echo "Pipeline complete."
echo "To search interactively: bash scripts/05_search.sh"
echo "Single query:            bash scripts/05_search.sh \"your query\""
