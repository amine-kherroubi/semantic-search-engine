#!/usr/bin/env bash
# =============================================================================
# 03_init_schema.sh
# Apply the SQL schema (enable pgvector, create tables & indexes).
# =============================================================================
set -euo pipefail

echo "════════════════════════════════════════"
echo "  Step 3 — Apply database schema"
echo "════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/scripts/setup_db.py"

echo ""
echo "✓ Schema applied."
echo "  Next: run  bash scripts/04_ingest.sh"
