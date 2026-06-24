#!/usr/bin/env bash
# 06_evaluate.sh
# Run the evaluation suite: latency, overlap, and score comparison.
# Outputs CSV and PNG charts to data/
set -euo pipefail

echo "  Step 5 - Evaluation & Analysis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/scripts/evaluate.py"

echo ""
echo "OK Evaluation complete. Results in data/"
