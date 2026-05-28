#!/usr/bin/env bash
# =============================================================================
# 05_search.sh
# Run the interactive search CLI (semantic + TF-IDF comparison).
#
# Usage:
#   bash scripts/05_search.sh                         # interactive mode
#   bash scripts/05_search.sh "your query here"       # single query
#   bash scripts/05_search.sh "your query" --no-tfidf # semantic only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/.venv/bin/activate"

if [ $# -gt 0 ]; then
    python "$PROJECT_ROOT/scripts/search.py" --query "$1" "${@:2}"
else
    python "$PROJECT_ROOT/scripts/search.py"
fi
