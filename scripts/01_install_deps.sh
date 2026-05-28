#!/usr/bin/env bash
# =============================================================================
# 01_install_deps.sh
# Install system packages, PostgreSQL, pgvector, and Python dependencies.
# Run once on a fresh Ubuntu machine.
# =============================================================================
set -euo pipefail

echo "════════════════════════════════════════"
echo "  Step 1 — System & Python dependencies"
echo "════════════════════════════════════════"

# ── System packages ────────────────────────────────────────────────────────
echo "[1/4] Updating apt …"
sudo apt-get update -qq

echo "[2/4] Installing PostgreSQL 16 + dev headers …"
sudo apt-get install -y --no-install-recommends \
    postgresql-16 \
    postgresql-server-dev-16 \
    postgresql-client-16 \
    build-essential \
    git \
    curl \
    python3-dev \
    python3-pip \
    python3-venv

# ── pgvector ──────────────────────────────────────────────────────────────
echo "[3/4] Building & installing pgvector …"
PGVEC_DIR="/tmp/pgvector"
if [ -d "$PGVEC_DIR" ]; then
    rm -rf "$PGVEC_DIR"
fi
git clone --depth 1 https://github.com/pgvector/pgvector.git "$PGVEC_DIR"
cd "$PGVEC_DIR"
make -j"$(nproc)"
sudo make install
cd -
echo "  pgvector built."

# ── Python virtual environment ────────────────────────────────────────────
echo "[4/4] Creating Python venv and installing packages …"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

python3 -m venv "$PROJECT_ROOT/.venv"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

pip install --upgrade pip wheel -q
pip install -r "$PROJECT_ROOT/requirements.txt"

echo ""
echo "✓ All dependencies installed."
echo "  Activate the venv with:  source .venv/bin/activate"
