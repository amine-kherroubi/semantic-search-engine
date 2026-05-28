#!/usr/bin/env bash
# 02_setup_postgres.sh
# Create the PostgreSQL database and user for the project.
# Reads from .env if present, otherwise uses defaults.
set -euo pipefail

echo "Step 2 - PostgreSQL database setup"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Load .env if it exists
if [ -f "$ENV_FILE" ]; then
    # Export only the DB_* variables
    set -a
    # shellcheck disable=SC1090
    source <(grep -E '^DB_' "$ENV_FILE")
    set +a
fi

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-semantic_search}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

echo "  Host:     $DB_HOST:$DB_PORT"
echo "  Database: $DB_NAME"
echo "  User:     $DB_USER"

# Ensure PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -q; then
    echo "[!] PostgreSQL is not running. Starting ..."
    sudo service postgresql start
    sleep 2
fi

# Create user (ignore error if already exists)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER';" \
    | grep -q 1 \
    || sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"

# Create database (ignore error if already exists)
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';" \
    | grep -q 1 \
    || sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Grant privileges
sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
sudo -u postgres psql -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"

echo ""
echo "OK Database '$DB_NAME' ready."
echo "  Next: run  bash scripts/03_init_schema.sh"
