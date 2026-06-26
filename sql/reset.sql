-- Drops everything created by schema.sql, for a clean slate.
-- Not run automatically by any script -- run manually, e.g.:
--   psql "$DATABASE_URL" -f sql/reset.sql
-- Then re-apply with: python scripts/setup_db.py

-- must go first, it depends on both tables below
drop view if exists documents_with_embeddings;

-- drop before `documents` (holds the FK to it)
drop table if exists embeddings;

drop table if exists documents;
