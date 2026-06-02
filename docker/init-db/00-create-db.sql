-- ──────────────────────────────────────────────────────────────
-- Auto-created by Docker entrypoint (POSTGRES_USER/POSTGRES_DB).
-- PostgreSQL 16 + asyncpg works best with these settings.
-- ──────────────────────────────────────────────────────────────

-- Ensure the public schema is usable (default anyway, but explicit)
GRANT ALL PRIVILEGES ON SCHEMA public TO nexus;
GRANT ALL PRIVILEGES ON DATABASE nexus TO nexus;

-- Raise the default work_mem for analytics / heavy queries
ALTER ROLE nexus SET work_mem = '64MB';

-- pgvector extension for in-database vector similarity search
-- Used by the memory system for fast ANN (approximate nearest neighbor)
-- queries on embeddings instead of loading all memories into Python.
CREATE EXTENSION IF NOT EXISTS vector;
