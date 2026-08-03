-- Product Recommendation System — PostgreSQL schema (Phase 1)
-- Normalized schema for users, items, and interactions.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    external_id     INTEGER NOT NULL UNIQUE,
    age             SMALLINT,
    gender          CHAR(1),
    occupation      VARCHAR(50),
    zip_code        VARCHAR(10),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS items (
    id              SERIAL PRIMARY KEY,
    external_id     INTEGER NOT NULL UNIQUE,
    title           VARCHAR(255) NOT NULL,
    release_date    DATE,
    video_release   DATE,
    imdb_url        VARCHAR(255),
    genres          TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TYPE signal_type AS ENUM ('explicit', 'implicit');
CREATE TYPE interaction_type AS ENUM ('rating', 'click', 'view', 'purchase');

CREATE TABLE IF NOT EXISTS interactions (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id             INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    rating              SMALLINT CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5)),
    signal_type         signal_type NOT NULL,
    interaction_type    interaction_type NOT NULL DEFAULT 'rating',
    interacted_at       TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, item_id, interaction_type, interacted_at)
);

CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_item_id ON interactions(item_id);
CREATE INDEX IF NOT EXISTS idx_interactions_interacted_at ON interactions(interacted_at);
CREATE INDEX IF NOT EXISTS idx_interactions_signal_type ON interactions(signal_type);

-- ETL audit log for row counts before/after cleaning
CREATE TABLE IF NOT EXISTS etl_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID NOT NULL DEFAULT uuid_generate_v4(),
    dataset         VARCHAR(100) NOT NULL,
    table_name      VARCHAR(50) NOT NULL,
    stage           VARCHAR(20) NOT NULL,  -- raw, cleaned, loaded
    row_count       BIGINT NOT NULL,
    null_flags      JSONB DEFAULT '{}',
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_etl_runs_run_id ON etl_runs(run_id);
