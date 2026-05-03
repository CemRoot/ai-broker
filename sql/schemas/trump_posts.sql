-- TrumpMonitor persistence (Faz 1.5-d / Faz 2).
-- Apply immediately after sql/schemas/001_memory.sql
-- Matches INSERT ... ON CONFLICT (post_id) in app/services/trump_monitor.py

CREATE TABLE IF NOT EXISTS trump_posts (
    post_id VARCHAR(50) PRIMARY KEY,
    post_text TEXT NOT NULL,
    image_analysis TEXT,
    posted_at TIMESTAMPTZ NOT NULL,
    impact_score DOUBLE PRECISION,
    sentiment VARCHAR(20),
    affected_sectors TEXT[],
    affected_tickers TEXT[],
    reasoning TEXT,
    telegram_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trump_posts_posted_at ON trump_posts (posted_at DESC);
