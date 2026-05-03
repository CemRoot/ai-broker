-- Faz 3: Paper Agent ek alanları (cycle log / risk fields)
-- Apply after: 001_memory.sql, 002_paper_trading.sql

ALTER TABLE paper_trades
ADD COLUMN IF NOT EXISTS stop_loss DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS target DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS invalidation_condition TEXT,
ADD COLUMN IF NOT EXISTS chain_of_thought TEXT,
ADD COLUMN IF NOT EXISTS cycle_event VARCHAR(32),
ADD COLUMN IF NOT EXISTS emergency BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker_created_at
ON paper_trades (ticker, created_at DESC);

