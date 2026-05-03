-- Queued T212 orders (weekend / pending) until filled — poller writes paper_trades mirror on fill.
-- Apply after 004_t212_paper_execution.sql

CREATE TABLE IF NOT EXISTS paper_t212_pending_mirror (
    t212_order_id BIGINT PRIMARY KEY,
    yf_ticker VARCHAR(16) NOT NULL,
    action VARCHAR(4) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_poll_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_paper_t212_pending_poll ON paper_t212_pending_mirror (created_at);

COMMENT ON TABLE paper_t212_pending_mirror IS 'T212 order ids waiting for fill; T212MirrorPoller completes record_mirror_trade when filled';
COMMENT ON COLUMN paper_t212_pending_mirror.meta IS 'JSON: reasoning, stop_loss, target, invalidation_condition, chain_of_thought, cycle_event, emergency, avg_price_paid (SELL), source';
