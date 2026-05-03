-- T212 demo execution audit (apply after 003_paper_agent.sql)
-- Mirrors bot trades executed via Trading 212 API into paper_trades for RAG / invalidation / stats.

ALTER TABLE paper_trades
ADD COLUMN IF NOT EXISTS t212_order_id BIGINT,
ADD COLUMN IF NOT EXISTS execution_broker VARCHAR(16) NOT NULL DEFAULT 'supabase';

COMMENT ON COLUMN paper_trades.t212_order_id IS 'Trading 212 order id when execution_broker=t212';
COMMENT ON COLUMN paper_trades.execution_broker IS 'supabase = virtual PaperBroker ledger — t212 = order sent via Public API';
