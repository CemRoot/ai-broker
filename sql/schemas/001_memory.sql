-- Faz 2: AI Broker Hafıza ve Supabase Şemaları (pgvector destekli)

-- 1. pgvector eklentisini etkinleştir
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. trade_memories Tablosu (Ana RAG Hafızası)
CREATE TABLE IF NOT EXISTS trade_memories (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    memory_type VARCHAR(20) NOT NULL,   -- LESSON / SUCCESS / WARNING
    context TEXT NOT NULL,              -- Anının içeriği (İngilizce)
    outcome VARCHAR(10),                -- WIN / LOSS / MISSED / OPEN
    pnl_percent FLOAT,                  -- Gerçekleşen kâr/zarar %
    opportunity_cost FLOAT,             -- Masada bırakılan $
    confidence_score FLOAT,             -- Ajanın analiz güveni (0-1)
    timeframe VARCHAR(10),              -- 1D / 4H / 1H vb.
    embedding VECTOR(768),              -- nomic-embed-text vektörleri
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- pgvector HNSW İndeksi (Kosinüs benzerliği için)
CREATE INDEX IF NOT EXISTS idx_trade_memories_embedding_hnsw
ON trade_memories USING hnsw (embedding vector_cosine_ops);

-- 3. daily_reports Tablosu (Docker-safe kalıcı günlükler)
CREATE TABLE IF NOT EXISTS daily_reports (
    id BIGSERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    content TEXT NOT NULL,              -- MD formatında metin
    report_type VARCHAR(20),            -- DAILY / TRADE / LESSON
    ticker VARCHAR(10),                 -- İlgili hisse (trade raporu için)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(report_date, report_type, ticker)
);

-- 4. trump_posts — tek tanım: ``sql/schemas/trump_posts.sql`` (bu dosyadan hemen sonra uygula).
--    Böylece ERD / DBML ile çakışma olmaz; ``ON CONFLICT (post_id)`` TrumpMonitor ile uyumludur.

-- 5. Paper trading (sanal bakiye, portföy, işlem geçmişi): ``sql/schemas/002_paper_trading.sql``
--    Bu dosyadan sonra uygulanmalıdır (Faz 3 PR #1). Eski basit ``paper_trades`` tanımı kaldırıldı.

-- 6. punishment_log Tablosu (Ceza Zinciri mekanizması)
CREATE TABLE IF NOT EXISTS punishment_log (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    penalty_type VARCHAR(20) NOT NULL,  -- CONFIDENCE_DROP / COOLDOWN
    reason TEXT NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 7. match_trade_memories (Cosine Similarity Arama Fonksiyonu)
CREATE OR REPLACE FUNCTION match_trade_memories(
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  p_ticker varchar DEFAULT NULL
)
RETURNS TABLE (
  id bigint,
  ticker varchar,
  memory_type varchar,
  context text,
  outcome varchar,
  pnl_percent float,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.ticker,
    m.memory_type,
    m.context,
    m.outcome,
    m.pnl_percent,
    1 - (m.embedding <=> query_embedding) AS similarity
  FROM trade_memories m
  WHERE 1 - (m.embedding <=> query_embedding) > match_threshold
    AND (p_ticker IS NULL OR m.ticker = p_ticker)
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
