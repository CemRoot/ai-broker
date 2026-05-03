-- Faz 3: Otonom Paper Agent Şemaları
--
-- DROP: Var olan paper_* tablolarını sıfırlar (geliştirme / yeniden kurulum).
-- Veriyi korumak için: aşağıdaki üç DROP satırını yorum yap ve yalnızca yeni ortamda çalıştır.
--
DROP TABLE IF EXISTS paper_trades;
DROP TABLE IF EXISTS paper_portfolio;
DROP TABLE IF EXISTS paper_account;

-- 1. Paper Account (Sanal Bakiye — tek satır, id=1)
CREATE TABLE IF NOT EXISTS paper_account (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    balance DOUBLE PRECISION NOT NULL DEFAULT 20000.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO paper_account (id, balance) VALUES (1, 20000.0) ON CONFLICT (id) DO NOTHING;

-- 2. Paper Portfolio (Açık Pozisyonlar)
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    shares DOUBLE PRECISION NOT NULL,
    avg_cost DOUBLE PRECISION NOT NULL,
    current_value DOUBLE PRECISION,
    status VARCHAR(10) DEFAULT 'OPEN',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Paper Trades (İşlem Geçmişi ve Öğrenme Verisi)
CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(4) NOT NULL,         -- BUY / SELL
    shares DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    reasoning TEXT,                     -- İşlem gerekçesi
    macro_risk_score DOUBLE PRECISION,             -- İşlem anındaki makro risk skoru
    sentiment_score DOUBLE PRECISION,              -- İşlem anındaki sentiment
    pnl_percent DOUBLE PRECISION,                  -- Satışta gerçekleşmiş kâr/zarar %
    realized_pnl_usd DOUBLE PRECISION,            -- Satışta USD cinsinden gerçekleşmiş PnL
    was_punished BOOLEAN NOT NULL DEFAULT FALSE,
    lesson_written BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
