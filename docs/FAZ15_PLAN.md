# Faz 1.5 — Haber + gelişmiş teknik (plan ve durum)

Bu doküman, [PokieTicker](https://github.com/owengetinfo-design/PokieTicker) `layer1` / `features` mantığının AI Broker’a **kontrollü entegrasyonu** ve sonraki adımları özetler. Upstream klonları: [external/README.md](../external/README.md).

## Hedef

- **Haber:** Tek Groq çağrısında çoklu makale skoru (ilgili / duygu / özet / up-down gerekçesi), PokieTicker Layer 1 ile aynı JSON şeması.
- **Teknik:** yfinance OHLCV üzerinde PokieTicker **fiyat** feature set’inin son satırı (haber sütunları şimdilik 0 veya yok — haber skorları ayrı pipeline).
- **Kaynak:** Finnhub `company-news` (isteğe bağlı `FINNHUB_API_KEY`) veya `POST /internal/news/batch` ile gönderilen makale listesi (aynı JSON gövdesi).

## Fazlar

| Alt-faz | İçerik | Durum |
|--------|--------|--------|
| **1.5-a** | `news_pipeline` (batch prompt + parse), `finnhub_news` çekme, `/internal/news/*`, `/news SYMBOL` | Bu sürümde uygulanır |
| **1.5-b** | `/internal/analyze` veya `/analyze` ile haber özetini birleştirme; TOON ile prompt sıkıştırma (`[integrations]`) | **Uygulandı** — `app/services/analysis_runner.py`, `USE_TOON_PROMPTS`, `/analyze SYM news` / `news full` |
| **1.5-c** | TrendRadar entegrasyonu — **iptal / kapsam dışı** (2026-04); ek makale kaynakları `POST /internal/news/batch` ile | Karar kaydı: `CHANGELOG.md` |
| **1.5-d** | Truth Social Trump postları (`trump_monitor`) — SSE/WS, Groq impact + görsel, Telegram eşik, SQL şema (`sql/schemas/trump_posts.sql`) | **Repo içi uygulandı** — `TRUTH_SOCIAL_ACCESS_TOKEN` vb.; Supabase yazımı Faz 2 |

## API özeti

- `POST /internal/news/batch` — `{ "symbol", "articles": [{ "title", "description?" }] }` → Groq analiz satırları.
- `GET /internal/news/analyze?symbol=AMD&limit=20` — Finnhub’tan çek + batch (anahtar yoksa 503).

## Telegram

- `/news AMD` — son günlerin Finnhub haberleri + batch özet (Groq gerekli).

## Riskler

- Finnhub ücretsiz katman kota / gecikme.
- Groq çıktısı bazen geçersiz JSON; parser kısmi kurtarma dener.
- XGBoost / tam 31 feature + haber birleşik eğitim **bilerek** Faz 2’ye bırakıldı.
