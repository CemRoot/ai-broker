# Harici kaynaklar (upstream klonları)

Bu dizin, AI Broker’a **adaptasyon** için kullanılan **iki** açık kaynağın **yerel aynalarını** tutar. Klonlar `.gitignore` ile repoya girmez; geliştirici makinesinde aşağıdaki komutlarla güncellenir.

## Klonlama (tek sefer veya yenileme)

```bash
mkdir -p external && cd external

rm -rf PokieTicker toon  # isteğe bağlı temiz başlangıç

git clone --depth 1 https://github.com/owengetinfo-design/PokieTicker.git PokieTicker
git clone --depth 1 https://github.com/toon-format/toon.git toon
```

Kaynaklar:

- [PokieTicker](https://github.com/owengetinfo-design/PokieTicker) — haber pipeline + 31 feature + benzerlik.
- [toon-format/toon](https://github.com/toon-format/toon) — TOON spesifikasyonu ve TS referans; Python’da **`toon-format`** PyPI paketi (`pyproject.toml` → `[integrations]`).

---

## Entegrasyon planı (faz sırası)

**Repo içi uygulama durumu:** Faz 1.5-a (batch haber + Finnhub + extended technical) → [docs/FAZ15_PLAN.md](../docs/FAZ15_PLAN.md).

### Faz A — Referans + hizalama (düşük risk)

| Kaynak | Ne okunur | AI Broker çıktısı |
|--------|-----------|-------------------|
| **toon** | `SPEC.md`, `docs/guide/llm-prompts.md`, `docs/reference/spec.md` | Mevcut prompt’larda TOON blokları için şablon; kodda `import toon_format` (`encode`/`decode`) ile pilot kullanım. |
| **PokieTicker** | `backend/pipeline/layer0.py`, `layer1.py`, `alignment.py` | `app/services/news_pipeline.py` taslağı: batch boyutu, prompt şekli, metin kırpma mantığı (LLM = Groq, DB = sonra Supabase). |
| **PokieTicker** | `backend/ml/features.py` (ve gerekirse `features_v2.py`) | `app/tools/technical.py` genişletmesi: 31 `FEATURE_COLS` ile hizalama (OHLCV kaynağı: yfinance). |
| **PokieTicker** | `backend/pipeline/similarity.py`, `backend/ml/similar.py` | Faz 2 öncesi notlar; üretimde pgvector + `nomic-embed-text` (pickle yok). |

### Faz B — Kod taşıma (orta risk)

1. **news_pipeline:** `layer1.py` mantığını kopyalamak yerine **fonksiyon düzeyinde** port et; Anthropic batch → Groq chat; SQLite → önce bellek/liste, sonra Supabase.
2. **technical:** `features.py` içinden saf pandas/numpy özellik hesapları + `FEATURE_COLS` listesi; T212 ticker map ile sembol uyumu.
3. **TOON:** Analiz ve haber batch promptlarında JSON yerine TOON tabloları (token ölçümü ile doğrula).

---

## Hızlı dosya haritası (klon içi)

**PokieTicker** (`PokieTicker/`)

- `backend/pipeline/layer1.py` — batch haber + sentiment şeması  
- `backend/pipeline/layer0.py` — kural filtresi  
- `backend/ml/features.py` — 31 feature  
- `backend/pipeline/similarity.py` — benzerlik (referans; pickle üretimde yok)  
- `backend/api/` — FastAPI router’lar (API sözleşmesi fikri için)

**toon** (`toon/`)

- `SPEC.md` — format  
- `packages/toon/src/` — TS referans uygulama  
- Python uygulama: `uv pip install -e ".[integrations]"` → `toon_format`

---

## Güncelleme

```bash
cd external/PokieTicker && git pull --ff-only
cd ../toon && git pull --ff-only
```

Shallow clone kullandıysan ve `git pull` sorun çıkarırsa: dizini silip yeniden `git clone --depth 1` yap.
