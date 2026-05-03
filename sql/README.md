# SQL şemaları

## Uygulama sırası (Supabase / Postgres)

| Sıra | Dosya | Açıklama |
|------|--------|----------|
| 1 | `schemas/001_memory.sql` | `vector`, `trade_memories`, `daily_reports`, `punishment_log`, `match_trade_memories` |
| 2 | `schemas/trump_posts.sql` | `trump_posts` (TrumpMonitor) |
| 3 | `schemas/002_paper_trading.sql` | `paper_account`, `paper_portfolio`, `paper_trades` (başta DROP var — dikkat) |
| 4 | `schemas/003_paper_agent.sql` | `paper_trades` ek kolonlar + indeks |
| 5 | `schemas/004_t212_paper_execution.sql` | `paper_trades` T212 ayna kolonları |
| 6 | `schemas/005_t212_pending_mirror.sql` | `paper_t212_pending_mirror` — bekleyen T212 emirleri (poller → fill sonrası ayna) |

## Terminalden tek seferde uygula

```bash
# Repo kökünden (``SUPABASE_DB_URL`` .env’de dolu olmalı)
uv run python scripts/apply_sql_schemas.py
```

Mevcut `paper_*` verisini **silmeden** güncellemek için:

```bash
uv run python scripts/apply_sql_schemas.py --no-paper-drop
```

## ERD / şema görselleştirme

- **dbdiagram.io:** `schema.dbml` dosyasını içe aktar (Import). Tek görünür model; SQL’deki parçalı `ALTER` dosyaları burada birleşik tabloda gösterilir.
- **VS Code / Cursor:** `.vscode/extensions.json` içinde önerilen `matt-meyers.vscode-dbml` veya [dbdiagram VS Code](https://docs.dbdiagram.io/vs-code-extension) ile `schema.dbml` düzenleme.

## Çakışma notu

- `trump_posts` yalnızca `trump_posts.sql` içinde tanımlanır (`001_memory.sql` içinde tekrar yok).

## Eski ortam (legacy)

Daha önce yalnızca eski `trump_posts.sql` ile (`id BIGSERIAL` + `post_id UNIQUE`) kurduysan, şu anki canonical şema **`post_id` PRIMARY KEY** (TrumpMonitor `ON CONFLICT (post_id)` ile uyumlu). Tablo yapısı farklıysa elle migrasyon veya yeniden oluşturma gerekir.
