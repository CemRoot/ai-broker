# Faz 4 — Üretim dağıtımı (AI Broker)

Bu doküman `AI_BROKER_PROJECT.md` Faz 4 hedefleriyle uyumludur: Docker, compose, tünel, kapasite ve gözlemlenebilirlik notları.

## Ubuntu sunucu — tek seferlik host kurulumu

Fresh VPS’te (Ubuntu 22.04/24.04) kök olarak:

```bash
# Repoyu önce klonlamadan da çalıştırabilirsiniz: ham dosyayı çekin veya scp ile gönderin.
sudo bash scripts/bootstrap_ubuntu_ai_broker.sh
```

Script: **`docker.io`** + **`docker compose`**, **UFW** (22/80/443/8000), mimariye göre **cloudflared** `.deb` (**amd64** veya **arm64**), proje dizini (`ubuntu` veya `opc` kullanıcısı). **Host üzerinde systemd Ollama kurmaz** — `docker-compose.yml` içindeki **`ollama`** servisi kullanılır (11434 çakışması olmasın diye). Sonra: repo + `.env`, `docker compose up --build -d`, `docker compose exec ollama ollama pull nomic-embed-text`.

## Docker imajı

- **Dockerfile:** multi-stage; çalışma imajı `python:3.12-slim-bookworm` + `uv sync --frozen --no-dev` ile üretilen `.venv`.
- **Boyut:** `pandas` / `pandas-ta` / `yfinance` nedeniyle **~180MB** hedefi genelde gerçekçi değildir; tipik sıkıştırılmamış imaj **~400–700MB** bandında olabilir. İnceleme: `docker images ai-broker-ai-broker`.
- **Kullanıcı:** container içinde `UID 10001` (`appuser`); root ile çalışmaz.

```bash
docker build -t ai-broker:local .
docker run --rm -p 8000:8000 --env-file .env ai-broker:local
```

## docker-compose (uygulama + Ollama)

```bash
docker compose up --build -d
curl -s http://127.0.0.1:8000/health
```

- **`.env` değişince:** `docker compose restart ai-broker` çoğu zaman **host `.env`’deki yeni değişkenleri konteynere taşımaz** (env ilk `up` anında sabitlenir). Yenilemek için: `docker compose up -d --force-recreate ai-broker` (veya `down` + `up`).

- **`OLLAMA_BASE_URL`:** compose dosyası `http://ollama:11434` yazar; host `.env` içindeki `localhost` üzerine yazar.
- **Modeller (ilk kurulum):** Ollama konteynerinde:
  - `docker compose exec ollama ollama pull deepseek-r1:14b`
  - `docker compose exec ollama ollama pull nomic-embed-text`
- **GPU:** `docker-compose.yml` içindeki NVIDIA örneği yorumunu açın (Linux + `nvidia-container-toolkit` gerekir). macOS’ta genelde CPU inference.
- **Sorun giderme — `ollama` unhealthy:** Resmi imajda `curl` yok; eski compose’ta `curl` ile healthcheck kullanılıyorsa konteyner sürekli **unhealthy** kalır ve `ai-broker` `depends_on` yüzünden ayağa kalkmaz. Güncel `docker-compose.yml` **`ollama list`** kullanır. Doğrulama: `docker inspect … --format '{{json .State.Health}}'` ve `docker logs … ollama`.
- **`git pull` — “local changes would be overwritten … docker-compose.yml”:** VPS’te bir kerelik `sed`/elle yapılan compose değişiklikleri commit’li değilse merge bloklanır. **Çözüm (sadece bu repo için yerel diff’i atmak OK):** `git stash push -m compose docker-compose.yml && git pull && git stash drop` **veya** `git fetch origin && git reset --hard origin/main`. Sonra `docker compose down && docker compose up -d --build`.

## Cloudflare Tunnel (Telegram webhook)

1. [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) kurun.
2. Örnek: `cloudflared tunnel --url http://127.0.0.1:8000` (hızlı test) veya named tunnel + DNS.
3. `.env`: `TELEGRAM_WEBHOOK_URL=https://<senin-domainin>` (sonunda `/telegram/webhook` **ekleme** — uygulama path’i kendisi birleştirir; bkz. `app/main.py`).
4. `TELEGRAM_WEBHOOK_SECRET` doldurun; Telegram’da secret token ayarıyla uyumlu olsun.
5. **Public Hostname tablosundaki “Origin configurations” sütununda `0` görünmesi** çoğu zaman **ek origin policy sayısı**dır; **connector’ın çalışmadığı** anlamına gelmez. Doğrulama: dış ağdan `curl -sS https://<hostname>/health` → HTTP **200** ve JSON; veya Tunnel sayfasında **Status: Healthy** / log’da `Registered tunnel connection`.

## OCI ARM (ör. 4 vCPU / 24GB)

- **Groq birincil LLM** ise ağır yük çoğunlukla dışarıdadır; bu makine API + bot + DB bağlantısı için genelde yeterli.
- **Ollama `deepseek-r1:14b` + nomic-embed-text** aynı host’ta: RAM yoğun; eşzamanlı yüksek trafikte **CPU inference yavaş** kalır. Üretimde:
  - Groq’u ana yol tutun,
  - veya ayrı GPU’lu bir Ollama host’u düşünün,
  - veya daha küçük Ollama modeli (dokümandaki Faz 0 kararına aykırı olmadan CEO onayıyla).

## Token / istek ölçümü

- **Groq:** `GET /internal/usage` (günlük sayaç); ayrıca [Groq Console](https://console.groq.com) kullanımı.
- **FMP / Finnhub:** sağlayıcı panelleri; uygulama içinde aggregator yok (ileride Prometheus / structured metrics eklenebilir).
- **T212:** istemci zaten ~1 req/s throttle kullanır (`app/services/t212/`).

## 7/24 güvenilirlik (checklist)

- `restart: unless-stopped` (compose) veya orchestrator health policy.
- `GET /health` — `memory_db`, `telegram` özetleri.
- Log toplama (journald, Docker logging driver, veya harici APM) — repo dışı yapılandırma.
- Düzenli yedek: Supabase (CEO konsolu); yerel Ollama volume: `ollama_data` volume yedekleri.

## Güvenlik

- **`.env` asla imaja kopyalanmaz** (`.dockerignore`). Sadece çalışma anında `env_file` / `-e` ile verilir.
- Üretimde API anahtarları platform secret store (Fly, K8s Secret, vb.) tercih edilir.

## Sunucu seçimi — “Token derdi olmadan, lokal LLM, 7/24” (CTO önerisi, 2026-05-03)

CEO kuralı: **cloud LLM token kotasıyla uğraşmak istemiyoruz** → birincil LLM **Ollama `deepseek-r1:14b`** + embedder **`nomic-embed-text`** aynı sunucuda. Bot 7/24 ayakta, ev makinesi (M4 Pro Max) sürekli açık tutulmayacak.

### Kapasite tahmini (tek host)

| Bileşen | Hot RAM | Disk | Notlar |
| --- | --- | --- | --- |
| `deepseek-r1:14b` (q4) | ~10–14 GB | ~9 GB | İlk yüklemede yavaş; ayakta tutulursa hızlı |
| `nomic-embed-text` | ~0.6 GB | ~0.3 GB | Sürekli yüklü kalmasında sorun yok |
| FastAPI + asyncpg + worker | ~0.5–1 GB | <100 MB | tek `--workers 1` |
| Pandas/yfinance/pgvector client | ~0.3–0.5 GB | — | Ölçek küçük, tek bot kullanıcı |
| **Toplam (rahat)** | **~16 GB RAM** | **~10 GB** | Eşzamanlı 1 cycle + Telegram |

### Seçenek tablosu (üst → ideal, alt → ekonomik)

| # | Sağlayıcı / paket | CPU / GPU | RAM | Aylık (€) | LLM hızı (deepseek 14B) | Yorum |
| --- | --- | --- | --- | --- | --- | --- |
| **1 (önerilen)** | **Hetzner GEX44** (RTX 4000 SFF Ada, 20 GB VRAM) | 16-core Ryzen + GPU | 64 GB | **~189** | **30–60 tok/s** | GPU varlığı sayesinde PaperAgent cycle’ları 5–10 sn. 70B’ye ileride çıkma marjı (`deepseek-r1:32b` rahat sığar). EU lokasyonu, T212’ye düşük gecikme. |
| 2 | **Scaleway Apple Silicon (M4 Pro mac mini, 24 GB)** | M4 Pro 12-core | 24 GB unified | ~130 | **30–50 tok/s** | Dev makineyle **birebir parite** (Metal). cloudflared tünel kolay; Docker yerine native runtime. Mac stok + zon yoğunluğu sıkıntı olabilir. |
| 3 | **Hetzner AX42** (Ryzen 7 7700, 64 GB DDR5, **GPU yok**) | 8-core CPU | 64 GB | ~48 | **3–7 tok/s** | Telegram chat ve Trump alert için idare eder; PaperAgent cycle 60–120 sn olur. Pratik bir “bütçe sunucusu”. |
| 4 | **Hetzner CCX23 / OCI ARM (4 vCPU, 24 GB)** | 4 vCPU | 16–24 GB | ~26 / 0 (OCI Free) | **<2 tok/s** | 14B sıkışır. Sadece embedder + Groq fallback için tutulabilir. CEO kuralıyla uyumsuz. |
| 5 | **Kendi M4 Pro Max + Cloudflare Tunnel** | M4 Pro Max | 36–64 GB | 0 (elektrik) | aynı dev | UPS şart; fiziksel kesintiler riskli. Geçici çözüm. |
| 6 | **RunPod / Vast.ai spot GPU** | RTX 3090/4090 (paylaşımlı) | değişken | ~0.20–0.50 €/saat | yüksek | 7/24 stabilite **yok** (preempt). PaperAgent için uygun değil. |

### Karar matrisi

- **Para öncelik değilse:** **Seçenek 1 (Hetzner GEX44)** — tek host, üst sınırı yüksek, dev/prod mantıken aynı.
- **Dev parite önemli (Metal davranışı bire bir):** **Seçenek 2 (Scaleway Mac mini M4 Pro)**.
- **Bütçe önceliği + cycle yavaşlığı kabul:** **Seçenek 3 (Hetzner AX42)** + sadece kritik anlarda Groq fallback’i devre dışı bırakıp Ollama’ya bırak.
- **Geçici köprü:** **Seçenek 5** (ev makinesi + Cloudflare Tunnel) — 1–2 hafta MVP doğrulaması için yeterli.

### Geçiş kontrol listesi (Hetzner GEX44 örneği)

1. Ubuntu 24.04 LTS + Docker Engine + `nvidia-container-toolkit`.
2. `git clone` + `.env` (host’ta üret, repoya commit etme).
3. `docker compose up -d --build`.
4. `docker compose exec ollama ollama pull deepseek-r1:14b nomic-embed-text`.
5. Cloudflare Tunnel (`cloudflared` + named tunnel) → Telegram webhook’a `https://...` ver.
6. systemd unit: `docker compose up -d` reboot’ta otomatik.
7. Log rotasyonu: `daemon.json` → `log-driver=json-file`, `max-size=20m`, `max-file=5` (compose içinde de tanımlı).
8. Backup: `ollama_data` volume snapshot + Supabase otomatik.
9. Monitoring: `/health` → uptime servisi (UptimeRobot / Hetzner Cloud Monitoring).
10. CEO erişimi: `TELEGRAM_ALLOWED_USER_IDS` ile sadece sizin chat ID.

> **Not:** GEX44 olmadan 14B + Trump + PaperAgent + chat’i tek tek anlık tutmak teknik olarak **mümkün** ama gecikme CEO'nun “openclaw” beklentisinden uzak kalır. CPU-only bir başlangıç planlanırsa `deepseek-r1:7b`'ye düşürmek (CEO onayıyla) daha gerçekçi.
