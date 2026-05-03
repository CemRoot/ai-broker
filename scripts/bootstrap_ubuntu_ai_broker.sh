#!/usr/bin/env bash
# Bootstrap Ubuntu host for AI Broker (Docker stack per docker-compose.yml).
#
# Run ON THE VPS as root (Ubuntu 22.04/24.04 cloud image):
#   curl -fsSL ... | sudo bash
#   OR: sudo bash scripts/bootstrap_ubuntu_ai_broker.sh
#
# Does NOT install host-level Ollama — compose already runs ollama/ollama:latest.
# After cloning the repo and placing .env:
#   cd /home/ubuntu/ai-broker && docker compose up --build -d
#   docker compose exec ollama ollama pull nomic-embed-text
#
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "🚀 AI Broker — Ubuntu host bootstrap"

apt-get update -y
apt-get upgrade -y -o Dpkg::Options::="--force-confold"

apt-get install -y curl git wget nano ufw ca-certificates

# Docker Engine + Compose v2 (Ubuntu Noble+: apt package; yoksa GitHub plugin)
apt-get install -y docker.io
set +e
apt-get install -y docker-compose-v2
_compose_pkg=$?
set -e
if ! docker compose version >/dev/null 2>&1; then
  PA=""
  case "$(uname -m)" in
    x86_64) PA=x86_64 ;;
    aarch64) PA=aarch64 ;;
    *) echo "docker compose: bilinmeyen mimari; paket kurulumu rc=$_compose_pkg" >&2; exit 1 ;;
  esac
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -fsSL "https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-${PA}" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

systemctl enable docker
systemctl start docker

# Prefer ubuntu; fall back to opc (Oracle Cloud) or skip if missing
DOCKER_USER="ubuntu"
if id ubuntu &>/dev/null; then
  DOCKER_USER="ubuntu"
elif id opc &>/dev/null; then
  DOCKER_USER="opc"
fi
usermod -aG docker "$DOCKER_USER" || true

# Firewall — SSH first
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
# Optional: direct access to FastAPI (prefer Cloudflare Tunnel only in prod)
ufw allow 8000/tcp
ufw --force enable

# cloudflared — correct arch for AMD64 (Hetzner) vs ARM64 (OCI Ampere)
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) CF_DEB_ARCH="amd64" ;;
  aarch64|arm64) CF_DEB_ARCH="arm64" ;;
  *)
    echo "Unsupported machine architecture: $ARCH" >&2
    exit 1
    ;;
esac

CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_DEB_ARCH}.deb"
curl -fsSL "$CF_URL" -o /tmp/cloudflared.deb
dpkg -i /tmp/cloudflared.deb || apt-get install -y -f

PROJECT_DIR="/home/${DOCKER_USER}/ai-broker"
mkdir -p "$PROJECT_DIR"
chown -R "${DOCKER_USER}:${DOCKER_USER}" "$PROJECT_DIR"

echo ""
echo "✅ Host hazır."
echo "   Docker:     $(docker --version)"
echo "   Compose:    $(docker compose version)"
echo "   Cloudflared:$(cloudflared --version 2>/dev/null || echo ' n/a')"
echo ""
echo "Sonraki adımlar (${DOCKER_USER} olarak):"
echo "  1. Repo klonla → ${PROJECT_DIR}"
echo "  2. cp .env.example .env && düzenle"
echo "  3. docker compose up --build -d"
echo "  4. docker compose exec ollama ollama pull nomic-embed-text"
echo "  5. cloudflared tunnel run … (veya FAZ4_DEPLOY.md)"
echo ""
echo "Not: Host üzerinde systemd Ollama kurulmadı — compose içindeki Ollama kullanılır."
