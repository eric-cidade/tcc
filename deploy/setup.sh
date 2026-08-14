#!/usr/bin/env bash
# Provisionamento do servidor (Ubuntu 24.04) para a Busca Hibrida CorPop-Saude.
#
# Idempotente: pode rodar de novo sem quebrar o que ja existe. NAO indexa e NAO
# inicia a API — isso e feito depois, quando o .env estiver preenchido (ver
# deploy/README.md).
#
# Uso:  sudo bash /opt/corpop-saude/deploy/setup.sh
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/corpop-saude}
APP_USER=${APP_USER:-corpop}
# Confira a ultima versao em https://github.com/meilisearch/meilisearch/releases
MEILI_VERSION=${MEILI_VERSION:-v1.22.3}
# Swap adicional. O bge-m3 sozinho ocupa ~2,3GB dos 4GB de RAM; sem folga de swap
# um pico de indexacao derruba a maquina inteira.
SWAP_EXTRA_GB=${SWAP_EXTRA_GB:-3}
SWAPFILE=/swapfile.corpop

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
aviso() { printf '\033[1;33m [!] %s\033[0m\n' "$*"; }

# WSL e conteineres nao controlam o proprio swap nem o firewall do host: la essas
# etapas sao puladas, para o script poder ser ensaiado antes de ir ao servidor.
EM_WSL=false
grep -qi microsoft /proc/version 2>/dev/null && EM_WSL=true
$EM_WSL && aviso "Rodando em WSL: swap e firewall serao pulados (o host os gerencia)."

if [[ $EUID -ne 0 ]]; then
    echo "Rode como root: sudo bash $0" >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/api.py" ]]; then
    echo "Repositorio nao encontrado em $APP_DIR (esperava $APP_DIR/api.py)." >&2
    echo "Clone o projeto ali antes de rodar este script." >&2
    exit 1
fi

log "1/8  Pacotes do sistema"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx curl ca-certificates python3.12 python3.12-venv ufw

log "2/8  Usuario de servico ($APP_USER)"
if id "$APP_USER" &>/dev/null; then
    echo "    usuario ja existe"
else
    useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi

log "3/8  Swap adicional (${SWAP_EXTRA_GB}GB)"
if $EM_WSL; then
    echo "    pulado (no WSL o swap vem do .wslconfig do Windows)"
elif [[ -f "$SWAPFILE" ]]; then
    echo "    $SWAPFILE ja existe"
else
    # fallocate falha em alguns FS; dd e o fallback universal.
    fallocate -l "${SWAP_EXTRA_GB}G" "$SWAPFILE" 2>/dev/null \
        || dd if=/dev/zero of="$SWAPFILE" bs=1M count=$((SWAP_EXTRA_GB * 1024)) status=none
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE" >/dev/null
    # Provedores que ja gerenciam swap (ou kernels sem suporte a swapfile) fazem
    # o swapon falhar; nesse caso desfaz e segue, em vez de abortar tudo.
    if swapon "$SWAPFILE" 2>/dev/null; then
        grep -q "^$SWAPFILE " /etc/fstab || echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
    else
        aviso "swapon falhou — seguindo sem swap adicional. Confira 'free -h'."
        rm -f "$SWAPFILE"
    fi
fi
# Swap so como rede de seguranca: com swappiness alto o kernel manda os pesos do
# modelo para o disco e cada busca fica lentissima.
if sysctl -q -w vm.swappiness=10 2>/dev/null; then
    grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
else
    aviso "nao foi possivel ajustar vm.swappiness (normal em conteiner/WSL)."
fi
free -h | sed 's/^/    /'

log "4/8  Meilisearch $MEILI_VERSION"
if command -v meilisearch &>/dev/null && [[ "$(meilisearch --version)" == *"${MEILI_VERSION#v}"* ]]; then
    echo "    ja instalado na versao certa"
else
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/meilisearch.deb" \
        "https://github.com/meilisearch/meilisearch/releases/download/${MEILI_VERSION}/meilisearch.deb"
    dpkg -i "$tmp/meilisearch.deb" >/dev/null
    rm -rf "$tmp"
fi
id meilisearch &>/dev/null || useradd --system --home-dir /var/lib/meilisearch \
    --shell /usr/sbin/nologin meilisearch
install -d -o meilisearch -g meilisearch -m 750 /var/lib/meilisearch

log "5/8  Ambiente Python (uv + dependencias de producao)"
if ! command -v uv &>/dev/null; then
    curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
fi
install -d -o "$APP_USER" -g "$APP_USER" "$APP_DIR/.cache"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
# --no-dev: sem matplotlib (so avaliar.py usa). O torch vem do indice CPU, fixado
# no pyproject.toml — o wheel default arrastaria a stack CUDA inteira.
sudo -u "$APP_USER" env HOME="$APP_DIR" UV_CACHE_DIR="$APP_DIR/.cache/uv" \
    uv sync --no-dev --project "$APP_DIR"

log "6/8  .env"
if [[ -f "$APP_DIR/.env" ]]; then
    echo "    ja existe — preservado"
else
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chave=$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 43)
    sed -i "s|^MEILI_MASTER_KEY=.*|MEILI_MASTER_KEY=$chave|" "$APP_DIR/.env"
    sed -i "s|^MEILI_URL=.*|MEILI_URL=http://127.0.0.1:7700|" "$APP_DIR/.env"
    sed -i "s|^EMBED_BATCH=.*|EMBED_BATCH=8|" "$APP_DIR/.env"
    aviso "Gerado $APP_DIR/.env com uma master key nova."
    aviso "CONFIRA EMBED_MODEL e CHROMA_COLLECTION antes de indexar."
fi
# O systemd do Meilisearch le o MESMO .env (a master key precisa bater dos dois
# lados). Grupo `meilisearch` + 640 da leitura para ele sem ACL nenhuma: dono
# (corpop) le e escreve, grupo le, resto nao ve a chave.
chown "$APP_USER:meilisearch" "$APP_DIR/.env"
chmod 640 "$APP_DIR/.env"

log "7/8  Servicos (systemd) e nginx"
install -m 644 "$APP_DIR/deploy/meilisearch.service" /etc/systemd/system/meilisearch.service
install -m 644 "$APP_DIR/deploy/corpop-saude-api.service"     /etc/systemd/system/corpop-saude-api.service
install -m 644 "$APP_DIR/deploy/nginx-corpop-saude.conf"      /etc/nginx/sites-available/corpop-saude
ln -sfn /etc/nginx/sites-available/corpop-saude /etc/nginx/sites-enabled/corpop-saude
# O site default responde na porta 80 e ganharia do nosso por ser o primeiro.
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl daemon-reload
systemctl enable meilisearch nginx >/dev/null
systemctl restart meilisearch nginx

log "8/8  Firewall"
if $EM_WSL || [[ "${PULAR_FIREWALL:-}" == "1" ]]; then
    echo "    pulado (quem filtra e o host)"
else
    # 7700 (Meili) e 8000 (uvicorn) ficam em loopback e nao precisam de regra.
    # A porta do SSH sai da config REAL do sshd: assumir 22 e ligar o ufw numa
    # maquina com SSH em outra porta te tranca do lado de fora do servidor.
    ssh_port=$(sshd -T 2>/dev/null | awk '/^port /{print $2; exit}' || true)
    ssh_port=${ssh_port:-22}
    echo "    liberando SSH na porta $ssh_port"
    ufw allow "$ssh_port/tcp" >/dev/null
    ufw allow 80/tcp >/dev/null
    ufw --force enable >/dev/null
    ufw status | sed 's/^/    /'
fi

cat <<FIM

  Provisionamento concluido.

  Proximos passos (ver deploy/README.md):
    1. Revise  $APP_DIR/.env  (EMBED_MODEL / CHROMA_COLLECTION)
    2. Indexe: sudo -u $APP_USER env HOME=$APP_DIR uv run --project $APP_DIR python embedding.py
    3. Suba:   sudo systemctl enable --now corpop-saude-api

  A API NAO foi iniciada: sem indice ela sobe e falha nas consultas.
FIM
