#!/usr/bin/env bash
# Indexa bulas novas/revisadas no ChromaDB e no Meilisearch, COM A API NO AR.
#
# Rode depois de acrescentar arquivos em csv_metadata/ e atualizar os mapas
# remedios_*_map.csv. O site continua respondendo durante todo o processo.
#
# Como isso e possivel:
#   1. Modo INCREMENTAL — para cada medicamento, remove os chunks antigos dele e
#      grava os novos. A colecao nunca e apagada, entao a API nao perde o handle.
#   2. Indexador MAGRO — os embeddings sao pedidos ao endpoint /embed da API, que
#      ja tem o modelo residente. Sem isso seriam duas copias do modelo na
#      memoria (~4,2GB com bge-m3), o que nao cabe nos 4GB do servidor.
#
# Uso:  sudo bash /opt/corpop-saude/deploy/reindexar.sh
#       sudo bash /opt/corpop-saude/deploy/reindexar.sh --reconstruir   (ver abaixo)
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/corpop-saude}
APP_USER=${APP_USER:-corpop}
RECONSTRUIR=""
[[ "${1:-}" == "--reconstruir" ]] && RECONSTRUIR="--reconstruir"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
aviso() { printf '\033[1;33m [!] %s\033[0m\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
    echo "Rode como root: sudo bash $0" >&2
    exit 1
fi

# O Meilisearch precisa estar no ar: a indexacao escreve nos dois motores.
if ! systemctl is-active --quiet meilisearch; then
    log "Meilisearch parado — iniciando"
    systemctl start meilisearch
    sleep 3
fi

api_ativa=false
systemctl is-active --quiet corpop-saude-api && api_ativa=true

if [[ -n "$RECONSTRUIR" ]]; then
    # Reconstruir APAGA a colecao. A API guarda um handle pelo UUID dela; apagada,
    # todas as consultas passam a falhar com NotFoundError ate a API reiniciar —
    # mesmo depois de o indice novo ficar pronto. Por isso aqui a API e parada.
    aviso "Modo --reconstruir: a API sera parada (a colecao e apagada e recriada)."
    if [[ "$api_ativa" == true ]]; then
        log "Parando a API"
        systemctl stop corpop-saude-api
    fi
    # Sobe a API de volta mesmo se a indexacao falhar no meio.
    trap '[[ "$api_ativa" == true ]] && { log "Subindo a API de volta"; systemctl start corpop-saude-api; }' EXIT
else
    if [[ "$api_ativa" == true ]]; then
        log "API no ar — os embeddings virao dela (/embed), sem segunda copia do modelo"
    else
        aviso "API parada: o modelo sera carregado localmente (usa mais memoria)."
    fi
fi

log "Indexando (o progresso sai por medicamento)"
sudo -u "$APP_USER" env HOME="$APP_DIR" UV_CACHE_DIR="$APP_DIR/.cache/uv" \
    uv run --project "$APP_DIR" --no-dev python "$APP_DIR/embedding.py" $RECONSTRUIR

log "Indexacao concluida"
