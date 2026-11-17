#!/usr/bin/env bash
# Entrypoint do container da API (docker-compose.yml / Coolify).
#
# 1. Espera o Meilisearch.
# 2. Coleção vazia (1º boot ou volume novo): indexa o corpus inteiro no ChromaDB
#    e no Meilisearch, com o modelo carregado aqui mesmo. Leva alguns minutos com
#    o e5-small, uma vez só: o resultado fica no volume /data/chroma.
#    Coleção já populada: só repopula o Meilisearch (~1s, sem modelo), o que
#    cobre o caso de o volume do Meili ter sido perdido. É idempotente (upsert).
# 3. Sobe o uvicorn. A indexação já terminou e liberou a memória do modelo, então
#    nunca há duas cópias dele ao mesmo tempo.
set -euo pipefail
cd /app

: "${MEILI_URL:?defina MEILI_URL}"
: "${CHROMA_COLLECTION:?defina CHROMA_COLLECTION}"
: "${EMBED_MODEL:?defina EMBED_MODEL}"

echo "==> Esperando o Meilisearch em $MEILI_URL"
for i in $(seq 1 60); do
    if python -c "import urllib.request,os; urllib.request.urlopen(os.environ['MEILI_URL'].rstrip('/')+'/health', timeout=3)" 2>/dev/null; then
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "Meilisearch não respondeu em 2 min — subindo a API mesmo assim (a busca dará 503)." >&2
    fi
    sleep 2
done

colecao_populada() {
    python - <<'PY'
import os, sys
import chromadb
cliente = chromadb.PersistentClient(path=os.environ["CHROMA_PATH"])
try:
    n = cliente.get_collection(os.environ["CHROMA_COLLECTION"]).count()
except Exception:
    n = 0
print(f"    coleção {os.environ['CHROMA_COLLECTION']}: {n} chunks")
sys.exit(0 if n > 0 else 1)
PY
}

if colecao_populada; then
    echo "==> Índice vetorial presente — sincronizando só o Meilisearch"
    python embedding.py --apenas-meili
else
    echo "==> Índice vetorial vazio — indexando o corpus com $EMBED_MODEL (alguns minutos)"
    # EMBED_API vazio: a API ainda não existe, então o modelo é carregado aqui.
    EMBED_API= python embedding.py
fi

echo "==> Subindo a API"
exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
