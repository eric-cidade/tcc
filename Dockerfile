# Imagem da API da Busca Híbrida CorPop-Saúde (FastAPI + e5-small + ChromaDB).
# O Meilisearch roda em container próprio — ver docker-compose.yml e deploy/COOLIFY.md.
FROM ghcr.io/astral-sh/uv:0.9.11 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/.cache/huggingface \
    CHROMA_PATH=/data/chroma \
    # Sem isso o torch abre uma thread por core da máquina inteira e disputa CPU
    # com os outros serviços do servidor.
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Dependências primeiro: camada que só é refeita quando o lock muda. --no-dev deixa
# o matplotlib de fora; o torch vem do índice CPU (ver [tool.uv.sources] no
# pyproject.toml), sem a stack CUDA.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /root/.cache/uv

# Modelo embutido na imagem: o container sobe sem depender do HuggingFace, e um
# restart não baixa ~470MB de novo. Passa por carregar_modelo para baixar
# exatamente os arquivos que a API vai usar.
ARG EMBED_MODEL=intfloat/multilingual-e5-small
COPY modelos.py ./
RUN python -c "from modelos import carregar_modelo; carregar_modelo('${EMBED_MODEL}')"
# Daqui em diante, nada de rede para o Hub: o cache já tem tudo.
ENV HF_HUB_OFFLINE=1

COPY . .
# Proteção contra CRLF caso o script seja editado no Windows.
RUN sed -i 's/\r$//' deploy/entrypoint.sh \
    && useradd --system --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /data/chroma \
    && chown -R app:app /app /data

USER app
EXPOSE 8000

# start_period longo: no 1º boot o entrypoint indexa o corpus antes de subir a API.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["bash", "deploy/entrypoint.sh"]
