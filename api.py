"""API HTTP da Busca Híbrida CorPop-Saúde.

Expõe a busca híbrida (léxica via Meilisearch + semântica via ChromaDB)
para ser consumida por um site. Reaproveita search.pesquisar().

Rodar (dev):  uv run uvicorn api:app --reload --port 8000
Rodar (prod): uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
Pré-requisitos: Meilisearch rodando, .env configurado e embedding.py já executado.
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import search
import simplicidade
from modelos import encode_docs

load_dotenv()

# Origens liberadas no CORS, separadas por vírgula. Vazio (default) = nenhum
# middleware de CORS: é o caso de produção, em que o nginx serve o front e a API
# na MESMA origem e requisição cross-origin nenhuma acontece. Em dev, sirva o
# front por outra porta e ponha essa origem aqui (ex.: http://localhost:5500).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo e conecta nos motores uma única vez, antes de atender
    # requisições — o carregamento leva de segundos a minutos na CPU. Modelo e
    # coleção saem do .env (EMBED_MODEL/CHROMA_COLLECTION), com os defaults
    # definidos em search.py: o servidor roda um modelo mais leve que a máquina
    # de desenvolvimento, então isso é configuração, não constante de código.
    search.init()
    yield


app = FastAPI(title="Busca Híbrida CorPop-Saúde", lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["*"],
    )


def _jsonificar(resultados: dict) -> dict:
    """Garante que os valores sejam serializáveis em JSON (ex.: distâncias)."""
    chroma = resultados["chroma"]
    return {
        "meili": resultados["meili"],
        "sinonimos": list(resultados.get("sinonimos", [])),
        "chroma": {
            "ids": list(chroma["ids"]),
            "docs": list(chroma["docs"]),
            "metadatas": list(chroma["metadatas"]),
            "distances": [float(d) for d in chroma["distances"]],
            "scores": [1 - float(d) for d in chroma["distances"]],
            # Os dois registros completos de cada medicamento (para comparação).
            "originais": list(chroma.get("originais", [])),
            "simplificadas": list(chroma.get("simplificadas", [])),
            # Frase da versão simplificada validada que mais casou com a query.
            "trechos_match": list(chroma.get("trechos_match", [])),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


class LoteTextos(BaseModel):
    """Lote de textos a embeddar. O teto de 256 limita o tamanho da resposta
    JSON (256 × 768 floats ≈ 4MB) e o pico de RAM do encode."""
    textos: list[str] = Field(..., min_length=1, max_length=256)


def _exigir_loopback(request: Request):
    """Recusa a chamada se ela não veio da própria máquina.

    /embed existe para a indexação reaproveitar o modelo já carregado aqui, em
    vez de carregar uma segunda cópia (o que não caberia nos 4GB do servidor).
    Não é rota pública: o nginx não a encaminha, e este guarda é a segunda
    barreira. A presença de X-Forwarded-* indica que a chamada passou por um
    proxy — nesse caso o IP de origem seria o do proxy, não o do cliente real.
    """
    origem = request.client.host if request.client else ""
    via_proxy = any(h.lower().startswith("x-forwarded-") for h in request.headers)
    if origem not in ("127.0.0.1", "::1") or via_proxy:
        raise HTTPException(status_code=403, detail="Endpoint restrito a chamadas locais.")


@app.post("/embed")
def embed(lote: LoteTextos, request: Request):
    """Embeda um lote de textos com o modelo já carregado (lado DOCUMENTO).

    Devolve também o nome do modelo: quem indexa PRECISA conferir que é o mesmo
    que a coleção-alvo espera — vetores de modelos diferentes no mesmo índice
    produzem resultados silenciosamente errados.
    """
    _exigir_loopback(request)
    if search.model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    vetores = encode_docs(search.model, lote.textos).tolist()
    return {"modelo": search.model_atual, "vetores": vetores}


@app.get("/config")
def config():
    """Configuração efetiva da busca, para o frontend não hardcodar nada.

    O limiar de corte depende do modelo (cada um espalha os cossenos numa faixa
    própria), então o slider da página precisa perguntar em vez de assumir.
    """
    return {
        "modelo": search.model_atual,
        "colecao": search.colecao_atual,
        "min_score_padrao": search.min_score_atual,
    }


@app.get("/simplicidade")
def comparar_simplicidade(
    a: str = Query(..., min_length=1, description="Primeiro termo (pode ser multipalavra)"),
    b: str = Query(..., min_length=1, description="Segundo termo"),
):
    """Compara dois termos e aponta o mais simples, com base no corpus paralelo.

    Não usa os motores de busca: a evidência é a frequência relativa de cada
    termo nas bulas simplificadas × originais (ver simplicidade.py).
    """
    try:
        return simplicidade.comparar(a, b)
    except ValueError as exc:  # termo sem nenhuma letra (ex.: só pontuação)
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/buscar")
def buscar(
    q: str = Query(..., min_length=1, description="Termo de busca"),
    n: int = Query(5, ge=1, le=50, description="Número de resultados por motor"),
    min_score: float | None = Query(None, ge=0.0, le=1.0,
                                    description="Score mínimo (similaridade cosseno no "
                                                "ChromaDB). Omitido: usa o limiar do "
                                                "modelo em uso (ver /config)"),
    somente_simplificada: bool = Query(False, description="Restringe a busca semântica aos "
                                       "chunks da bula simplificada (linguagem acessível)"),
    somente_original: bool = Query(False, description="Restringe a busca semântica aos "
                                   "chunks da bula original (texto técnico)"),
):
    try:
        resultados = search.pesquisar(q, n, min_score, somente_simplificada,
                                      somente_original)
    except Exception as exc:  # Meilisearch fora do ar, índice ausente, etc.
        raise HTTPException(status_code=503, detail=f"Erro ao consultar os motores de busca: {exc}")
    return _jsonificar(resultados)
