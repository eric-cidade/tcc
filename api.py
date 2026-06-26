"""API HTTP da Busca Híbrida CorPop-Saúde.

Expõe a busca híbrida (léxica via Meilisearch + semântica via ChromaDB)
para ser consumida por um site. Reaproveita search.pesquisar().

Rodar:  uv run uvicorn api:app --reload --port 8000
Pré-requisitos: Meilisearch rodando, .env configurado e embedding.py já executado.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo BAAI/bge-m3 e conecta nos motores uma única vez,
    # antes de começar a atender requisições.
    search.init()
    yield


app = FastAPI(title="Busca Híbrida CorPop-Saúde", lifespan=lifespan)

# Em desenvolvimento liberamos qualquer origem; em produção restrinja
# allow_origins para o domínio do site.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            # Score do reranker (~[0,1]) quando rerank=True; vazio caso contrário.
            "rerank_scores": [float(s) for s in chroma.get("rerank_scores", [])],
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/buscar")
def buscar(
    q: str = Query(..., min_length=1, description="Termo de busca"),
    n: int = Query(5, ge=1, le=50, description="Número de resultados por motor"),
    min_score: float = Query(0.4, ge=0.0, le=1.0,
                             description="Score mínimo (ChromaDB): cosseno sem rerank, "
                                         "score do reranker com rerank"),
    rerank: bool = Query(False, description="Reordena os resultados semânticos com o "
                                            "cross-encoder bge-reranker-v2-m3 (2º estágio)"),
    somente_simplificada: bool = Query(False, description="Restringe a busca semântica aos "
                                       "chunks da bula simplificada (linguagem acessível)"),
    min_score_rerank: float | None = Query(None, ge=0.0, le=1.0,
                                           description="Limiar no modo rerank (escala do "
                                           "reranker). Default baixo se omitido"),
):
    try:
        resultados = search.pesquisar(q, n, min_score, rerank, somente_simplificada,
                                      min_score_rerank)
    except Exception as exc:  # Meilisearch fora do ar, índice ausente, etc.
        raise HTTPException(status_code=503, detail=f"Erro ao consultar os motores de busca: {exc}")
    return _jsonificar(resultados)
