"""API HTTP da Busca Híbrida CorPop-Saúde.

Expõe a busca híbrida (léxica via Meilisearch + semântica via ChromaDB)
para ser consumida por um site. Reaproveita search.pesquisar().

Rodar (dev):  uv run uvicorn api:app --reload --port 8000
Rodar (prod): uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

Pré-requisitos da BUSCA (/buscar): Meilisearch rodando, .env configurado e
embedding.py já executado. Faltando qualquer um deles a API sobe assim mesmo,
porque /simplicidade e /escopos só leem os .txt do corpus — é /buscar (e
/embed) que passa a responder 503, com o motivo. /health e /config dizem se a
busca está disponível.
"""
import os
import sys
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
    #
    # Falhar aqui NÃO impede a API de subir: metade das rotas (/simplicidade,
    # /escopos) só lê os .txt do corpus e não depende de motor nenhum, então
    # derrubar o processo por falta de Meilisearch ou de índice tiraria do ar
    # também o que funcionaria. O motivo fica em search.indisponivel e vira 503
    # nas rotas que precisam de busca.
    try:
        search.init()
    except search.BuscaIndisponivel as exc:
        print(f"⚠️  Busca indisponível: {exc}\n"
              f"   A API vai subir mesmo assim; /simplicidade e /escopos "
              f"funcionam normalmente.", file=sys.stderr)
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
    """Status do processo e, à parte, o da busca.

    `status` é do serviço HTTP (se responde, está ok). `busca` diz se os
    motores subiram — é o que distingue "API no ar sem índice" de "API fora do
    ar", que da parte do cliente pareceriam a mesma coisa.
    """
    return {
        "status": "ok",
        "busca": "indisponivel" if search.indisponivel else "ok",
        "motivo": search.indisponivel,
    }


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
        raise HTTPException(
            status_code=503,
            detail=f"Modelo não carregado. {search.indisponivel or ''}".strip())
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
        # Mesmo com a busca fora, estes três vêm preenchidos (search.init() os
        # define antes de carregar qualquer coisa), então o slider da página
        # continua certo e ela ainda consegue avisar que a busca está fora.
        "busca_disponivel": search.indisponivel is None,
        "motivo_indisponivel": search.indisponivel,
    }


@app.get("/escopos")
def escopos():
    """Os dois eixos de recorte aceitos em /simplicidade.

    Existe para o frontend montar os filtros sem hardcodar nada: hoje são bulas
    de hipertensão e oncologia, e quando entrarem outros gêneros de documento
    (termos de consentimento, por exemplo) eles aparecem aqui sozinhos, sem
    mudança no cliente.

      - `escopos`: a árvore temática. Cada nó tem id ("bula/oncologia"),
        rótulo, nível, filhos, contagem de documentos e
        `documentos_por_proveniencia` — que é o que permite ao cliente
        desabilitar cruzamentos vazios. O id vazio significa "todo o corpus".
      - `proveniencias`: o eixo de como o lado simplificado foi produzido
        (validado por linguistas × gerado por IA sem revisão). Vazio =
        qualquer, o que MISTURA as duas referências.
    """
    return {"escopos": simplicidade.escopos(),
            "proveniencias": simplicidade.proveniencias(),
            "corpus": simplicidade.resumo_corpus(None)}


@app.get("/simplicidade")
def comparar_simplicidade(
    a: str = Query(..., min_length=1, description="Primeiro termo (pode ser multipalavra)"),
    b: str = Query(..., min_length=1, description="Segundo termo"),
    escopo: str | None = Query(None, description="Recorte temático (ver /escopos): "
                                                 "ex. 'bula', 'bula/oncologia'. "
                                                 "Omitido: todo o corpus"),
    proveniencia: str | None = Query(None, description="Procedência do lado simplificado "
                                                       "(ver /escopos): 'humana' ou 'ia'. "
                                                       "Omitido: qualquer"),
):
    """Compara dois termos e aponta o mais simples, com base no corpus paralelo.

    Não usa os motores de busca: a evidência é a frequência relativa de cada
    termo nas bulas simplificadas × originais (ver simplicidade.py). `escopo`
    (tema) e `proveniencia` restringem a evidência, e se cruzam: o mesmo par de
    termos pode trocar de lugar entre um recorte e outro. O recorte usado, seu
    tamanho e a procedência da evidência voltam no campo `corpus` da resposta —
    incluindo `proveniencia_mista`, que avisa quando o score sai de uma
    referência híbrida (parte validada por humanos, parte gerada por IA).
    """
    try:
        return simplicidade.comparar(a, b, escopo, proveniencia)
    except ValueError as exc:  # termo sem letras, recorte inexistente ou vazio
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
    except search.BuscaIndisponivel as exc:
        # Falta de infraestrutura (sem .env, sem índice), não erro de consulta:
        # a mensagem já diz o que fazer, então vai inteira para o cliente.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # Meilisearch fora do ar no meio da consulta, etc.
        raise HTTPException(status_code=503, detail=f"Erro ao consultar os motores de busca: {exc}")
    return _jsonificar(resultados)
