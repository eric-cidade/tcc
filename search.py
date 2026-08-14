import argparse
import os
import re
import sys
import time
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import util

from caminhos import CHROMA_PATH
from sinonimos import expandir_query, sinonimos_da_query
from modelos import (carregar_modelo, encode_query, encode_docs, min_score_padrao,
                     EMBED_MODEL_PADRAO, CHROMA_COLLECTION_PADRAO)

# Globais preenchidas por init(). Permanecem None até a inicialização ser
# chamada (pelo bloco __main__ da CLI ou pelo startup da API em api.py).
model = None
meili_client = None
meili_index = None
chroma_client = None
chroma_coll = None
# Configuração efetiva, preenchida por init(). Exposta para a API informar o
# frontend (endpoint /config) — o limiar depende do modelo, então a página não
# tem como saber sozinha.
model_atual = None
colecao_atual = None
min_score_atual = None



def init(embed_model=None, chroma_collection=None):
    """Carrega o modelo de embedding e conecta nos motores de busca.

    Idempotente: chamadas repetidas não recarregam o modelo. Deve ser
    chamada uma única vez antes de pesquisar().

    `embed_model` e `chroma_collection` (se passados) têm prioridade sobre as
    variáveis de ambiente EMBED_MODEL/CHROMA_COLLECTION — a api.py usa isso para
    fixar bge-m3 + coleção estrutural. O modelo e a coleção precisam CASAR com o
    que foi usado na indexação.
    """
    global model, meili_client, meili_index, chroma_client, chroma_coll
    global model_atual, colecao_atual, min_score_atual
    if model is not None:
        return

    load_dotenv()
    meili_key = os.getenv('MEILI_MASTER_KEY')
    # 127.0.0.1 e NÃO 'localhost': o Meilisearch escuta só em IPv4, e com
    # 'localhost' o cliente tenta ::1 primeiro e espera o timeout — medido em
    # 2,03s por consulta contra 0,009s aqui, 226x mais lento, sem nada aparecer
    # nos logs porque o próprio Meili reporta processingTimeMs=0.
    meili_url = os.getenv('MEILI_URL', 'http://127.0.0.1:7700')
    if not meili_key:
        print("❌ Erro: MEILI_MASTER_KEY não encontrada no arquivo .env", file=sys.stderr)
        sys.exit(1)

    print("Conectando aos motores de busca...", file=sys.stderr)
    embed_model = embed_model or os.getenv('EMBED_MODEL') or EMBED_MODEL_PADRAO
    chroma_collection = (chroma_collection or os.getenv('CHROMA_COLLECTION')
                         or CHROMA_COLLECTION_PADRAO)
    model = carregar_modelo(embed_model)
    # Cada modelo tem sua faixa de cosseno: o limiar acompanha o modelo, senão
    # trocar de modelo silenciosamente passa a filtrar demais ou de menos.
    model_atual = embed_model
    colecao_atual = chroma_collection
    min_score_atual = min_score_padrao(embed_model)

    meili_client = meilisearch.Client(meili_url, meili_key)
    meili_index = meili_client.index('corpop_saude')

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_coll = chroma_client.get_or_create_collection(
        name=chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    # get_or_create devolve uma coleção VAZIA quando o nome não existe, e o erro
    # só apareceria lá na frente como um TypeError obscuro do Chroma. Falhar aqui,
    # dizendo o que fazer, é o que separa "índice não construído" de "bug".
    if chroma_coll.count() == 0:
        print(f"❌ Erro: a coleção '{chroma_collection}' está vazia (modelo "
              f"'{embed_model}').\n   Rode a indexação antes de buscar:\n"
              f"   EMBED_MODEL={embed_model} CHROMA_COLLECTION={chroma_collection} "
              f"uv run python embedding.py", file=sys.stderr)
        sys.exit(1)


# --- Função de Pesquisa Híbrida ---
def pesquisar(query, limite=1, min_score=None, somente_simplificada=False,
              somente_original=False):
    """Busca híbrida. `min_score=None` usa o limiar próprio do modelo carregado."""
    if model is None:
        init()
    if min_score is None:
        min_score = min_score_atual

    # A. Busca Léxica (Meilisearch)
    # matchingStrategy="all": exige que TODAS as palavras do termo estejam no
    # documento (o padrão "last" iria descartando palavras até achar algo).
    # Pedimos destaque/recorte e a posição dos matches para conseguir devolver
    # exatamente o termo (e o trecho ao redor) que casou em cada bula.
    res_meili = meili_index.search(query, {
        "limit": limite,
        "matchingStrategy": "all",
        "attributesToHighlight": ["nome", "conteudo_original", "conteudo_simplificado"],
        "attributesToCrop": ["conteudo_original", "conteudo_simplificado"],
        "cropLength": 30,
        "showMatchesPosition": True,
        "highlightPreTag": "<mark>",
        "highlightPostTag": "</mark>",
    })
    for hit in res_meili['hits']:
        _anexar_match_lexico(hit)

    # B. Busca Semântica (ChromaDB)
    # Expande a query com sinônimos conhecidos antes do embedding (ex.:
    # "cefaleia" -> "cefaleia dor de cabeça"), melhorando o recall do modelo,
    # que sozinho é fraco em sinônimo curto.
    query_sem = expandir_query(query)
    query_vec = encode_query(model, query_sem).tolist()
    # Cada bula está indexada em vários chunks, então pedimos bem mais que
    # `limite` para que os melhores chunks cubram pelo menos `limite`
    # medicamentos distintos depois de agregar.
    n_consulta = min(max(limite * 20, 60), chroma_coll.count())
    # Opcionalmente restringe a busca a um único registro: só a simplificada
    # (linguagem acessível) ou só a original (texto técnico). Se ambos forem
    # pedidos, a simplificada tem precedência (evita filtro contraditório).
    if somente_simplificada:
        where = {"registro": "simplificada"}
    elif somente_original:
        where = {"registro": "original"}
    else:
        where = None
    res_chroma = chroma_coll.query(
        query_embeddings=[query_vec],
        n_results=n_consulta,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    ids = res_chroma['ids'][0] if res_chroma['ids'] else []
    docs = res_chroma['documents'][0] if res_chroma['documents'] else []
    metas = res_chroma['metadatas'][0] if res_chroma['metadatas'] else []
    dists = res_chroma['distances'][0] if res_chroma['distances'] else []

    # Agrega os chunks por medicamento: guarda só o melhor chunk (menor
    # distância) de cada um. Os resultados já vêm ordenados por distância,
    # então o primeiro chunk visto de cada medicamento é o melhor.
    melhor_por_id = {}
    for i, d, m, dist in zip(ids, docs, metas, dists):
        base = str(m.get("id"))
        if base not in melhor_por_id:
            melhor_por_id[base] = (i, d, m, dist)

    # Com espaço cosseno, score = 1 - distância ∈ [-1, 1]. Filtra abaixo do
    # limiar e mantém os `limite` melhores medicamentos.
    filtered = [
        (i, d, m, dist)
        for (i, d, m, dist) in sorted(melhor_por_id.values(), key=lambda t: t[3])
        if (1 - dist) >= min_score
    ][:limite]

    # Cada medicamento foi indexado em dois registros (..._orig e ..._simp).
    # A query devolve só o que casou; aqui buscamos os dois textos de cada
    # medicamento presente nos resultados, para permitir comparação lado a lado.
    textos = _buscar_ambos_registros([str(m.get("id")) for _, _, m, _ in filtered])

    def _txt(meta, chave):
        return textos.get(str(meta.get("id")), {}).get(chave)

    simplificadas = [_txt(t[2], "simplificada") for t in filtered]

    # O "trecho que casou" é a sentença mais próxima da query DENTRO do chunk
    # que casou (e não da bula inteira — re-encodar todo o texto de cada
    # resultado era o maior custo de CPU), pelo cosseno do modelo de embedding.
    _t = time.perf_counter()
    trechos_match = [_frase_mais_proxima(query_vec, doc) for (_, doc, _, _) in filtered]
    print(f"[tempo] trechos: {len(filtered)} resultados em "
          f"{time.perf_counter() - _t:.2f}s", file=sys.stderr)

    return {
        "meili": res_meili['hits'],
        # Sinônimos conhecidos acrescentados à busca (ex.: ['dor de cabeça'] para
        # a query 'cefaleia'). Valem para as duas lanes: o Meili expande via seus
        # synonyms, e a semântica via expansão de query.
        "sinonimos": sinonimos_da_query(query),
        "chroma": {
            "ids": [t[0] for t in filtered],
            "docs": [t[1] for t in filtered],
            "metadatas": [t[2] for t in filtered],
            "distances": [t[3] for t in filtered],
            "originais": [_txt(t[2], "original") for t in filtered],
            "simplificadas": simplificadas,
            "trechos_match": trechos_match,
        }
    }


def _anexar_match_lexico(hit):
    """Anexa ao hit do Meilisearch o termo e o trecho que casaram.

    - `termos_match`: lista de {'campo', 'termo'} com os termos literais casados,
      extraídos de `_matchesPosition` (deduplicados).
    - `trecho_match`: snippet com contexto (de `_formatted`, com <mark>…</mark>),
      preferindo o campo simplificado quando ele tiver match.
    """
    posicoes = hit.get("_matchesPosition", {}) or {}
    formatado = hit.get("_formatted", {}) or {}

    termos, vistos = [], set()
    for campo, matches in posicoes.items():
        valor = hit.get(campo)
        if not isinstance(valor, str):
            continue
        # As posições do Meilisearch (start/length) são offsets em BYTES UTF-8,
        # não índices de caractere — então fatiamos sobre os bytes do texto.
        valor_bytes = valor.encode("utf-8")
        for m in matches:
            ini = m.get("start", 0)
            termo = valor_bytes[ini:ini + m.get("length", 0)].decode("utf-8", "ignore").strip()
            chave = (campo, termo.lower())
            if termo and chave not in vistos:
                vistos.add(chave)
                termos.append({"campo": campo, "termo": termo})
    hit["termos_match"] = termos

    # Trecho com contexto: prioriza o simplificado (linguagem acessível).
    trecho = None
    for campo in ("conteudo_simplificado", "conteudo_original", "nome"):
        if campo in posicoes and formatado.get(campo):
            trecho = {"campo": campo, "texto": formatado[campo]}
            break
    hit["trecho_match"] = trecho


def _sentencas(texto):
    """Quebra `texto` em sentenças (descartando fragmentos curtos).

    Fallback: se NADA passar do filtro de tamanho — caso comum com o chunking
    estrutural, em que o chunk que casa é um item curto (ex.: 'Dor de cabeça;')
    —, devolve o próprio texto como uma sentença. Sem isso, o "trecho que casou"
    sairia vazio justamente para os chunks item-a-item.
    """
    partes = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", texto)
              if len(s.strip()) > 15]
    if partes:
        return partes
    limpo = texto.strip()
    return [limpo] if limpo else []


def _frase_mais_proxima(query_vec, texto):
    """Sentença de `texto` mais próxima da query → {'texto', 'score'} | None.

    As sentenças são o lado DOCUMENTO da comparação (o `query_vec` é o lado
    consulta), então vão por encode_docs — em modelos com prefixo, misturar os
    lados aqui deslocaria os vetores e escolheria a sentença errada.
    """
    if not texto:
        return None
    sentencas = _sentencas(texto)
    if not sentencas:
        return None
    sims = util.cos_sim(encode_docs(model, sentencas), [query_vec])[:, 0]   # (n,)
    melhor = int(sims.argmax())
    return {"texto": sentencas[melhor], "score": float(sims[melhor])}


def _buscar_ambos_registros(base_ids):
    """Para cada id de medicamento, devolve {'original': str|None, 'simplificada': str|None}.

    Cada bula vive fatiada em vários chunks no ChromaDB; aqui buscamos todos os
    chunks dos medicamentos pedidos e reconstruímos os textos completos,
    juntando-os na ordem de `chunk_idx`, para permitir a comparação lado a lado.
    """
    vistos = []
    for bid in base_ids:
        if bid and bid not in vistos:
            vistos.append(bid)
    if not vistos:
        return {}

    got = chroma_coll.get(where={"id": {"$in": vistos}}, include=["documents", "metadatas"])

    # Acumula (chunk_idx, texto) por medicamento e registro.
    partes = {}  # bid -> {'original': [...], 'simplificada': [...]}
    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
        bid = str(meta.get("id"))
        slot = "simplificada" if str(meta.get("registro", "")).startswith("simplific") else "original"
        partes.setdefault(bid, {"original": [], "simplificada": []})[slot].append(
            (meta.get("chunk_idx", 0), doc)
        )

    def _juntar(itens):
        return "\n".join(d for _, d in sorted(itens, key=lambda x: x[0])) if itens else None

    return {
        bid: {"original": _juntar(p["original"]), "simplificada": _juntar(p["simplificada"])}
        for bid, p in partes.items()
    }


# --- Interface de Terminal (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Busca Híbrida CorPop-Saúde (UFRGS)')
    parser.add_argument('query', type=str, help='Termo de busca')
    parser.add_argument('--n', type=int, default=1, help='Número de resultados')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Score mínimo (similaridade cosseno) para retornar um '
                             'resultado do ChromaDB (0.0–1.0). Default: o limiar '
                             'do modelo em uso (ver MODELOS em modelos.py).')
    parser.add_argument('--somente-simplificada', action='store_true',
                        help='Busca semântica só nos chunks da bula simplificada.')
    parser.add_argument('--somente-original', action='store_true',
                        help='Busca semântica só nos chunks da bula original (texto técnico).')

    args = parser.parse_args()
    init()
    resultados = pesquisar(args.query, args.n, args.min_score,
                           args.somente_simplificada, args.somente_original)

    print("\n" + "═"*50)
    print(f"RESULTADOS PARA: '{args.query}'")
    print("═"*50)

    # --- Exibição Meilisearch ---
    print("\n[MEILISEARCH - Busca por Palavra]")
    if resultados['meili']:
        for hit in resultados['meili']:
            # O 'id' aqui vem direto do documento indexado
            print(f"ID: {hit['id']} | Remédio: {hit['nome']}")
            termos = hit.get('termos_match') or []
            if termos:
                print("Termo(s) que casou: " +
                      ", ".join(f"{t['termo']} [{t['campo']}]" for t in termos))
            trecho = hit.get('trecho_match')
            if trecho:
                print(f"Trecho ({trecho['campo']}): ...{trecho['texto']}...")
            print(f"Simplificado: {hit['conteudo_simplificado'][:150]}...")
            print("-" * 20)
    else:
        print("Nenhum match exato encontrado.")

    # --- Exibição ChromaDB ---
    print("\n[CHROMADB - Busca Semântica]")
    if resultados['chroma']['ids']:
        # Iteramos usando o índice para combinar ID, Metadata e Documento
        for i in range(len(resultados['chroma']['ids'])):
            c_id = resultados['chroma']['ids'][i]
            c_meta = resultados['chroma']['metadatas'][i]
            c_doc = resultados['chroma']['docs'][i]
            c_dist = resultados['chroma']['distances'][i]
            # Chroma retorna distância (quanto menor, mais próximo); score = 1 - distância
            c_score = 1 - c_dist

            print(f"ID: {c_id} | Nome (Metadata): {c_meta.get('nome', 'N/A')}")
            print(f"Score: {c_score:.4f} (distância: {c_dist:.4f})")
            trecho = resultados['chroma']['trechos_match'][i]
            if trecho:
                print(f"Trecho acessível que casou (sim. {trecho['score']:.4f}): "
                      f"{trecho['texto']}")
            print(f"Sentido Encontrado: {c_doc[:150]}...")
            print("-" * 20)
    else:
        print("Nenhuma relação semântica encontrada.")
