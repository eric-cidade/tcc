import argparse
import os
import re
import sys
import time
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from sinonimos import expandir_query, sinonimos_da_query

# Globais preenchidas por init(). Permanecem None até a inicialização ser
# chamada (pelo bloco __main__ da CLI ou pelo startup da API em api.py).
model = None
meili_client = None
meili_index = None
chroma_client = None
chroma_coll = None

# Reranker (cross-encoder) carregado sob demanda — só quando uma busca pede
# rerank=True. Modelo companheiro do bge-m3, multilíngue. Configurável por env:
# RERANKER_MODEL (ex.: 'BAAI/bge-reranker-base' é ~2× mais rápido na CPU) e
# RERANK_POOL_MAX (teto de medicamentos que passam pelo cross-encoder).
reranker = None
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
RERANK_POOL_MAX = int(os.getenv('RERANK_POOL_MAX', '30'))
# Limiar do modo rerank. A escala do cross-encoder é diferente do cosseno (bem
# mais separada: relevantes ~0.5+, ruído ~0.01), então o `min_score` do cosseno
# (~0.35) é alto demais aqui e cortaria quase tudo. Usamos um default baixo.
RERANK_MIN_SCORE = float(os.getenv('RERANK_MIN_SCORE', '0.1'))


def _get_reranker():
    """Carrega o cross-encoder de reordenação na primeira vez que for pedido."""
    global reranker
    if reranker is None:
        print(f"Carregando reranker {RERANKER_MODEL} (1ª vez baixa o modelo)...",
              file=sys.stderr)
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return reranker


def init():
    """Carrega o modelo de embedding e conecta nos motores de busca.

    Idempotente: chamadas repetidas não recarregam o modelo. Deve ser
    chamada uma única vez antes de pesquisar().
    """
    global model, meili_client, meili_index, chroma_client, chroma_coll
    if model is not None:
        return

    load_dotenv()
    meili_key = os.getenv('MEILI_MASTER_KEY')
    meili_url = os.getenv('MEILI_URL', 'http://localhost:7700')
    if not meili_key:
        print("❌ Erro: MEILI_MASTER_KEY não encontrada no arquivo .env", file=sys.stderr)
        sys.exit(1)

    print("Conectando aos motores de busca...", file=sys.stderr)
    model = SentenceTransformer('BAAI/bge-m3')
    model.max_seq_length = 8192

    meili_client = meilisearch.Client(meili_url, meili_key)
    meili_index = meili_client.index('corpop_saude')

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_coll = chroma_client.get_or_create_collection(
        name="corpop_saude",
        metadata={"hnsw:space": "cosine"},
    )


# --- Função de Pesquisa Híbrida ---
def pesquisar(query, limite=1, min_score=0.45, rerank=False, somente_simplificada=False,
              min_score_rerank=None):
    if model is None:
        init()

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
    # "cefaleia" -> "cefaleia dor de cabeça"), melhorando o recall do bge-m3,
    # que sozinho é fraco em sinônimo curto. O reranker segue usando a query
    # original (ele já entende o sinônimo).
    query_sem = expandir_query(query)
    query_vec = model.encode(query_sem).tolist()
    # Cada bula está indexada em vários chunks, então pedimos bem mais que
    # `limite` para que os melhores chunks cubram pelo menos `limite`
    # medicamentos distintos depois de agregar.
    n_consulta = min(max(limite * 20, 60), chroma_coll.count())
    # Opcionalmente restringe a busca aos chunks do registro simplificado
    # (linguagem acessível), ignorando os chunks da bula original.
    where = {"registro": "simplificada"} if somente_simplificada else None
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

    if rerank:
        # 2º estágio (retrieve→rerank) NO NÍVEL DE CHUNK. O bge-m3 é fraco em
        # sinônimo curto (ex.: "cefaleia" fica mais perto de "febre"/"náusea" do
        # que de "dor de cabeça"), então a chunk certa costuma ter cosseno baixo.
        # Por isso NÃO pré-agregamos por medicamento pelo cosseno — isso
        # descartaria a chunk que o reranker promoveria. Reranqueamos os top-K
        # chunks do cosseno (recall já cobre o termo) e só então agregamos por
        # medicamento, ficando com o melhor chunk de cada pelo score do reranker.
        pool = min(RERANK_POOL_MAX, len(docs))
        ce = _get_reranker()
        _t = time.perf_counter()
        scores = [float(s) for s in ce.predict([(query, d) for d in docs[:pool]])]
        print(f"[tempo] reranker: {pool} chunks em "
              f"{time.perf_counter() - _t:.2f}s", file=sys.stderr)

        melhor_por_id = {}
        for i in range(pool):
            base = str(metas[i].get("id"))
            rs = scores[i]
            if base not in melhor_por_id or rs > melhor_por_id[base][1]:
                melhor_por_id[base] = ((ids[i], docs[i], metas[i], dists[i]), rs)

        # Limiar próprio do reranker (escala diferente do cosseno). Usa o default
        # baixo RERANK_MIN_SCORE, a menos que a chamada sobrescreva via
        # `min_score_rerank` — assim o `min_score` (cosseno) do slider não corta
        # quase tudo no modo rerank.
        limiar = min_score_rerank if min_score_rerank is not None else RERANK_MIN_SCORE
        selecionados = [
            (tup, rs)
            for tup, rs in sorted(melhor_por_id.values(), key=lambda x: -x[1])
            if rs >= limiar
        ][:limite]
        filtered = [tup for tup, _ in selecionados]
        rerank_scores = [rs for _, rs in selecionados]
    else:
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
        rerank_scores = []

    # Cada medicamento foi indexado em dois registros (..._orig e ..._simp).
    # A query devolve só o que casou; aqui buscamos os dois textos de cada
    # medicamento presente nos resultados, para permitir comparação lado a lado.
    textos = _buscar_ambos_registros([str(m.get("id")) for _, _, m, _ in filtered])

    def _txt(meta, chave):
        return textos.get(str(meta.get("id")), {}).get(chave)

    simplificadas = [_txt(t[2], "simplificada") for t in filtered]

    # O "trecho que casou" é a sentença mais próxima da query DENTRO do chunk
    # que casou (e não da bula inteira — re-encodar todo o texto de cada
    # resultado era o maior custo de CPU). Com rerank, escolhemos a sentença com
    # o cross-encoder (o bge-m3 erraria, pegando "náusea" em vez de "dor de
    # cabeça"); sem rerank, usamos o cosseno do bge-m3.
    _t = time.perf_counter()
    if rerank:
        ce = _get_reranker()
        trechos_match = [_frase_mais_proxima_ce(query, doc, ce) for (_, doc, _, _) in filtered]
    else:
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
            # Score do reranker (~[0,1]) quando rerank=True; lista vazia quando
            # a busca usou só o cosseno. Paralelo a `ids`.
            "rerank_scores": rerank_scores,
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
    """Quebra `texto` em sentenças (descartando fragmentos curtos)."""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", texto)
            if len(s.strip()) > 15]


def _frase_mais_proxima(query_vec, texto):
    """Sentença de `texto` mais próxima da query (bge-m3) → {'texto', 'score'} | None."""
    if not texto:
        return None
    sentencas = _sentencas(texto)
    if not sentencas:
        return None
    sims = util.cos_sim(model.encode(sentencas), [query_vec])[:, 0]   # (n,)
    melhor = int(sims.argmax())
    return {"texto": sentencas[melhor], "score": float(sims[melhor])}


def _frase_mais_proxima_ce(query, texto, ce):
    """Como _frase_mais_proxima, mas pontua as sentenças com o cross-encoder.

    O bge-m3 erra em sinônimo curto (escolheria "náusea" para "cefaleia"); o
    reranker acerta "dor de cabeça". Usado quando a busca está com rerank=True.
    """
    if not texto:
        return None
    sentencas = _sentencas(texto)
    if not sentencas:
        return None
    scores = [float(s) for s in ce.predict([(query, s) for s in sentencas])]
    melhor = max(range(len(sentencas)), key=lambda i: scores[i])
    return {"texto": sentencas[melhor], "score": scores[melhor]}


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
    parser.add_argument('--min-score', type=float, default=0.4,
                        help='Score mínimo para retornar um resultado do ChromaDB (0.0–1.0). '
                             'Sem --rerank, é a similaridade cosseno; com --rerank, é o score do reranker.')
    parser.add_argument('--rerank', action='store_true',
                        help='Reordena os resultados semânticos com o cross-encoder '
                             f'{RERANKER_MODEL} (2º estágio, mais preciso e mais lento).')
    parser.add_argument('--somente-simplificada', action='store_true',
                        help='Busca semântica só nos chunks da bula simplificada.')
    parser.add_argument('--min-score-rerank', type=float, default=None,
                        help='Limiar no modo --rerank (escala do reranker). '
                             f'Default: {RERANK_MIN_SCORE}.')

    args = parser.parse_args()
    init()
    resultados = pesquisar(args.query, args.n, args.min_score, args.rerank,
                           args.somente_simplificada, args.min_score_rerank)

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
    rerank_scores = resultados['chroma'].get('rerank_scores') or []
    titulo_sem = "Busca Semântica + Reranker" if rerank_scores else "Busca Semântica"
    print(f"\n[CHROMADB - {titulo_sem}]")
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
            if rerank_scores:
                # Com rerank a ordem é a do reranker; mostramos o cosseno ao lado
                # para dar pra comparar o que o 2º estágio promoveu/rebaixou.
                print(f"Rerank: {rerank_scores[i]:.4f} | cosseno: {c_score:.4f} "
                      f"(distância: {c_dist:.4f})")
            else:
                print(f"Score: {c_score:.4f} (distância: {c_dist:.4f})")
            trecho = resultados['chroma']['trechos_match'][i]
            if trecho:
                print(f"Trecho acessível que casou (sim. {trecho['score']:.4f}): "
                      f"{trecho['texto']}")
            print(f"Sentido Encontrado: {c_doc[:150]}...")
            print("-" * 20)
    else:
        print("Nenhuma relação semântica encontrada.")
