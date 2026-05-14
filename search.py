import argparse
import os
import sys
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Globais preenchidas por init(). Permanecem None até a inicialização ser
# chamada (pelo bloco __main__ da CLI ou pelo startup da API em api.py).
model = None
meili_client = None
meili_index = None
chroma_client = None
chroma_coll = None


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
def pesquisar(query, limite=1, min_score=0.45):
    if model is None:
        init()

    # A. Busca Léxica (Meilisearch)
    # matchingStrategy="all": exige que TODAS as palavras do termo estejam no
    # documento (o padrão "last" iria descartando palavras até achar algo).
    res_meili = meili_index.search(query, {"limit": limite, "matchingStrategy": "all"})

    # B. Busca Semântica (ChromaDB)
    query_vec = model.encode(query).tolist()
    # No Chroma, pedimos explicitamente IDs e Metadados
    res_chroma = chroma_coll.query(
        query_embeddings=[query_vec],
        n_results=limite,
        include=["documents", "metadatas", "distances"]
    )

    ids = res_chroma['ids'][0] if res_chroma['ids'] else []
    docs = res_chroma['documents'][0] if res_chroma['documents'] else []
    metas = res_chroma['metadatas'][0] if res_chroma['metadatas'] else []
    dists = res_chroma['distances'][0] if res_chroma['distances'] else []

    # Com espaço cosseno, score = 1 - distância ∈ [-1, 1].
    # Filtra resultados abaixo do limiar de similaridade.
    filtered = [
        (i, d, m, dist)
        for i, d, m, dist in zip(ids, docs, metas, dists)
        if (1 - dist) >= min_score
    ]

    # Cada medicamento foi indexado em dois registros (..._orig e ..._simp).
    # A query devolve só o que casou; aqui buscamos os dois textos de cada
    # medicamento presente nos resultados, para permitir comparação lado a lado.
    textos = _buscar_ambos_registros([str(m.get("id")) for _, _, m, _ in filtered])

    def _txt(meta, chave):
        return textos.get(str(meta.get("id")), {}).get(chave)

    return {
        "meili": res_meili['hits'],
        "chroma": {
            "ids": [t[0] for t in filtered],
            "docs": [t[1] for t in filtered],
            "metadatas": [t[2] for t in filtered],
            "distances": [t[3] for t in filtered],
            "originais": [_txt(t[2], "original") for t in filtered],
            "simplificadas": [_txt(t[2], "simplificada") for t in filtered],
        }
    }


def _buscar_ambos_registros(base_ids):
    """Para cada id de medicamento, devolve {'original': str|None, 'simplificada': str|None}.

    Lê os documentos `{id}_orig` e `{id}_simp` do ChromaDB numa única chamada.
    """
    vistos = []
    for bid in base_ids:
        if bid and bid not in vistos:
            vistos.append(bid)
    if not vistos:
        return {}

    wanted = [f"{bid}_{suf}" for bid in vistos for suf in ("orig", "simp")]
    got = chroma_coll.get(ids=wanted, include=["documents", "metadatas"])

    out = {}
    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
        bid = str(meta.get("id"))
        slot = out.setdefault(bid, {"original": None, "simplificada": None})
        if str(meta.get("registro", "")).startswith("simplific"):
            slot["simplificada"] = doc
        else:
            slot["original"] = doc
    return out


# --- Interface de Terminal (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Busca Híbrida CorPop-Saúde (UFRGS)')
    parser.add_argument('query', type=str, help='Termo de busca')
    parser.add_argument('--n', type=int, default=1, help='Número de resultados')
    parser.add_argument('--min-score', type=float, default=0.4,
                        help='Similaridade cosseno mínima para retornar um resultado do ChromaDB (0.0–1.0)')

    args = parser.parse_args()
    init()
    resultados = pesquisar(args.query, args.n, args.min_score)

    print("\n" + "═"*50)
    print(f"RESULTADOS PARA: '{args.query}'")
    print("═"*50)

    # --- Exibição Meilisearch ---
    print("\n[MEILISEARCH - Busca por Palavra]")
    if resultados['meili']:
        for hit in resultados['meili']:
            # O 'id' aqui vem direto do documento indexado
            print(f"ID: {hit['id']} | Remédio: {hit['nome']}")
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
            print(f"Sentido Encontrado: {c_doc[:150]}...")
            print("-" * 20)
    else:
        print("Nenhuma relação semântica encontrada.")
