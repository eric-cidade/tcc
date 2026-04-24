import argparse
import os
import sys
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
MEILI_KEY = os.getenv('MEILI_MASTER_KEY')
MEILI_URL = os.getenv('MEILI_URL', 'http://localhost:7700')

if not MEILI_KEY:
    print("❌ Erro: MEILI_MASTER_KEY não encontrada no arquivo .env", file=sys.stderr)
    sys.exit(1)

# --- 1. Inicialização ---
print("Conectando aos motores de busca...", file=sys.stderr)
model = SentenceTransformer('BAAI/bge-m3')
model.max_seq_length = 8192

meili_client = meilisearch.Client(MEILI_URL, MEILI_KEY)
meili_index = meili_client.index('corpop_saude_ht')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_coll = chroma_client.get_or_create_collection(
    name="corpop_saude_ht",
    metadata={"hnsw:space": "cosine"},
)

# --- 2. Função de Pesquisa Híbrida ---
def pesquisar(query, limite=1, min_score=0.0):
    # A. Busca Léxica (Meilisearch)
    res_meili = meili_index.search(query, {"limit": limite})

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

    return {
        "meili": res_meili['hits'],
        "chroma": {
            "ids": [t[0] for t in filtered],
            "docs": [t[1] for t in filtered],
            "metadatas": [t[2] for t in filtered],
            "distances": [t[3] for t in filtered],
        }
    }

# --- 3. Interface de Terminal (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Busca Híbrida CorPop-Saúde (UFRGS)')
    parser.add_argument('query', type=str, help='Termo de busca')
    parser.add_argument('--n', type=int, default=1, help='Número de resultados')
    parser.add_argument('--min-score', type=float, default=0.4,
                        help='Similaridade cosseno mínima para retornar um resultado do ChromaDB (0.0–1.0)')

    args = parser.parse_args()
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