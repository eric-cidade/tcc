import argparse
import sys
import meilisearch
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Inicialização ---
print("Conectando aos motores de busca...", file=sys.stderr)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

meili_client = meilisearch.Client('http://localhost:7700', 'teste123-teste123-teste123-teste123')
meili_index = meili_client.index('corpop_saude_ht')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_coll = chroma_client.get_or_create_collection(name="corpop_saude_ht")

# --- 2. Função de Pesquisa Híbrida ---
def pesquisar(query, limite=1):
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
    
    return {
        "meili": res_meili['hits'],
        "chroma": {
            "ids": res_chroma['ids'][0] if res_chroma['ids'] else [],
            "docs": res_chroma['documents'][0] if res_chroma['documents'] else [],
            "metadatas": res_chroma['metadatas'][0] if res_chroma['metadatas'] else []
        }
    }

# --- 3. Interface de Terminal (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Busca Híbrida CorPop-Saúde (UFRGS)')
    parser.add_argument('query', type=str, help='Termo de busca')
    parser.add_argument('--n', type=int, default=1, help='Número de resultados')

    args = parser.parse_args()
    resultados = pesquisar(args.query, args.n)

    print("\n" + "═"*50)
    print(f"🔍 RESULTADOS PARA: '{args.query}'")
    print("═"*50)

    # --- Exibição Meilisearch ---
    print("\n📦 [MEILISEARCH - Busca por Palavra]")
    if resultados['meili']:
        for hit in resultados['meili']:
            # O 'id' aqui vem direto do documento indexado
            print(f"ID: {hit['id']} | Remédio: {hit['nome']}")
            print(f"Simplificado: {hit['conteudo_simplificado'][:150]}...")
            print("-" * 20)
    else:
        print("Nenhum match exato encontrado.")

    # --- Exibição ChromaDB ---
    print("\n🧠 [CHROMADB - Busca Semântica]")
    if resultados['chroma']['ids']:
        # Iteramos usando o índice para combinar ID, Metadata e Documento
        for i in range(len(resultados['chroma']['ids'])):
            c_id = resultados['chroma']['ids'][i]
            c_meta = resultados['chroma']['metadatas'][i]
            c_doc = resultados['chroma']['docs'][i]
            
            print(f"ID: {c_id} | Nome (Metadata): {c_meta.get('nome', 'N/A')}")
            print(f"Sentido Encontrado: {c_doc[:150]}...")
            print("-" * 20)
    else:
        print("Nenhuma relação semântica encontrada.")