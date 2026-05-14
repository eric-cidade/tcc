import os
import pandas as pd
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- 1. Configurações Iniciais ---
# Usando o modelo multilingue recomendado para português
print("Carregando modelo de embedding...")
model = SentenceTransformer('BAAI/bge-m3')
model.max_seq_length = 8192
print("Modelo carregado com sucesso!")
# Inicialização dos Clientes
print("Conectando ao Meilisearch e ChromaDB...")
load_dotenv()
MEILI_KEY = os.getenv('MEILI_MASTER_KEY')
MEILI_URL = os.getenv('MEILI_URL', 'http://localhost:7700')
if not MEILI_KEY:
    print("❌ Erro: MEILI_MASTER_KEY não encontrada no arquivo .env")
    exit(1)
meili_client = meilisearch.Client(MEILI_URL, MEILI_KEY)
meili_index = meili_client.index('corpop_saude')

# Stop words: palavras ignoradas no casamento léxico. Com matchingStrategy="all"
# (em search.py) isso permite que "cloridrato DE propranolol" case com bulas que
# só dizem "propranolol", exigindo apenas as palavras de conteúdo.
STOP_WORDS_PT = [
    "de", "da", "do", "das", "dos", "e", "a", "o", "as", "os",
    "para", "por", "em", "no", "na", "nos", "nas",
    "com", "sem", "um", "uma", "uns", "umas", "ao", "aos",
]
meili_index.update_stop_words(STOP_WORDS_PT)
print("Conexões meili estabelecidas com sucesso!")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_coll = chroma_client.get_or_create_collection(
    name="corpop_saude",
    metadata={"hnsw:space": "cosine"},
)
print("Conexões chroma estabelecidas com sucesso!")

# Cada tipo de bula vive numa pasta própria, mas todos compartilham a mesma
# coleção/índice. O `tipo` entra como prefixo do id para evitar colisão entre
# mapas (ex.: ht_1 vs onco_1) e como metadado para filtragem futura.
TIPOS = [
    {
        "tipo": "hipertensao",
        "map_csv": "./remedios_ht_map.csv",
        "path_original": "./csv/ht/original",
        "path_simplificada": "./csv/ht/simplificada",
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_validada_limpo.txt",
    },
    {
        "tipo": "oncologia",
        "map_csv": "./remedios_onco_map.csv",
        "path_original": "./csv/onco/original",
        "path_simplificada": "./csv/onco/simplificada",
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_simplificada_limpo.txt",
    },
]


def _ids_existentes():
    """IDs base (sem sufixo _orig/_simp) já presentes na coleção do Chroma."""
    existentes = set(chroma_coll.get(include=[]).get("ids", []))
    bases = set()
    for cid in existentes:
        if cid.endswith("_orig") or cid.endswith("_simp"):
            bases.add(cid.rsplit("_", 1)[0])
    return bases


def realizar_indexacao():
    ja_indexados = _ids_existentes()
    print(f"Já indexados: {len(ja_indexados)} medicamentos. Verificando novos...")

    total_novos = 0
    for cfg in TIPOS:
        tipo = cfg["tipo"]
        if not os.path.exists(cfg["map_csv"]):
            print(f" [SKIP] Mapa não encontrado: {cfg['map_csv']}")
            continue

        df_map = pd.read_csv(cfg["map_csv"], dtype={"id": str})
        mapa = dict(zip(df_map["id"], df_map["nome"]))
        print(f"[{tipo}] {len(mapa)} medicamentos no mapa.")

        for n_id, nome_remedio in mapa.items():
            base_id = f"{tipo}_{n_id}"
            if base_id in ja_indexados:
                continue

            path_o = os.path.join(cfg["path_original"], f"{n_id}{cfg['suffix_orig']}")
            path_s = os.path.join(cfg["path_simplificada"], f"{n_id}{cfg['suffix_simp']}")
            if not (os.path.exists(path_o) and os.path.exists(path_s)):
                print(f" [ERRO] Arquivos para {base_id} ({nome_remedio}) não encontrados.")
                continue

            with open(path_o, "r", encoding="utf-8") as f_o, \
                 open(path_s, "r", encoding="utf-8") as f_s:
                texto_orig = f_o.read().strip()
                texto_simp = f_s.read().strip()

            emb_orig, emb_simp = model.encode([texto_orig, texto_simp]).tolist()

            #TODO : adicionar metadados de laboratorio/fabricante etc.
            chroma_coll.add(
                embeddings=[emb_orig, emb_simp],
                documents=[texto_orig, texto_simp],
                metadatas=[
                    {"id": base_id, "nome": nome_remedio, "tipo": tipo, "registro": "original"},
                    {"id": base_id, "nome": nome_remedio, "tipo": tipo, "registro": "simplificada"},
                ],
                ids=[f"{base_id}_orig", f"{base_id}_simp"],
            )

            meili_index.add_documents([{
                "id": base_id,
                "nome": nome_remedio,
                "tipo": tipo,
                "conteudo_original": texto_orig,
                "conteudo_simplificado": texto_simp,
            }])

            total_novos += 1
            print(f" [OK] {base_id}: {nome_remedio} indexado.")

    print(f"Indexação concluída. Novos: {total_novos}.")


if __name__ == "__main__":
    realizar_indexacao()