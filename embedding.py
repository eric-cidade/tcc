import os
import pandas as pd
import meilisearch
import chromadb
from dotenv import load_dotenv

from sinonimos import sinonimos_meili
from bulas import TIPOS, parse_bula, preparar_chunks
from modelos import carregar_modelo

# --- 1. Configurações Iniciais ---
load_dotenv()

# Modelo multilíngue de retrieval. Default gte-multilingual-base: ~305M params,
# 768 dim, contexto 8192. Mais leve que o bge-m3 (568M/1024) e sem prefixo
# obrigatório de query/passage. Exige trust_remote_code (arquitetura custom;
# precisa de einops) — inofensivo para modelos sem código remoto (ex.: bge-m3).
# Configurável por env p/ comparar modelos (ver CHROMA_COLLECTION abaixo).
EMBED_MODEL = os.getenv('EMBED_MODEL', 'Alibaba-NLP/gte-multilingual-base')
# Coleção do Chroma. O default novo ('corpop_saude_gte') deixa o índice antigo
# do bge-m3 ('corpop_saude') INTACTO para comparação A/B. Cada (modelo, coleção)
# precisa casar: para reindexar/buscar com o bge, rode com
# EMBED_MODEL=BAAI/bge-m3 e CHROMA_COLLECTION=corpop_saude.
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION', 'corpop_saude_gte')

print(f"Carregando modelo de embedding ({EMBED_MODEL})...")
model = carregar_modelo(EMBED_MODEL)
print("Modelo carregado com sucesso!")
# Inicialização dos Clientes
print("Conectando ao Meilisearch e ChromaDB...")
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

# Sinônimos: termos técnicos <-> leigos do domínio (ex.: cefaleia / dor de
# cabeça). Assim a busca léxica por "cefaleia" casa com bulas que só escrevem
# "dor de cabeça". Fonte única em sinonimos.py (reusada na expansão de query).
meili_index.update_synonyms(sinonimos_meili())
print("Conexões meili estabelecidas com sucesso!")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# A indexação reconstrói do zero APENAS a coleção-alvo (CHROMA_COLLECTION). O
# esquema de chunks (estrutural, item-a-item) e a dimensão do embedding mudaram,
# então não dá para misturar com vetores antigos da mesma coleção. Outras
# coleções (ex.: o índice antigo do bge-m3) ficam intactas para comparação.
try:
    chroma_client.delete_collection(CHROMA_COLLECTION)
except Exception:
    pass
chroma_coll = chroma_client.create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
print(f"Conexões chroma estabelecidas com sucesso! (coleção: {CHROMA_COLLECTION})")

def realizar_indexacao():
    total = 0
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
            path_o = os.path.join(cfg["path_original"], f"{n_id}{cfg['suffix_orig']}")
            path_s = os.path.join(cfg["path_simplificada"], f"{n_id}{cfg['suffix_simp']}")
            if not (os.path.exists(path_o) and os.path.exists(path_s)):
                print(f" [ERRO] Arquivos para {base_id} ({nome_remedio}) não encontrados.")
                continue

            with open(path_o, "r", encoding="utf-8") as f_o, \
                 open(path_s, "r", encoding="utf-8") as f_s:
                # As bulas vêm em XML (bloco <metadata> + <text>); separamos os
                # metadados e indexamos só o corpo de <text>.
                meta_bula, texto_orig = parse_bula(f_o.read())
                _, texto_simp = parse_bula(f_s.read())

            # Cada registro é fatiado em chunks estruturais (header fica no embed,
            # não no documento). As entradas dos dois registros são indexadas juntas.
            chunks_o = preparar_chunks(texto_orig, estrutural=True)
            chunks_s = preparar_chunks(texto_simp, estrutural=True)
            entradas = ([("original", i, c) for i, c in enumerate(chunks_o)]
                        + [("simplificada", i, c) for i, c in enumerate(chunks_s)])

            # Embeddamos `c["embed"]` (header + texto) mas ARMAZENAMOS só `c["texto"]`
            # cru em `documents` — o cabeçalho fica em metadata. Assim o vetor ganha
            # contexto e a reconstrução do documento completo (search.py) fica limpa.
            embs = model.encode([c["embed"] for _, _, c in entradas]).tolist()

            chroma_coll.add(
                embeddings=embs,
                documents=[c["texto"] for _, _, c in entradas],
                metadatas=[
                    {"id": base_id, "nome": nome_remedio, "tipo": tipo,
                     "registro": reg, "chunk_idx": idx,
                     "secao": c["secao"], "subsecao": c["subsecao"], "header": c["header"],
                     **meta_bula}
                    for reg, idx, c in entradas
                ],
                ids=[f"{base_id}_{'orig' if reg == 'original' else 'simp'}_{idx}"
                     for reg, idx, _ in entradas],
            )

            meili_index.add_documents([{
                "id": base_id,
                "nome": nome_remedio,
                "tipo": tipo,
                "conteudo_original": texto_orig,
                "conteudo_simplificado": texto_simp,
                **meta_bula,
            }])

            total += 1
            print(f" [OK] {base_id}: {nome_remedio} indexado "
                  f"({len(chunks_o)} chunks orig + {len(chunks_s)} simp).")

    print(f"Indexação concluída. Medicamentos indexados: {total}.")


if __name__ == "__main__":
    realizar_indexacao()