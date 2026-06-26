import os
import re
import pandas as pd
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from sinonimos import sinonimos_meili

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

# Sinônimos: termos técnicos <-> leigos do domínio (ex.: cefaleia / dor de
# cabeça). Assim a busca léxica por "cefaleia" casa com bulas que só escrevem
# "dor de cabeça". Fonte única em sinonimos.py (reusada na expansão de query).
meili_index.update_synonyms(sinonimos_meili())
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
        "path_original": "./csv_metadata/ht/original",
        "path_simplificada": "./csv_metadata/ht/simplificada",
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_validada_limpo.txt",
    },
    {
        "tipo": "oncologia",
        "map_csv": "./remedios_onco_map.csv",
        "path_original": "./csv_metadata/onco/original",
        "path_simplificada": "./csv_metadata/onco/simplificada",
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_simplificada_limpo.txt",
    },
]


# Campos de metadados (do bloco <metadata> das bulas XML) que guardamos junto
# de cada entrada no Chroma e no Meilisearch. Mantemos um conjunto curado e
# útil para busca/exibição (princípio ativo, doença, fabricante, etc.).
CAMPOS_METADADOS = [
    "doenca", "nome_medicamento", "nome_comercial",
    "medgenerico", "medreferencia", "fabricante", "apresentacao",
]


def _parse_bula(raw):
    """Separa uma bula no novo formato em (metadados: dict, texto: str).

    O arquivo tem um bloco <metadata>…</metadata> e um bloco <text>…</text>.
    Não dá para usar um parser XML padrão porque alguns nomes de tag têm espaço
    (ex.: <nome medicamento>) — o que é XML inválido —, então extraímos por
    regex tolerante. Só o conteúdo de <text> é indexado/embeddado; os metadados
    viram campos à parte. Se as tags não existirem, cai para o texto inteiro.
    """
    m = re.search(r"<text>(.*?)</text>", raw, re.DOTALL)
    texto = (m.group(1) if m else raw).strip()

    meta = {}
    bloco = re.search(r"<metadata>(.*?)</metadata>", raw, re.DOTALL)
    if bloco:
        for tag, valor in re.findall(r"<([^/>][^>]*?)>(.*?)</\1>", bloco.group(1), re.DOTALL):
            chave = re.sub(r"\s+", "_", tag.strip().lower())
            meta[chave] = valor.strip()

    # Só os campos curados, sempre como string não vazia (o Chroma rejeita None).
    meta_curado = {c: (meta.get(c) or "NA") for c in CAMPOS_METADADOS}
    return meta_curado, texto


# Tamanho-alvo de cada chunk em caracteres. Cada bula é fatiada em vários
# trechos curtos para indexação: assim uma menção pontual (ex.: "dor de cabeça"
# numa lista de reações) vira um vetor próprio e focado, em vez de ser diluída
# na média do documento inteiro — o que fazia a busca semântica não a encontrar.
CHUNK_ALVO_CHARS = 600


def _chunk(texto, alvo_chars=CHUNK_ALVO_CHARS):
    """Quebra `texto` em trechos de ~alvo_chars, preservando quebras de linha.

    Cada linha não vazia é uma unidade (mantém listas de sintomas/seções
    intactas); linhas muito longas são subdivididas por sentença para não
    recriar o problema do "vetor borrado". Unidades consecutivas são agrupadas
    até o tamanho-alvo. A reconstrução (em search.py) junta os chunks de volta.
    """
    unidades = []
    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha:
            continue
        if len(linha) > alvo_chars:
            unidades.extend(s.strip() for s in re.split(r"(?<=[.!?;])\s+", linha) if s.strip())
        else:
            unidades.append(linha)

    chunks, atual = [], ""
    for u in unidades:
        if atual and len(atual) + len(u) + 1 > alvo_chars:
            chunks.append(atual)
            atual = u
        else:
            atual = f"{atual}\n{u}".strip()
    if atual:
        chunks.append(atual)
    return chunks or [texto.strip()]


# Reconhece os ids de chunk gerados abaixo: "{base}_orig_{n}" / "{base}_simp_{n}".
_RE_CHUNK_ID = re.compile(r"_(orig|simp)_\d+$")


def _ids_existentes():
    """IDs base (sem sufixo _orig/_simp_N) já presentes na coleção do Chroma."""
    existentes = chroma_coll.get(include=[]).get("ids", [])
    bases = set()
    for cid in existentes:
        m = _RE_CHUNK_ID.search(cid)
        if m:
            bases.add(cid[:m.start()])
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
                # As bulas vêm em XML (bloco <metadata> + <text>); separamos os
                # metadados e indexamos só o corpo de <text>.
                meta_bula, texto_orig = _parse_bula(f_o.read())
                _, texto_simp = _parse_bula(f_s.read())

            # Em vez de 1 vetor por bula, geramos vários: cada registro é
            # fatiado em chunks e cada chunk vira uma entrada própria no Chroma.
            chunks_o = _chunk(texto_orig)
            chunks_s = _chunk(texto_simp)
            docs_chunks = chunks_o + chunks_s
            embs_chunks = model.encode(docs_chunks).tolist()

            def _meta_chunk(registro, idx):
                # Campos base + metadados curados da bula (fabricante, doença...).
                return {"id": base_id, "nome": nome_remedio, "tipo": tipo,
                        "registro": registro, "chunk_idx": idx, **meta_bula}

            chroma_coll.add(
                embeddings=embs_chunks,
                documents=docs_chunks,
                metadatas=(
                    [_meta_chunk("original", i) for i in range(len(chunks_o))]
                    + [_meta_chunk("simplificada", i) for i in range(len(chunks_s))]
                ),
                ids=(
                    [f"{base_id}_orig_{i}" for i in range(len(chunks_o))]
                    + [f"{base_id}_simp_{i}" for i in range(len(chunks_s))]
                ),
            )

            meili_index.add_documents([{
                "id": base_id,
                "nome": nome_remedio,
                "tipo": tipo,
                "conteudo_original": texto_orig,
                "conteudo_simplificado": texto_simp,
                **meta_bula,
            }])

            total_novos += 1
            print(f" [OK] {base_id}: {nome_remedio} indexado "
                  f"({len(chunks_o)} chunks orig + {len(chunks_s)} simp).")

    print(f"Indexação concluída. Novos: {total_novos}.")


if __name__ == "__main__":
    realizar_indexacao()