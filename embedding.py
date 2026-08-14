"""Indexação do corpus no ChromaDB (semântica) e no Meilisearch (léxica).

Rodar:
  uv run python embedding.py                 # incremental: acrescenta/atualiza
  uv run python embedding.py --reconstruir   # apaga a coleção e refaz do zero

INCREMENTAL é o default. Para cada medicamento, remove os chunks antigos DELE
(pelo metadado `id`) e grava os novos — idempotente, e correto tanto para bula
nova quanto para bula revisada, inclusive quando o número de chunks diminui.
Não há motivo para reconstruir tudo ao acrescentar documentos.

--reconstruir só é necessário quando muda algo que invalida os vetores já
gravados: o modelo de embedding (dimensão diferente) ou o esquema de chunking.

Sobre a memória: por padrão o script NÃO carrega o modelo. Ele pede os vetores
à API (endpoint /embed), que já tem o modelo residente. Isso é o que permite
indexar com a API no ar num servidor de 4GB — duas cópias do modelo não caberiam.
Se a API não responder, carrega o modelo localmente (necessário na 1ª indexação,
antes de a API existir).
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

import meilisearch
import chromadb
from dotenv import load_dotenv

from caminhos import CHROMA_PATH
from sinonimos import sinonimos_meili
from bulas import TIPOS, ler_mapa, parse_bula, preparar_chunks
from modelos import EMBED_MODEL_PADRAO, CHROMA_COLLECTION_PADRAO

# --- 1. Configurações Iniciais ---
load_dotenv()

# Modelo de embedding e coleção-alvo. Os dois PRECISAM casar entre si e com o que
# a busca usa: cada modelo tem dimensão e escala de score próprias, e por isso a
# sua própria coleção. Os defaults vêm de search.py (fonte única) e são
# configuráveis por env para comparar modelos.
EMBED_MODEL = os.getenv('EMBED_MODEL') or EMBED_MODEL_PADRAO
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION') or CHROMA_COLLECTION_PADRAO
# Lote do encode. É o principal controle de pico de RAM na indexação: no servidor
# (4GB, sem GPU) o default 32 do sentence-transformers pode dar pico feio.
EMBED_BATCH = int(os.getenv('EMBED_BATCH', '0')) or 32
# API a quem pedir os embeddings. Vazio desliga e força o modelo local.
EMBED_API = os.getenv('EMBED_API', 'http://127.0.0.1:8000')


def _embeddar_via_api(textos):
    """Pede os vetores ao endpoint /embed da API. None se ela não puder servir."""
    req = urllib.request.Request(
        f"{EMBED_API.rstrip('/')}/embed",
        data=json.dumps({"textos": textos}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def montar_embeddador():
    """Devolve (funcao_de_embed, descricao).

    Prefere a API — assim não há segunda cópia do modelo na memória. Só cai para
    o modelo local se a API estiver fora, e ABORTA se ela estiver no ar com um
    modelo diferente do alvo: misturar vetores de modelos distintos na mesma
    coleção corrompe a busca em silêncio, sem erro nenhum.
    """
    if EMBED_API:
        try:
            resp = _embeddar_via_api(["teste"])
        except (urllib.error.URLError, OSError, TimeoutError):
            resp = None
        if resp is not None:
            if resp.get("modelo") != EMBED_MODEL:
                raise SystemExit(
                    f"❌ A API em {EMBED_API} está com o modelo {resp.get('modelo')!r}, "
                    f"mas a indexação é para {EMBED_MODEL!r}.\n"
                    f"   Alinhe EMBED_MODEL no .env e reinicie a API, ou rode com "
                    f"EMBED_API= (vazio) para carregar o modelo localmente."
                )
            def embed(textos):
                saida = []
                for i in range(0, len(textos), EMBED_BATCH):
                    saida += _embeddar_via_api(textos[i:i + EMBED_BATCH])["vetores"]
                return saida
            return embed, f"API {EMBED_API} (sem carregar o modelo aqui)"

    from modelos import carregar_modelo, encode_docs
    print(f"API indisponível — carregando o modelo localmente ({EMBED_MODEL})...")
    modelo = carregar_modelo(EMBED_MODEL)
    return (lambda textos: encode_docs(modelo, textos, batch_size=EMBED_BATCH).tolist(),
            f"modelo local {EMBED_MODEL}")


# Inicialização dos Clientes
print("Conectando ao Meilisearch e ChromaDB...")
MEILI_KEY = os.getenv('MEILI_MASTER_KEY')
# 127.0.0.1 e não 'localhost' — ver a nota em search.py: com 'localhost' cada
# chamada custa ~2s de timeout de IPv6, e a indexação faz uma por medicamento.
MEILI_URL = os.getenv('MEILI_URL', 'http://127.0.0.1:7700')
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
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def abrir_colecao(reconstruir):
    """Abre a coleção-alvo. Com `reconstruir`, apaga e recria antes.

    ATENÇÃO: reconstruir com a API no ar a QUEBRA. Ela guarda um handle pelo UUID
    da coleção; apagada, o UUID some e todas as consultas passam a devolver
    NotFoundError até a API ser reiniciada — mesmo depois de o índice novo ficar
    pronto. O modo incremental não tem esse problema: preserva a coleção.
    """
    if reconstruir:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        return chroma_client.create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})


def _hms(seg):
    """Formata segundos como '1m23s' ou '45.2s' para os logs de progresso."""
    return f"{int(seg // 60)}m{int(seg % 60):02d}s" if seg >= 60 else f"{seg:.1f}s"


def realizar_indexacao(embed, chroma_coll, incremental=True):
    total = 0
    t0 = time.perf_counter()
    for cfg in TIPOS:
        tipo = cfg["tipo"]
        if not os.path.exists(cfg["map_csv"]):
            print(f" [SKIP] Mapa não encontrado: {cfg['map_csv']}")
            continue

        mapa = ler_mapa(cfg["map_csv"])
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
            t_med = time.perf_counter()
            embs = embed([c["embed"] for _, _, c in entradas])

            # No modo incremental, remove os chunks ANTERIORES deste medicamento
            # antes de gravar os novos. Sem isso, uma bula revisada com menos
            # chunks deixaria os excedentes órfãos no índice — o `add` só
            # sobrescreveria os ids que se repetem.
            if incremental:
                chroma_coll.delete(where={"id": base_id})

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
            # Tempo por medicamento e acumulado: a indexação no servidor (CPU,
            # 4 vCPU) é longa e sem isso não dá para saber se travou ou só demora.
            print(f" [OK] {base_id}: {nome_remedio} indexado "
                  f"({len(chunks_o)} chunks orig + {len(chunks_s)} simp) "
                  f"[{_hms(time.perf_counter() - t_med)} | "
                  f"total {_hms(time.perf_counter() - t0)}]", flush=True)

    print(f"Indexação concluída. Medicamentos indexados: {total} "
          f"em {_hms(time.perf_counter() - t0)}.")


def main():
    ap = argparse.ArgumentParser(description="Indexa o corpus no ChromaDB e no Meilisearch")
    ap.add_argument("--reconstruir", action="store_true",
                    help="apaga a coleção e refaz do zero. Só é necessário ao trocar "
                         "de modelo ou de esquema de chunking — e QUEBRA a API se ela "
                         "estiver no ar (precisa reiniciá-la depois).")
    args = ap.parse_args()

    embed, descricao = montar_embeddador()
    chroma_coll = abrir_colecao(args.reconstruir)
    modo = "RECONSTRUÇÃO (do zero)" if args.reconstruir else "incremental"
    print(f"Modo: {modo} | coleção: {CHROMA_COLLECTION} ({chroma_coll.count()} chunks) "
          f"| embeddings: {descricao}")
    realizar_indexacao(embed, chroma_coll, incremental=not args.reconstruir)
    print(f"Coleção '{CHROMA_COLLECTION}' agora com {chroma_coll.count()} chunks.")


if __name__ == "__main__":
    main()