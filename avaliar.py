"""Avaliação comparativa: {modelos de embedding} × {chunking simples, estrutural}.

Constrói (se faltar) as coleções do Chroma e compara o retrieval semântico PURO
— cosseno, SEM expansão de sinônimos — numa lista de queries leigas do domínio.
Tirar os sinônimos é proposital: são uma camada de compensação que mascararia a
diferença entre modelo e chunking, justamente o que queremos medir aqui. Ao
final, gera gráficos comparativos em ./graficos.

Serve a dois propósitos: a comparação de chunking do TCC (bge + simples vs.
bge + estrutural) e a ESCOLHA DO MODELO que vai para o servidor — onde só há 4GB
de RAM e o bge-m3 (2,3GB de pesos) é apertado. Por isso os candidatos leves são
avaliados só com chunking estrutural (o chunking já foi decidido).

Candidatos (ver CANDIDATOS abaixo), com o custo de RAM que decide o deploy:
  bge      BAAI/bge-m3                        568M / 1024 dim / ~2,3GB
  gte      Alibaba-NLP/gte-multilingual-base  305M /  768 dim / ~1,2GB
  e5-base  intfloat/multilingual-e5-base      278M /  768 dim / ~1,1GB
  e5-small intfloat/multilingual-e5-small     118M /  384 dim / ~0,5GB

Rodar:
  # comparação de chunking do TCC (bge nos dois chunkings)
  uv run python avaliar.py
  # o 2×2 original, agora com o gte já corrigido (ver modelos.py)
  uv run python avaliar.py --modelos bge,gte
  # escolha do modelo do servidor: candidatos no chunking que a API usa, num
  # recorte pequeno do corpus (só hipertensão, 1 bula por princípio ativo) —
  # 10 medicamentos em vez de 79, o que torna viável indexar 4 modelos
  uv run python avaliar.py --modelos bge,gte,e5-base,e5-small \
      --chunkings estrutural --tipos ht --um-por-remedio

Recortes de corpus (--tipos/--um-por-remedio) geram coleções com nome PRÓPRIO
(sufixo _ht, _1x, ...), separadas das de produção: um --rebuild aqui não tem como
apagar a 'corpop_saude_bge_estr' que a API usa. As queries com gabarito que não
existem no recorte são descartadas automaticamente — senão contariam como erro
em todos os modelos e achatariam a comparação.
  uv run python avaliar.py --k 5                  # top-5 por combinação
  uv run python avaliar.py --rebuild              # reconstrói TODAS as coleções
  uv run python avaliar.py --no-build             # só avalia o que já existe
  uv run python avaliar.py --out graficos         # diretório de saída dos PNGs

Cuidado de leitura: scores de cosseno de modelos DIFERENTES não estão na mesma
escala — compare score entre chunkings DO MESMO modelo; entre modelos, olhe a
posição (rank) e o trecho recuperado, não o número absoluto.
"""
import argparse
import os
import sys
import time

import chromadb
from dotenv import load_dotenv

from caminhos import CHROMA_PATH
from bulas import TIPOS, ler_mapa, parse_bula, preparar_chunks
from modelos import carregar_modelo, encode_query, encode_docs

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # backend sem display (salva PNG direto)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Apelido na CLI -> id do modelo no HuggingFace.
CANDIDATOS = {
    "bge": "BAAI/bge-m3",
    "gte": "Alibaba-NLP/gte-multilingual-base",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small",
    "granite": "ibm-granite/granite-embedding-311m-multilingual-r2",
    "gemma": "google/embeddinggemma-300m",
}

# Coleções herdadas: nomes que já existem no chroma_db de experimentos anteriores
# e que não seguem o padrão `corpop_saude_<apelido>_<chunking>`. Mantidos para não
# reindexar (e não invalidar) o que já foi medido e citado no TCC.
COLECOES_LEGADAS = {
    ("bge", False): "corpop_saude",
    ("bge", True): "corpop_saude_bge_estr",
    ("gte", True): "corpop_saude_gte",
    ("gte", False): "corpop_saude_gte_simples",
}


def _colecao(apelido, estrutural, sufixo=""):
    """Nome da coleção de um (modelo, chunking), respeitando os nomes legados.

    `sufixo` identifica o RECORTE do corpus (ver _perfil_corpus). Ele é o que
    impede o benchmark rápido de escrever por cima das coleções de produção: com
    corpus reduzido nenhum nome legado é reaproveitado, então um --rebuild aqui
    nunca apaga a 'corpop_saude_bge_estr' que a API usa.
    """
    if not sufixo and (apelido, estrutural) in COLECOES_LEGADAS:
        return COLECOES_LEGADAS[(apelido, estrutural)]
    base = f"corpop_saude_{apelido.replace('-', '_')}_{'estr' if estrutural else 'simples'}"
    return base + sufixo


CHUNKINGS = {
    "simples": [False],
    "estrutural": [True],
    "ambos": [False, True],
}

# Apelido na CLI -> valor do campo `tipo` em bulas.TIPOS.
TIPOS_APELIDO = {"ht": "hipertensao", "onco": "oncologia"}


def _perfil_corpus(tipos, um_por_remedio):
    """Sufixo que identifica o recorte do corpus no nome da coleção.

    Corpus cheio -> "" (coleções de produção). Qualquer recorte gera um nome
    próprio, para que as duas coisas coexistam no mesmo chroma_db.
    """
    partes = []
    if tipos:
        partes.append("".join(sorted(a for a, t in TIPOS_APELIDO.items() if t in tipos)))
    if um_por_remedio:
        partes.append("1x")
    return ("_" + "".join(partes)) if partes else ""


def montar_combos(apelidos, chunkings="ambos", sufixo=""):
    """Produto cartesiano modelos × chunkings, como lista de combinações.

    `ambos` (default) reproduz a comparação de chunking do TCC. Para escolher o
    modelo do servidor use `estrutural`: o chunking já está decidido (é o que a
    API usa), e reindexar cada candidato duas vezes custaria caro sem responder
    pergunta nenhuma.
    """
    combos = []
    for apelido in apelidos:
        if apelido not in CANDIDATOS:
            raise SystemExit(f"Modelo desconhecido: {apelido!r}. "
                             f"Conhecidos: {', '.join(CANDIDATOS)}")
        for estrutural in CHUNKINGS[chunkings]:
            combos.append({
                "rotulo": f"{apelido} + {'estrutural' if estrutural else 'simples'}",
                "modelo": CANDIDATOS[apelido],
                "estrutural": estrutural,
                "colecao": _colecao(apelido, estrutural, sufixo),
            })
    return combos


# Preenchido em main() a partir de --modelos; default = a comparação do TCC.
COMBOS = []

# Queries leigas/sinônimas do domínio. `esperado` (opcional) é um trecho do nome
# do medicamento que conta como acerto — usado só nas buscas por princípio ativo,
# onde há resposta certa. As de sintoma/conceito não têm gabarito (vários
# remédios listam o mesmo sintoma): para elas, julga-se o trecho recuperado.
QUERIES = [
    {"q": "dor de cabeça"},
    {"q": "cefaleia"},
    {"q": "enxaqueca"},
    {"q": "enjoo"},
    {"q": "tontura ao levantar"},
    {"q": "falta de ar"},
    {"q": "coceira na pele"},
    {"q": "inchaço no rosto"},
    {"q": "remédio para pressão alta"},
    # Com gabarito — princípio ativo. Misturam nome puro, nome com sal na query
    # (o modelo tem que ignorar o sal) e nome com sal no corpus mas puro na query
    # (tem que casar o parcial). n=15 para que a diferença entre modelos não seja
    # ruído de amostragem: com as 3 queries originais, 1 acerto valia 33%.
    {"q": "losartana", "esperado": "losartana"},
    {"q": "maleato de enalapril", "esperado": "enalapril"},
    {"q": "propranolol", "esperado": "propranolol"},
    {"q": "atenolol", "esperado": "atenolol"},
    {"q": "hidroclorotiazida", "esperado": "hidroclorotiazida"},
    {"q": "espironolactona", "esperado": "espironolactona"},
    {"q": "furosemida", "esperado": "furosemida"},
    {"q": "captopril", "esperado": "captopril"},
    {"q": "besilato de anlodipino", "esperado": "anlodipino"},
    {"q": "metoprolol", "esperado": "metoprolol"},
    {"q": "tamoxifeno", "esperado": "tamoxifeno"},
    {"q": "temozolomida", "esperado": "temozolomida"},
    {"q": "paclitaxel", "esperado": "paclitaxel"},
    {"q": "imatinibe", "esperado": "imatinibe"},
    {"q": "ácido zoledrônico", "esperado": "zoledrônico"},
]

_MODELOS = {}


def _hms(seg):
    """Formata segundos como '1m23s' ou '45.2s' para logs de tempo."""
    return f"{int(seg // 60)}m{int(seg % 60):02d}s" if seg >= 60 else f"{seg:.1f}s"


def get_modelo(nome):
    """Carrega (e cacheia) um modelo via modelos.carregar_modelo, com tempo."""
    if nome not in _MODELOS:
        print(f"  ⏳ carregando modelo {nome}...", file=sys.stderr, flush=True)
        t = time.perf_counter()
        _MODELOS[nome] = carregar_modelo(nome)
        print(f"  ✔ modelo pronto em {_hms(time.perf_counter() - t)}", file=sys.stderr, flush=True)
    return _MODELOS[nome]


def _itens_trabalho(tipos=None, um_por_remedio=False):
    """Lista (cfg, n_id, nome, path_o, path_s) dos medicamentos indexáveis.

    `tipos`: conjunto de valores de `cfg["tipo"]` a manter (None = todos).
    `um_por_remedio`: o corpus tem ~3 bulas por princípio ativo (fabricantes
    diferentes); manter uma só corta o tempo de indexação por ~3 sem mudar quais
    remédios existem — é o que torna viável comparar vários modelos.
    """
    itens, vistos = [], set()
    for cfg in TIPOS:
        if tipos and cfg["tipo"] not in tipos:
            continue
        if not os.path.exists(cfg["map_csv"]):
            continue
        for n_id, nome_rem in ler_mapa(cfg["map_csv"]).items():
            # O corpus tem nomes com espaço sobrando ('furosemida '), que sem
            # normalizar contariam como um segundo remédio na deduplicação.
            chave = (cfg["tipo"], " ".join(nome_rem.split()).casefold())
            if um_por_remedio and chave in vistos:
                continue
            path_o = os.path.join(cfg["path_original"], f"{n_id}{cfg['suffix_orig']}")
            path_s = os.path.join(cfg["path_simplificada"], f"{n_id}{cfg['suffix_simp']}")
            if os.path.exists(path_o) and os.path.exists(path_s):
                vistos.add(chave)
                itens.append((cfg, n_id, nome_rem, path_o, path_s))
    return itens


def construir(combo, client, rebuild, itens):
    """Garante a coleção do combo. Pula se já populada (a menos de --rebuild)."""
    nome = combo["colecao"]
    existentes = [c.name for c in client.list_collections()]
    if nome in existentes and not rebuild:
        coll = client.get_collection(nome)
        if coll.count() > 0:
            print(f"[skip ] {combo['rotulo']:<18} '{nome}' já populada ({coll.count()} chunks).")
            return coll
    if nome in existentes:
        client.delete_collection(nome)
    coll = client.create_collection(nome, metadata={"hnsw:space": "cosine"})

    modelo = get_modelo(combo["modelo"])
    total = len(itens)
    print(f"[build] {combo['rotulo']:<18} '{nome}' — {total} medicamentos a indexar")
    t0 = time.perf_counter()
    n_chunks = 0
    for idx, (cfg, n_id, nome_rem, path_o, path_s) in enumerate(itens, 1):
        td = time.perf_counter()
        base_id = f"{cfg['tipo']}_{n_id}"
        with open(path_o, encoding="utf-8") as f:
            meta_bula, texto_o = parse_bula(f.read())
        with open(path_s, encoding="utf-8") as f:
            _, texto_s = parse_bula(f.read())

        ck = 0
        for registro, texto in (("original", texto_o), ("simplificada", texto_s)):
            chunks = preparar_chunks(texto, combo["estrutural"])
            if not chunks:
                continue
            sufixo = "orig" if registro == "original" else "simp"
            embs = encode_docs(modelo, [c["embed"] for c in chunks]).tolist()
            coll.add(
                embeddings=embs,
                documents=[c["texto"] for c in chunks],
                metadatas=[{"id": base_id, "nome": nome_rem, "tipo": cfg["tipo"],
                            "registro": registro, "chunk_idx": i,
                            "header": chunks[i]["header"], **meta_bula}
                           for i in range(len(chunks))],
                ids=[f"{base_id}_{sufixo}_{i}" for i in range(len(chunks))],
            )
            ck += len(chunks)
        n_chunks += ck
        decorrido = time.perf_counter() - t0
        # ETA simples pela média por medicamento já processado.
        eta = decorrido / idx * (total - idx)
        print(f"  ({idx:>2}/{total}) {base_id:<16} {nome_rem:<26} {ck:>4} chunks  "
              f"[{_hms(time.perf_counter() - td)} | decorrido {_hms(decorrido)} | "
              f"resta ~{_hms(eta)}]", flush=True)
    print(f"        ✔ {total} medicamentos, {coll.count()} chunks em {_hms(time.perf_counter() - t0)}.")
    return coll


def ranquear(coll, modelo, query):
    """Ranking completo de medicamentos por melhor chunk (menor distância).

    Devolve a lista ordenada por score desc. Buscamos fundo (até 500 chunks) para
    que o ranking inclua o medicamento esperado mesmo que não esteja no topo.
    """
    qv = encode_query(modelo, query).tolist()
    n = min(500, coll.count())
    res = coll.query(query_embeddings=[qv], n_results=n,
                     include=["documents", "metadatas", "distances"])
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    melhor = {}
    for doc, m, dist in zip(docs, metas, dists):
        bid = str(m.get("id"))
        if bid not in melhor:  # resultados já vêm ordenados por distância
            melhor[bid] = {"nome": m.get("nome"), "score": 1 - dist,
                           "doc": doc, "registro": m.get("registro")}
    return sorted(melhor.values(), key=lambda x: -x["score"])


def _rank_esperado(ranking, esperado):
    """Posição (1-based) do 1º medicamento cujo nome contém `esperado`, ou None."""
    for i, r in enumerate(ranking):
        if esperado.lower() in (r["nome"] or "").lower():
            return i + 1
    return None


def queries_aplicaveis(itens):
    """Descarta as queries cujo gabarito não existe no recorte de corpus indexado.

    Com `--tipos ht` as queries de oncologia não têm como acertar: mantê-las
    contaria como erro em TODOS os modelos e diluiria a diferença entre eles,
    que é justamente o que o benchmark existe para medir. As queries sem gabarito
    ficam sempre (não pontuam hit@k, só alimentam o heatmap de score).
    """
    nomes = [(nome or "").lower() for _, _, nome, _, _ in itens]
    return [item for item in QUERIES
            if not item.get("esperado")
            or any(item["esperado"].lower() in nome for nome in nomes)]


def avaliar(colls, k, queries=None):
    """Roda as queries em cada combinação, imprime os resultados e coleta métricas."""
    queries = QUERIES if queries is None else queries
    registros = []  # um dict por (combinação, query)
    t_aval = time.perf_counter()
    print(f"\n>>> Avaliando {len(queries)} queries × {len(COMBOS)} combinações...")
    for nq, item in enumerate(queries, 1):
        q, esperado = item["q"], item.get("esperado")
        tq = time.perf_counter()
        print("\n" + "═" * 78)
        print(f"QUERY {nq}/{len(queries)}: {q!r}"
              + (f"   (esperado ~ {esperado})" if esperado else "   (sem gabarito)"))
        print("═" * 78)
        for combo in COMBOS:
            coll, modelo = colls[combo["rotulo"]]
            reg = {"combo": combo["rotulo"], "query": q, "esperado": esperado,
                   "top1": float("nan"), "rank": None}
            if coll is None:
                print(f"\n[{combo['rotulo']}]  (coleção ausente — rode sem --no-build)")
                registros.append(reg)
                continue
            tb = time.perf_counter()
            try:
                ranking = ranquear(coll, modelo, q)
            except Exception as exc:
                print(f"\n[{combo['rotulo']}]  (erro na busca: {exc})")
                registros.append(reg)
                continue
            print(f"\n[{combo['rotulo']}]  ({_hms(time.perf_counter() - tb)})")
            reg["top1"] = ranking[0]["score"] if ranking else float("nan")
            reg["rank"] = _rank_esperado(ranking, esperado) if esperado else None
            for r in ranking[:k]:
                hit = esperado and esperado.lower() in (r["nome"] or "").lower()
                snippet = (r["doc"] or "").replace("\n", " ").strip()[:84]
                print(f"  {r['score']:.3f}  {(r['nome'] or '?'):<26} «{snippet}»{'  ✓' if hit else ''}")
            registros.append(reg)
        print(f"\n… query processada em {_hms(time.perf_counter() - tq)}")
    print(f"\n>>> Avaliação concluída em {_hms(time.perf_counter() - t_aval)}.")

    _resumo_texto(registros, k)
    return registros


def _resumo_texto(registros, k):
    rotulados = [r for r in registros if r["esperado"]]
    if not rotulados:
        return
    print("\n" + "═" * 78)
    n_lab = len({r["query"] for r in rotulados})
    print(f"RESUMO — hit@{k} nas {n_lab} queries com gabarito (busca por princípio ativo):")
    for combo in COMBOS:
        hits = sum(1 for r in rotulados
                   if r["combo"] == combo["rotulo"] and r["rank"] and r["rank"] <= k)
        print(f"  {combo['rotulo']:<18} {hits}/{n_lab}")


# --- Gráficos ------------------------------------------------------------------

def _coletar_granularidade(itens):
    """Tamanho e nº de chunks por documento, por estratégia (independe do modelo).

    Usa os MESMOS itens indexados, para que o gráfico descreva o corpus que foi
    de fato avaliado e não um recorte diferente.
    """
    dados = {"simples": {"counts": [], "lens": []},
             "estrutural": {"counts": [], "lens": []}}
    for _, _, _, path_o, path_s in itens:
        for caminho in (path_o, path_s):
            with open(caminho, encoding="utf-8") as f:
                _, texto = parse_bula(f.read())
            for estrut, chave in ((False, "simples"), (True, "estrutural")):
                ch = preparar_chunks(texto, estrut)
                dados[chave]["counts"].append(len(ch))
                dados[chave]["lens"].extend(len(c["texto"]) for c in ch)
    return dados


def _heatmap(ax, matriz, row_labels, col_labels, titulo, cmap, fmt, vmin=None, vmax=None):
    arr = np.ma.masked_invalid(np.array(matriz, dtype=float))
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("lightgray")
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "—" if v is np.ma.masked else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    ax.set_title(titulo, fontsize=10)
    return im


def gerar_graficos(registros, k, out_dir, itens, queries):
    if plt is None:
        print("\n[gráficos] matplotlib/numpy ausentes — pulei. "
              "Instale com: uv sync (grupo dev)", file=sys.stderr)
        return
    print(f"\n>>> Gerando gráficos em {out_dir}/...")
    tg = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    rotulos = [c["rotulo"] for c in COMBOS]

    # 1) Granularidade do chunking (depende só da estratégia) -------------------
    print("  • granularidade (recalculando chunks das bulas avaliadas)...", flush=True)
    gran = _coletar_granularidade(itens)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    estrategias = ["simples", "estrutural"]
    medias = [np.mean(gran[e]["counts"]) for e in estrategias]
    barras = ax1.bar(estrategias, medias, color=["#c0392b", "#27ae60"])
    ax1.set_ylabel("nº médio de chunks por documento")
    ax1.set_title("Granularidade: chunks por documento")
    for b, v in zip(barras, medias):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax2.boxplot([gran[e]["lens"] for e in estrategias], tick_labels=estrategias, showfliers=False)
    ax2.set_ylabel("tamanho do chunk (caracteres)")
    ax2.set_title("Distribuição do tamanho dos chunks")
    fig.suptitle("Chunking simples vs. estrutural — quanto menor/mais focado, menos 'vetor borrado'",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "granularidade.png"), dpi=130)
    plt.close(fig)
    print("    ↳ granularidade.png", flush=True)

    # 2) hit@k por combinação (queries com gabarito) ----------------------------
    rotulados = [r for r in registros if r["esperado"]]
    n_lab = len({r["query"] for r in rotulados})
    if n_lab:
        hits = [sum(1 for r in rotulados if r["combo"] == rot and r["rank"] and r["rank"] <= k)
                for rot in rotulos]
        # Uma cor por combinação, geradas do colormap: o número de combinações
        # agora varia com --modelos, então uma lista fixa de 4 cores quebraria.
        cores = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(rotulos)))
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(rotulos)), 4.2))
        barras = ax.bar(rotulos, hits, color=cores)
        ax.set_ylim(0, n_lab)
        ax.set_ylabel(f"acertos (de {n_lab})")
        ax.set_title(f"hit@{k} — busca por princípio ativo")
        for b, v in zip(barras, hits):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(rotulos)))
        ax.set_xticklabels(rotulos, rotation=15, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "hit_at_k.png"), dpi=130)
        plt.close(fig)
        print("    ↳ hit_at_k.png", flush=True)

        # 3) Rank do esperado (heatmap; scale-free, justo entre modelos) ---------
        labq = [r["query"] for r in rotulados if r["combo"] == rotulos[0]]
        matriz = [[next((r["rank"] for r in rotulados
                         if r["combo"] == rot and r["query"] == q), None) or float("nan")
                   for rot in rotulos] for q in labq]
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(rotulos) + 3), 0.7 * len(labq) + 2))
        im = _heatmap(ax, matriz, labq, rotulos,
                      "Posição do medicamento esperado (1 = topo; menor é melhor)",
                      "RdYlGn_r", "{:.0f}", vmin=1, vmax=max(2, k))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rank")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rank_esperado.png"), dpi=130)
        plt.close(fig)
        print("    ↳ rank_esperado.png", flush=True)

    # 4) Score top-1 por query (heatmap; comparar DENTRO do modelo) --------------
    labels_q = [it["q"] for it in queries]
    matriz = [[next((r["top1"] for r in registros
                     if r["combo"] == rot and r["query"] == q), float("nan"))
               for rot in rotulos] for q in labels_q]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.7 * len(rotulos) + 3), 0.55 * len(labels_q) + 2))
    im = _heatmap(ax, matriz, labels_q, rotulos,
                  "Score (cosseno) do top-1 por query\n"
                  "⚠ escalas de modelos diferentes NÃO são comparáveis — "
                  "compare colunas do MESMO modelo",
                  "viridis", "{:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_top1.png"), dpi=130)
    plt.close(fig)
    print("    ↳ score_top1.png", flush=True)

    print(f">>> Gráficos prontos em {_hms(time.perf_counter() - tg)} "
          f"({out_dir}/: granularidade, hit_at_k, rank_esperado, score_top1).")


def main():
    global COMBOS

    ap = argparse.ArgumentParser(description="Avaliação: modelo × chunking")
    ap.add_argument("--k", type=int, default=3, help="resultados por combinação (default 3)")
    ap.add_argument("--modelos", default="bge",
                    help="apelidos separados por vírgula: "
                         f"{', '.join(CANDIDATOS)} (default: bge)")
    ap.add_argument("--chunkings", default="ambos", choices=sorted(CHUNKINGS),
                    help="estratégias de chunking a comparar (default: ambos). "
                         "Use 'estrutural' ao comparar modelos para o servidor.")
    ap.add_argument("--tipos", default="",
                    help="restringe o corpus por tipo de bula: "
                         f"{', '.join(TIPOS_APELIDO)} (default: todos)")
    ap.add_argument("--um-por-remedio", action="store_true",
                    help="indexa só UMA bula por princípio ativo (o corpus tem ~3 "
                         "de cada). Corta o tempo de indexação por ~3.")
    ap.add_argument("--rebuild", action="store_true",
                    help="reconstrói TODAS as coleções da seleção atual")
    ap.add_argument("--no-build", action="store_true",
                    help="não constrói nada; avalia só as coleções que já existem")
    ap.add_argument("--out", default="graficos", help="diretório de saída dos PNGs")
    args = ap.parse_args()

    apelidos_tipo = [t.strip() for t in args.tipos.split(",") if t.strip()]
    for apelido in apelidos_tipo:
        if apelido not in TIPOS_APELIDO:
            raise SystemExit(f"Tipo desconhecido: {apelido!r}. "
                             f"Conhecidos: {', '.join(TIPOS_APELIDO)}")
    tipos = {TIPOS_APELIDO[a] for a in apelidos_tipo}

    # Recorte do corpus -> sufixo das coleções. Corpus cheio reaproveita os nomes
    # de produção; qualquer recorte ganha nomes próprios, então o benchmark não
    # tem como sobrescrever o índice que a API usa.
    sufixo = _perfil_corpus(tipos, args.um_por_remedio)
    COMBOS = montar_combos([m.strip() for m in args.modelos.split(",") if m.strip()],
                           args.chunkings, sufixo)

    itens = _itens_trabalho(tipos, args.um_por_remedio)
    if not itens:
        raise SystemExit("Nenhum medicamento encontrado com esse recorte de corpus.")
    queries = queries_aplicaveis(itens)
    n_gab = sum(1 for q in queries if q.get("esperado"))
    print(f">>> Corpus: {len(itens)} medicamentos"
          f"{' (' + ', '.join(sorted(tipos)) + ')' if tipos else ''}"
          f"{', 1 bula por princípio ativo' if args.um_por_remedio else ''}"
          f" | {len(queries)} queries ({n_gab} com gabarito)"
          f" | coleções sufixo {sufixo or '(produção)'}")

    load_dotenv()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    t_total = time.perf_counter()
    print(">>> Preparando coleções (constrói o que faltar)...")
    t_build = time.perf_counter()
    colls = {}
    for combo in COMBOS:
        if args.no_build:
            nomes = [c.name for c in client.list_collections()]
            if combo["colecao"] in nomes:
                colls[combo["rotulo"]] = (client.get_collection(combo["colecao"]),
                                          get_modelo(combo["modelo"]))
            else:
                print(f"[falta] {combo['rotulo']:<18} '{combo['colecao']}' não existe.")
                colls[combo["rotulo"]] = (None, None)
        else:
            coll = construir(combo, client, args.rebuild, itens)
            colls[combo["rotulo"]] = (coll, get_modelo(combo["modelo"]))
    print(f">>> Coleções prontas em {_hms(time.perf_counter() - t_build)}.")

    registros = avaliar(colls, args.k, queries)
    gerar_graficos(registros, args.k, args.out, itens, queries)
    print(f"\n✔ Tudo concluído em {_hms(time.perf_counter() - t_total)}.")


if __name__ == "__main__":
    main()
