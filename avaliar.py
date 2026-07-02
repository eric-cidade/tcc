"""Avaliação 2×2: {bge-m3, gte} × {chunking simples, estrutural}.

Constrói (se faltar) as coleções do Chroma e compara o retrieval semântico PURO
— cosseno, SEM expansão de sinônimos e SEM rerank — numa lista de queries leigas
do domínio. Tirar os sinônimos/rerank é proposital: eles são camadas de
compensação que mascarariam a diferença entre modelo e chunking, justamente o
que queremos medir aqui. Ao final, gera gráficos comparativos em ./graficos.

As quatro combinações (coleções no mesmo ./chroma_db):
  corpop_saude              bge-m3 + chunking simples    (JÁ EXISTE — não rebuilda)
  corpop_saude_bge_estr     bge-m3 + chunking estrutural
  corpop_saude_gte_simples  gte    + chunking simples
  corpop_saude_gte          gte    + chunking estrutural

Rodar:
  uv run python avaliar.py                # constrói o que faltar, avalia e plota
  uv run python avaliar.py --k 5          # top-5 por combinação
  uv run python avaliar.py --rebuild      # reconstrói TODAS (inclui corpop_saude)
  uv run python avaliar.py --no-build     # só avalia o que já existe
  uv run python avaliar.py --out graficos # diretório de saída dos PNGs

Cuidado de leitura: scores de cosseno de modelos DIFERENTES não estão na mesma
escala — compare score entre chunkings DO MESMO modelo; entre modelos, olhe a
posição (rank) e o trecho recuperado, não o número absoluto.
"""
import argparse
import os
import sys
import time

import pandas as pd
import chromadb
from dotenv import load_dotenv

from bulas import TIPOS, parse_bula, preparar_chunks
from modelos import carregar_modelo

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # backend sem display (salva PNG direto)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CHROMA_PATH = "./chroma_db"

BGE = "BAAI/bge-m3"
GTE = "Alibaba-NLP/gte-multilingual-base"

COMBOS = [
    {"rotulo": "bge + simples",    "modelo": BGE, "estrutural": False, "colecao": "corpop_saude"},
    {"rotulo": "bge + estrutural", "modelo": BGE, "estrutural": True,  "colecao": "corpop_saude_bge_estr"},
    {"rotulo": "gte + simples",    "modelo": GTE, "estrutural": False, "colecao": "corpop_saude_gte_simples"},
    {"rotulo": "gte + estrutural", "modelo": GTE, "estrutural": True,  "colecao": "corpop_saude_gte"},
]

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
    {"q": "losartana", "esperado": "losartana"},
    {"q": "maleato de enalapril", "esperado": "enalapril"},
    {"q": "propranolol", "esperado": "propranolol"},
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


def _itens_trabalho():
    """Lista (cfg, n_id, nome) dos medicamentos com os dois arquivos presentes."""
    itens = []
    for cfg in TIPOS:
        if not os.path.exists(cfg["map_csv"]):
            continue
        df = pd.read_csv(cfg["map_csv"], dtype={"id": str})
        for n_id, nome_rem in zip(df["id"], df["nome"]):
            path_o = os.path.join(cfg["path_original"], f"{n_id}{cfg['suffix_orig']}")
            path_s = os.path.join(cfg["path_simplificada"], f"{n_id}{cfg['suffix_simp']}")
            if os.path.exists(path_o) and os.path.exists(path_s):
                itens.append((cfg, n_id, nome_rem, path_o, path_s))
    return itens


def construir(combo, client, rebuild):
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
    itens = _itens_trabalho()
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
            embs = modelo.encode([c["embed"] for c in chunks]).tolist()
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
    qv = modelo.encode(query).tolist()
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


def avaliar(colls, k):
    """Roda as queries em cada combinação, imprime os resultados e coleta métricas."""
    registros = []  # um dict por (combinação, query)
    t_aval = time.perf_counter()
    print(f"\n>>> Avaliando {len(QUERIES)} queries × {len(COMBOS)} combinações...")
    for nq, item in enumerate(QUERIES, 1):
        q, esperado = item["q"], item.get("esperado")
        tq = time.perf_counter()
        print("\n" + "═" * 78)
        print(f"QUERY {nq}/{len(QUERIES)}: {q!r}"
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

def _coletar_granularidade():
    """Tamanho e nº de chunks por documento, por estratégia (independe do modelo)."""
    dados = {"simples": {"counts": [], "lens": []},
             "estrutural": {"counts": [], "lens": []}}
    for cfg in TIPOS:
        if not os.path.exists(cfg["map_csv"]):
            continue
        df = pd.read_csv(cfg["map_csv"], dtype={"id": str})
        for n_id in df["id"]:
            for sub, suf in (("path_original", "suffix_orig"), ("path_simplificada", "suffix_simp")):
                caminho = os.path.join(cfg[sub], f"{n_id}{cfg[suf]}")
                if not os.path.exists(caminho):
                    continue
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


def gerar_graficos(registros, k, out_dir):
    if plt is None:
        print("\n[gráficos] matplotlib/numpy ausentes — pulei. "
              "Instale com: uv pip install matplotlib", file=sys.stderr)
        return
    print(f"\n>>> Gerando gráficos em {out_dir}/...")
    tg = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    rotulos = [c["rotulo"] for c in COMBOS]

    # 1) Granularidade do chunking (depende só da estratégia) -------------------
    print("  • granularidade (recalculando chunks de todas as bulas)...", flush=True)
    gran = _coletar_granularidade()
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
        fig, ax = plt.subplots(figsize=(7, 4.2))
        barras = ax.bar(rotulos, hits, color=["#c0392b", "#e67e22", "#2980b9", "#27ae60"])
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
        fig, ax = plt.subplots(figsize=(8, 0.7 * len(labq) + 2))
        im = _heatmap(ax, matriz, labq, rotulos,
                      "Posição do medicamento esperado (1 = topo; menor é melhor)",
                      "RdYlGn_r", "{:.0f}", vmin=1, vmax=max(2, k))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rank")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "rank_esperado.png"), dpi=130)
        plt.close(fig)
        print("    ↳ rank_esperado.png", flush=True)

    # 4) Score top-1 por query (heatmap; comparar DENTRO do modelo) --------------
    queries = [it["q"] for it in QUERIES]
    matriz = [[next((r["top1"] for r in registros
                     if r["combo"] == rot and r["query"] == q), float("nan"))
               for rot in rotulos] for q in queries]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(queries) + 2))
    im = _heatmap(ax, matriz, queries, rotulos,
                  "Score (cosseno) do top-1 por query\n"
                  "⚠ escalas de bge e gte diferem — compare colunas do MESMO modelo",
                  "viridis", "{:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_top1.png"), dpi=130)
    plt.close(fig)
    print("    ↳ score_top1.png", flush=True)

    print(f">>> Gráficos prontos em {_hms(time.perf_counter() - tg)} "
          f"({out_dir}/: granularidade, hit_at_k, rank_esperado, score_top1).")


def main():
    ap = argparse.ArgumentParser(description="Avaliação 2×2: modelo × chunking")
    ap.add_argument("--k", type=int, default=3, help="resultados por combinação (default 3)")
    ap.add_argument("--rebuild", action="store_true",
                    help="reconstrói TODAS as coleções, inclusive corpop_saude")
    ap.add_argument("--no-build", action="store_true",
                    help="não constrói nada; avalia só as coleções que já existem")
    ap.add_argument("--out", default="graficos", help="diretório de saída dos PNGs")
    args = ap.parse_args()

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
            coll = construir(combo, client, args.rebuild)
            colls[combo["rotulo"]] = (coll, get_modelo(combo["modelo"]))
    print(f">>> Coleções prontas em {_hms(time.perf_counter() - t_build)}.")

    registros = avaliar(colls, args.k)
    gerar_graficos(registros, args.k, args.out)
    print(f"\n✔ Tudo concluído em {_hms(time.perf_counter() - t_total)}.")


if __name__ == "__main__":
    main()
