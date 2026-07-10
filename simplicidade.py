"""Comparador de simplicidade lexical baseado no corpus paralelo de bulas.

Dado um par de termos (ex.: "cefaleia" vs "dor de cabeça"), decide qual é o
mais simples usando como evidência o próprio corpus original/simplificada:
um termo relativamente mais frequente nas bulas SIMPLIFICADAS do que nas
ORIGINAIS é, por construção do corpus, o que os simplificadores preferiram —
logo, o mais acessível. É a abordagem clássica de simplificação lexical por
estatística de corpus (razão de frequências entre corpus complexo e simples),
com heurísticas de superfície (sílabas, comprimento) como desempate.

Não carrega modelo nem conecta em motor de busca nenhum: só lê os .txt do
corpus (uma vez, em memória) e conta tokens. Suporta termos multipalavra
("dor de cabeça"), que são contados como sequência de tokens.

CLI:  uv run python simplicidade.py cefaleia "dor de cabeça"
API:  GET /simplicidade?a=cefaleia&b=dor%20de%20cabeça  (ver api.py)
"""
import argparse
import math
import os
import re
from collections import Counter

from bulas import TIPOS, parse_bula

# Tokens = sequências de letras PT (com acentos), aceitando hífen interno
# ("pós-operatório" é UM token). O texto é minusculizado antes, então o padrão
# só precisa cobrir minúsculas.
_RE_TOKEN = re.compile(r"[a-záéíóúâêôãõàüç]+(?:-[a-záéíóúâêôãõàüç]+)*")

# Margem mínima (em bits) entre os scores de corpus para o corpus decidir
# sozinho. Diferenças menores que isso são ruído de amostragem num corpus de
# ~80 pares; abaixo dela caímos para familiaridade (frequência total) e
# heurísticas de superfície.
MARGEM_SCORE = 0.5

# Corpus carregado sob demanda por _carregar(). Por lado ('original' /
# 'simplificada'): lista de listas de tokens (uma por documento), lista dos
# textos crus (para extrair frases de exemplo), Counter de unigramas e total
# de tokens do lado.
_CORPUS = None


def _tokenizar(texto):
    return _RE_TOKEN.findall(texto.lower())


def _carregar():
    """Lê todas as bulas do corpus e monta as contagens por lado (uma vez)."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS

    lados = {
        "original": {"docs": [], "textos": []},
        "simplificada": {"docs": [], "textos": []},
    }
    for cfg in TIPOS:
        for lado, path, sufixo in (
            ("original", cfg["path_original"], cfg["suffix_orig"]),
            ("simplificada", cfg["path_simplificada"], cfg["suffix_simp"]),
        ):
            if not os.path.isdir(path):
                continue
            for arq in sorted(os.listdir(path)):
                if not arq.endswith(sufixo):
                    continue
                with open(os.path.join(path, arq), "r", encoding="utf-8") as f:
                    _, texto = parse_bula(f.read())
                lados[lado]["textos"].append(texto)
                lados[lado]["docs"].append(_tokenizar(texto))

    for lado in lados.values():
        lado["unigramas"] = Counter(t for doc in lado["docs"] for t in doc)
        lado["total"] = sum(len(doc) for doc in lado["docs"])

    _CORPUS = lados
    return _CORPUS


def _contar_no_lado(termo_tokens, lado):
    """(ocorrências, nº de documentos com o termo) de `termo_tokens` num lado.

    Unigramas saem direto do Counter; termos multipalavra são contados como
    subsequência exata de tokens dentro de cada documento (o que ignora
    pontuação/caixa entre as palavras, de propósito).
    """
    k = len(termo_tokens)
    if k == 1:
        alvo = termo_tokens[0]
        occ = lado["unigramas"][alvo]
        docs = sum(1 for doc in lado["docs"] if alvo in doc) if occ else 0
        return occ, docs

    occ = doc_freq = 0
    for doc in lado["docs"]:
        n = sum(1 for i in range(len(doc) - k + 1) if doc[i:i + k] == termo_tokens)
        occ += n
        doc_freq += 1 if n else 0
    return occ, doc_freq


def _exemplo(termo, textos):
    """Primeira sentença do lado que contém o termo (evidência de uso), ou None."""
    padrao = re.compile(
        r"\b" + r"\s+".join(re.escape(p) for p in termo.split()) + r"\b",
        re.IGNORECASE,
    )
    for texto in textos:
        for sent in re.split(r"(?<=[.!?;])\s+|\n+", texto):
            sent = sent.strip()
            if len(sent) > 15 and padrao.search(sent):
                return sent
    return None


def _silabas(termo):
    """Nº aproximado de sílabas (PT): grupos de vogais contíguas, por palavra.

    Ditongos contam como uma sílaba e hiatos são subestimados ("saúde" sai com
    2 em vez de 3) — suficiente como heurística de desempate, não como análise
    fonológica. Para termos multipalavra, soma as palavras.
    """
    vogais = set("aeiouáéíóúâêôãõàü")
    grupos, anterior_vogal = 0, False
    for ch in termo.lower():
        eh_vogal = ch in vogais
        if eh_vogal and not anterior_vogal:
            grupos += 1
        anterior_vogal = eh_vogal
    return max(grupos, 1)


def estatisticas(termo):
    """Estatísticas de corpus de um termo: frequências por lado + score.

    O `score_simplicidade` é a razão log2 (com suavização de Laplace) entre a
    frequência relativa do termo no lado simplificado e no original:
      > 0  → termo característico das bulas simplificadas (mais simples);
      < 0  → característico das originais (mais técnico);
      = 0  → ausente do corpus (fixado em 0 por definição: sem evidência).
    Sem esse ajuste do caso ausente, a suavização daria um score positivo
    espúrio (~log2 da razão de tamanhos dos lados, já que as simplificadas
    são bem mais curtas), fazendo palavras inventadas parecerem "simples".
    """
    corpus = _carregar()
    termo_norm = " ".join(_tokenizar(termo))
    tokens = termo_norm.split()
    if not tokens:
        raise ValueError(f"Termo vazio ou sem letras: {termo!r}")

    occ_o, docs_o = _contar_no_lado(tokens, corpus["original"])
    occ_s, docs_s = _contar_no_lado(tokens, corpus["simplificada"])
    n_o = corpus["original"]["total"]
    n_s = corpus["simplificada"]["total"]

    # Frequência por milhão de tokens, comparável entre lados de tamanhos
    # diferentes (as simplificadas são bem mais curtas que as originais).
    fpm_o = occ_o / n_o * 1e6
    fpm_s = occ_s / n_s * 1e6
    if occ_o + occ_s == 0:
        score = 0.0
    else:
        score = math.log2(((occ_s + 0.5) / (n_s + 1)) / ((occ_o + 0.5) / (n_o + 1)))

    return {
        "termo": termo_norm,
        "no_corpus": (occ_o + occ_s) > 0,
        "original": {"ocorrencias": occ_o, "documentos": docs_o,
                     "freq_por_milhao": round(fpm_o, 2)},
        "simplificada": {"ocorrencias": occ_s, "documentos": docs_s,
                         "freq_por_milhao": round(fpm_s, 2)},
        "score_simplicidade": round(score, 3),
        # Familiaridade: quão comum o termo é no corpus como um todo (palavras
        # frequentes tendem a ser mais conhecidas do leitor leigo).
        "freq_total_por_milhao": round((occ_o + occ_s) / (n_o + n_s) * 1e6, 2),
        "silabas": _silabas(termo_norm),
        "caracteres": len(termo_norm),
        "exemplo_simplificada": _exemplo(termo_norm, corpus["simplificada"]["textos"]),
        "exemplo_original": _exemplo(termo_norm, corpus["original"]["textos"]),
    }


def comparar(termo_a, termo_b):
    """Compara dois termos e aponta o mais simples, com a evidência usada.

    Cascata de decisão (campo `criterio` diz qual regra bateu o martelo):
      1. 'corpus'        — os DOIS termos aparecem no corpus e
                           |Δscore| ≥ MARGEM_SCORE: vence o de maior score.
                           (Exigir os dois evita comparar evidência real com
                           o score neutro de um termo nunca visto.)
      2. 'familiaridade' — scores empatados ou só um termo no corpus; vence o
                           mais frequente no corpus como um todo (dif. > 20%).
      3. 'heuristica'    — sem evidência de corpus suficiente; vence o de
                           menos sílabas, depois o mais curto.
      4. 'empate'        — nada distingue os termos.
    """
    a = estatisticas(termo_a)
    b = estatisticas(termo_b)

    mais_simples, criterio = None, "empate"
    delta = a["score_simplicidade"] - b["score_simplicidade"]
    if a["no_corpus"] and b["no_corpus"] and abs(delta) >= MARGEM_SCORE:
        mais_simples = a if delta > 0 else b
        criterio = "corpus"
    elif a["no_corpus"] or b["no_corpus"]:
        fa, fb = a["freq_total_por_milhao"], b["freq_total_por_milhao"]
        if max(fa, fb) > 1.2 * min(fa, fb):
            mais_simples = a if fa > fb else b
            criterio = "familiaridade"

    if mais_simples is None and a["silabas"] != b["silabas"]:
        mais_simples = a if a["silabas"] < b["silabas"] else b
        criterio = "heuristica"
    if mais_simples is None and a["caracteres"] != b["caracteres"]:
        mais_simples = a if a["caracteres"] < b["caracteres"] else b
        criterio = "heuristica"

    return {
        "a": a,
        "b": b,
        "mais_simples": mais_simples["termo"] if mais_simples else None,
        "criterio": criterio,
        "delta_score": round(delta, 3),
    }


# --- Interface de Terminal (CLI) ---
def _imprimir_termo(rotulo, est):
    print(f"\n[{rotulo}] «{est['termo']}»")
    if not est["no_corpus"]:
        print("  (termo NÃO encontrado no corpus — score não é confiável)")
    print(f"  Originais:      {est['original']['ocorrencias']} ocorrências em "
          f"{est['original']['documentos']} bulas "
          f"({est['original']['freq_por_milhao']}/milhão)")
    print(f"  Simplificadas:  {est['simplificada']['ocorrencias']} ocorrências em "
          f"{est['simplificada']['documentos']} bulas "
          f"({est['simplificada']['freq_por_milhao']}/milhão)")
    print(f"  Score simplicidade: {est['score_simplicidade']:+.3f}  "
          f"| familiaridade: {est['freq_total_por_milhao']}/milhão  "
          f"| sílabas: {est['silabas']}  | caracteres: {est['caracteres']}")
    if est["exemplo_simplificada"]:
        print(f"  Ex. (simplificada): {est['exemplo_simplificada'][:120]}")
    elif est["exemplo_original"]:
        print(f"  Ex. (original): {est['exemplo_original'][:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara a simplicidade de dois termos usando o corpus "
                    "paralelo de bulas (original × simplificada).")
    parser.add_argument("termo_a", type=str, help="Primeiro termo (pode ser multipalavra)")
    parser.add_argument("termo_b", type=str, help="Segundo termo")
    args = parser.parse_args()

    res = comparar(args.termo_a, args.termo_b)

    print("═" * 60)
    print(f"SIMPLICIDADE: «{res['a']['termo']}» × «{res['b']['termo']}»")
    print("═" * 60)
    _imprimir_termo("A", res["a"])
    _imprimir_termo("B", res["b"])

    print("\n" + "─" * 60)
    if res["mais_simples"]:
        print(f"Mais simples: «{res['mais_simples']}» "
              f"(critério: {res['criterio']}, Δscore = {res['delta_score']:+.3f})")
    else:
        print(f"Empate — nada distingue os termos (Δscore = {res['delta_score']:+.3f}).")
