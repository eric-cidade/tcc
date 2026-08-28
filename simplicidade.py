"""Comparador de simplicidade lexical baseado no corpus paralelo de bulas.

Dado um par de termos (ex.: "cefaleia" vs "dor de cabeça"), decide qual é o
mais simples usando como evidência o próprio corpus original/simplificada:
um termo relativamente mais frequente nas bulas SIMPLIFICADAS do que nas
ORIGINAIS é, por construção do corpus, o que os simplificadores preferiram —
logo, o mais acessível. É a abordagem clássica de simplificação lexical por
estatística de corpus (razão de frequências entre corpus complexo e simples),
com duas correções calibradas neste corpus (ver SIMPLICIDADE.md):

  1. Ocorrências entre PARÊNTESES não contam. Na bula simplificada o padrão
     dominante é a glosa "termo leigo (termo técnico)" — a versão validada é
     obrigada a exibir o termo técnico —, então contá-las faz o termo técnico
     parecer simples ("hipertensão" tem 43 de 62 ocorrências nas simplificadas
     dentro de parênteses).
  2. Estar atestado nas simplificadas vale um bônus. A mera presença do termo
     na versão validada já é indício forte de simplicidade, então o score soma
     à razão de frequências um bônus que cresce com o número de bulas
     simplificadas DISTINTAS em que o termo aparece (dispersão).

O comprimento do termo entra só como desempate, quando o corpus não tem
evidência nenhuma.

A comparação pode ser restrita por DOIS eixos cruzáveis (ver bulas.py):

  - ESCOPO (tema/gênero): "bula" compara dentro de todas as bulas,
    "bula/hipertensao" só dentro das bulas de hipertensão, e assim por diante
    para gêneros de documento que venham a ser acrescentados;
  - PROCEDÊNCIA do lado simplificado: "humana" (validada por linguistas) ou
    "ia" (gerada por IA, sem revisão profissional).

Os dois são independentes — "só o que foi validado por humanos" é uma pergunta
diferente de "só oncologia" —, e cada um omitido significa "qualquer". Isso
importa porque é o lado simplificado que define o que o score chama de simples:
misturar procedências mistura as referências.

Não carrega modelo nem conecta em motor de busca nenhum: só lê os .txt do
corpus (uma vez, em memória) e conta tokens. Suporta termos multipalavra
("dor de cabeça"), que são contados como sequência de tokens.

CLI:  uv run python simplicidade.py cefaleia "dor de cabeça"
      uv run python simplicidade.py náusea enjoo --proveniencia humana
      uv run python simplicidade.py cefaleia "dor de cabeça" --escopo bula/oncologia
      uv run python simplicidade.py --listar-escopos
API:  GET /simplicidade?a=cefaleia&b=dor%20de%20cabeça&escopo=bula&proveniencia=humana
"""
import argparse
import math
import os
import re
from collections import Counter

from bulas import (TIPOS, arvore_escopos, documento_casa, escopo_casa,
                   normalizar_escopo, normalizar_proveniencia, parse_bula,
                   proveniencias_de, proveniencias_disponiveis, rotulo_escopo)

# Tokens = sequências de letras PT (com acentos), aceitando hífen interno
# ("pós-operatório" é UM token). O texto é minusculizado antes, então o padrão
# só precisa cobrir minúsculas.
_RE_TOKEN = re.compile(r"[a-záéíóúâêôãõàüç]+(?:-[a-záéíóúâêôãõàüç]+)*")

# Margem mínima (em bits) entre os scores de corpus para o corpus decidir
# sozinho. Diferenças menores que isso são ruído de amostragem num corpus de
# ~80 pares; abaixo dela caímos para familiaridade (frequência total) e
# heurísticas de superfície. Num escopo restrito (menos bulas) a margem é ainda
# mais otimista — a resposta traz o tamanho do escopo justamente para isso.
MARGEM_SCORE = 0.5

# Peso máximo (em bits) do bônus de atestação — o quanto "aparecer nas bulas
# simplificadas" pode, sozinho, empurrar o score para cima. O bônus vai de 0
# (termo ausente das simplificadas) a PESO_ATESTACAO (presente em todas as
# bulas simplificadas do escopo), crescendo em log com o nº de bulas distintas.
# Como o que decide uma comparação é a DIFERENÇA de bônus entre os dois termos,
# ele quase se cancela quando ambos são atestados e pesa justamente no caso que
# motiva a regra: um termo atestado na versão validada contra outro que não é.
PESO_ATESTACAO = 1.5

# Documentos do corpus, lidos e tokenizados uma única vez (_carregar_docs).
_DOCS = None
# Índices (contagens) por escopo, montados sob demanda: {escopo ou "": lados}.
_INDICES = {}

_ROTULO_CORPUS_INTEIRO = "Todo o corpus"


def _spans_parenteses(texto):
    """Trechos (início, fim) entre parênteses, só os mais externos.

    Varredura com pilha, para que "(a (b) c)" conte como UM trecho. Parêntese
    que abre e nunca fecha é ignorado — assim um "(" solto não engole o resto
    da bula.
    """
    spans, pilha = [], []
    for i, ch in enumerate(texto):
        if ch == "(":
            pilha.append(i)
        elif ch == ")" and pilha:
            ini = pilha.pop()
            if not pilha:
                spans.append((ini, i))
    return spans


def _tokenizar(texto):
    return _RE_TOKEN.findall(texto.lower())


def _tokenizar_marcado(texto):
    """(tokens, dentro_de_parenteses) — duas listas alinhadas pelo índice.

    Como `_spans_parenteses` devolve os trechos em ordem, basta um ponteiro
    avançando junto com os tokens (nada de busca por trecho a cada token).
    """
    texto = texto.lower()
    spans = _spans_parenteses(texto)
    tokens, dentro = [], []
    i = 0
    for m in _RE_TOKEN.finditer(texto):
        while i < len(spans) and spans[i][1] < m.start():
            i += 1
        tokens.append(m.group())
        dentro.append(i < len(spans) and spans[i][0] < m.start() < spans[i][1])
    return tokens, dentro


def _carregar_docs():
    """Lê e tokeniza todas as bulas do corpus (uma vez), marcando o escopo.

    Guarda os documentos crus por lado; as CONTAGENS de cada escopo são
    montadas depois, por `_indice`. Assim a leitura de disco e a tokenização —
    a parte cara — acontecem uma vez só, mesmo que o usuário compare termos em
    vários escopos na mesma sessão.
    """
    global _DOCS
    if _DOCS is not None:
        return _DOCS

    docs = {"original": [], "simplificada": []}
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
                tokens, dentro = _tokenizar_marcado(texto)
                docs[lado].append({
                    "escopo": cfg["escopo"],
                    "proveniencia": cfg["proveniencia"],
                    "tipo": cfg["tipo"],
                    "texto": texto,
                    "tokens": tokens,
                    "par": dentro,
                    # Tokens do doc FORA de parênteses, para a frequência
                    # documental de unigrama sair sem varrer a lista inteira.
                    "fora": {t for t, p in zip(tokens, dentro) if not p},
                })

    _DOCS = docs
    return _DOCS


def _indice(escopo=None, proveniencia=None):
    """Contagens do corpus restritas ao cruzamento tema × procedência.

    Cada eixo em None = "qualquer". Memoizado pela combinação: montar o índice
    é só refiltrar documentos já tokenizados e refazer os Counters.

    O filtro de procedência vale para os DOIS lados, e não só para o
    simplificado: o par original/simplificada de uma bula é a unidade de
    evidência, e comparar o lado simplificado de um subcorpus com o original de
    outro produziria uma razão de frequências sem sentido.
    """
    chave = (escopo or "", proveniencia or "")
    if chave in _INDICES:
        return _INDICES[chave]

    docs = _carregar_docs()
    lados = {}
    for nome in ("original", "simplificada"):
        selecionados = [d for d in docs[nome]
                        if documento_casa(d["escopo"], d["proveniencia"],
                                          escopo, proveniencia)]
        lado = {
            "docs": selecionados,
            "textos": [d["texto"] for d in selecionados],
            "unigramas": Counter(
                t for d in selecionados for t, p in zip(d["tokens"], d["par"]) if not p),
            "unigramas_par": Counter(
                t for d in selecionados for t, p in zip(d["tokens"], d["par"]) if p),
            "n_docs": len(selecionados),
        }
        # Total do lado = tokens FORA de parênteses: é a base contra a qual as
        # ocorrências (também fora) são normalizadas.
        lado["total"] = sum(lado["unigramas"].values())
        lado["total_parenteses"] = sum(lado["unigramas_par"].values())
        lados[nome] = lado

    for nome, lado in lados.items():
        if lado["total"] == 0:
            recorte = f"escopo={escopo or 'todos'}, procedência={proveniencia or 'qualquer'}"
            raise ValueError(
                f"Recorte vazio ({recorte}): nenhum documento do lado '{nome}'. "
                f"As combinações que existem estão em /escopos.")

    _INDICES[chave] = lados
    return lados


def _contar_no_lado(termo_tokens, lado):
    """(ocorrências, ocorrências entre parênteses, nº de docs) num lado.

    As duas contagens são disjuntas: a primeira só conta ocorrências em texto
    corrido — é ela que alimenta o score —, e a segunda registra as que caem
    dentro de parênteses, devolvidas à parte como evidência (na bula
    simplificada elas são, quase sempre, o termo técnico obrigatório entre
    parênteses depois da paráfrase leiga). Uma ocorrência multipalavra que
    encoste em parêntese conta como parentética: não é uso corrido limpo.

    Unigramas saem direto dos Counters; termos multipalavra são contados como
    subsequência exata de tokens dentro de cada documento (o que ignora
    pontuação/caixa entre as palavras, de propósito).
    """
    k = len(termo_tokens)
    if k == 1:
        alvo = termo_tokens[0]
        occ = lado["unigramas"][alvo]
        occ_par = lado["unigramas_par"][alvo]
        docs = sum(1 for d in lado["docs"] if alvo in d["fora"]) if occ else 0
        return occ, occ_par, docs

    occ = occ_par = doc_freq = 0
    for d in lado["docs"]:
        tokens, par = d["tokens"], d["par"]
        n = n_par = 0
        for i in range(len(tokens) - k + 1):
            if tokens[i:i + k] == termo_tokens:
                if any(par[i:i + k]):
                    n_par += 1
                else:
                    n += 1
        occ += n
        occ_par += n_par
        doc_freq += 1 if n else 0
    return occ, occ_par, doc_freq


def _sentenca_em(texto, pos):
    """Sentença de `texto` que contém a posição `pos` (corte por .!?; e \\n)."""
    ini = max(texto.rfind(c, 0, pos) for c in ".!?;\n")
    fins = [p for p in (texto.find(c, pos) for c in ".!?;\n") if p != -1]
    return texto[ini + 1:min(fins) if fins else len(texto)].strip()


def _exemplo(termo, textos):
    """Primeira sentença do lado que contém o termo (evidência de uso), ou None.

    Só ocorrências fora de parênteses valem: o exemplo tem que ilustrar o mesmo
    uso que o score contou, não a glosa "(termo)".
    """
    padrao = re.compile(
        r"\b" + r"\s+".join(re.escape(p) for p in termo.split()) + r"\b",
        re.IGNORECASE,
    )
    for texto in textos:
        achados = list(padrao.finditer(texto))
        if not achados:
            continue
        spans = _spans_parenteses(texto)
        for m in achados:
            if any(ini < m.start() < fim for ini, fim in spans):
                continue
            sent = _sentenca_em(texto, m.start())
            if len(sent) > 15:
                return sent
    return None


def _bonus_atestacao(docs_s, n_docs_s):
    """Bônus (em bits) por estar atestado nas bulas simplificadas do escopo.

    Vale 0 para termo ausente das simplificadas e cresce até PESO_ATESTACAO
    com o nº de bulas simplificadas DISTINTAS que usam o termo — dispersão, não
    frequência bruta: 12 ocorrências espalhadas por 12 bulas são evidência bem
    mais forte do que 12 na mesma bula (que pode ser um medicamento cujo tema é
    aquele termo). O log satura: a primeira bula já vale boa parte do bônus e a
    quadragésima acrescenta pouco. A normalização é pelo tamanho do ESCOPO, e
    não do corpus inteiro, para que "presente em metade das bulas" valha o
    mesmo dentro de qualquer recorte.
    """
    if docs_s <= 0 or n_docs_s <= 0:
        return 0.0
    return PESO_ATESTACAO * math.log2(1 + docs_s) / math.log2(1 + n_docs_s)


def escopos():
    """Árvore de escopos disponíveis, com o tamanho de cada nó.

    O cliente monta o filtro a partir daqui — nada de gênero hardcodado — e
    mostra quantos documentos há em cada recorte, que é o que permite ao
    usuário desconfiar de um score calculado sobre meia dúzia de textos.
    Conta a partir dos documentos realmente lidos do disco, então um escopo
    declarado mas ainda sem arquivos aparece com zero (e não quebra a lista).
    """
    docs = _carregar_docs()

    def enriquecer(no):
        no = dict(no)
        no["documentos"] = {
            lado: sum(1 for d in docs[lado] if escopo_casa(d["escopo"], no["id"]))
            for lado in docs
        }
        # Quantos documentos o nó tem de CADA procedência: é o que permite ao
        # cliente desabilitar combinações vazias dos dois eixos (ex.: bulas de
        # hipertensão × simplificação por IA não existe).
        no["documentos_por_proveniencia"] = {
            p["id"]: sum(1 for d in docs["simplificada"]
                         if documento_casa(d["escopo"], d["proveniencia"], no["id"], p["id"]))
            for p in proveniencias_disponiveis()
        }
        no["filhos"] = [enriquecer(f) for f in no["filhos"]]
        return no

    return [enriquecer(no) for no in arvore_escopos()]


def proveniencias():
    """Eixo de procedência disponível, para o cliente montar o segundo filtro."""
    return proveniencias_disponiveis()


def resumo_corpus(escopo=None, proveniencia=None):
    """Tamanho do recorte de corpus usado numa comparação.

    Vai na resposta porque o score só é interpretável com ele à vista: os
    mesmos dois termos podem trocar de lugar entre "todas as bulas" e "só
    oncologia", e um escopo pequeno torna a margem de decisão bem mais
    otimista do que ela é.

    Traz também a PROCEDÊNCIA do lado simplificado do recorte. O score toma
    esse lado como referência do que é linguagem acessível; se ele foi gerado
    por IA sem revisão, a referência é o que a IA deixou passar, e um recorte
    que mistura procedências mistura as duas referências (`proveniencia_mista`).
    """
    escopo = normalizar_escopo(escopo)
    proveniencia = normalizar_proveniencia(proveniencia)
    idx = _indice(escopo, proveniencia)
    procs = proveniencias_de(escopo, proveniencia)
    return {
        "escopo": escopo or "",
        "rotulo": rotulo_escopo(escopo) if escopo else _ROTULO_CORPUS_INTEIRO,
        "proveniencia": proveniencia or "",
        "documentos": {lado: idx[lado]["n_docs"] for lado in idx},
        "tokens": {lado: idx[lado]["total"] for lado in idx},
        "tokens_parenteses": {lado: idx[lado]["total_parenteses"] for lado in idx},
        # Procedências efetivamente presentes no recorte — filtrar pelo eixo
        # deixa esta lista com um item só, e a mistura deixa de existir.
        "proveniencias": procs,
        "proveniencia_mista": len(procs) > 1,
    }


def estatisticas(termo, escopo=None, proveniencia=None):
    """Estatísticas de corpus de um termo: frequências por lado + score.

    `escopo` (tema) e `proveniencia` (como o lado simplificado foi produzido)
    restringem o corpus e se cruzam: None em qualquer um = "qualquer". Ver
    bulas.py.

    O `score_simplicidade` soma duas parcelas:

      score = log2( freq. relativa nas simplificadas / freq. relativa nas
                    originais )              [razão de frequências, com Laplace]
              + bônus de atestação nas simplificadas          [0 .. PESO_ATESTACAO]

      > 0  → termo característico das bulas simplificadas (mais simples);
      < 0  → característico das originais (mais técnico);
      = 0  → sem evidência utilizável no corpus.

    Todas as contagens IGNORAM ocorrências entre parênteses (reportadas à parte
    em `ocorrencias_parenteses`), inclusive nos totais de cada lado: na versão
    simplificada o parêntese é o lugar onde o termo técnico é reexibido depois
    da paráfrase leiga, então contá-lo inverteria o sinal da evidência.

    O caso sem evidência é fixado em 0 explicitamente. Sem isso, a suavização
    daria um score positivo espúrio (~log2 da razão de tamanhos dos lados, já
    que as simplificadas são bem mais curtas), fazendo palavras inventadas —
    ou termos que só aparecem entre parênteses — parecerem "simples".
    """
    escopo = normalizar_escopo(escopo)
    proveniencia = normalizar_proveniencia(proveniencia)
    corpus = _indice(escopo, proveniencia)
    termo_norm = " ".join(_tokenizar(termo))
    tokens = termo_norm.split()
    if not tokens:
        raise ValueError(f"Termo vazio ou sem letras: {termo!r}")

    occ_o, par_o, docs_o = _contar_no_lado(tokens, corpus["original"])
    occ_s, par_s, docs_s = _contar_no_lado(tokens, corpus["simplificada"])
    n_o = corpus["original"]["total"]
    n_s = corpus["simplificada"]["total"]

    # Frequência por milhão de tokens, comparável entre lados de tamanhos
    # diferentes (as simplificadas são bem mais curtas que as originais).
    fpm_o = occ_o / n_o * 1e6
    fpm_s = occ_s / n_s * 1e6
    if occ_o + occ_s == 0:
        razao = 0.0
    else:
        razao = math.log2(((occ_s + 0.5) / (n_s + 1)) / ((occ_o + 0.5) / (n_o + 1)))
    bonus = _bonus_atestacao(docs_s, corpus["simplificada"]["n_docs"])

    return {
        "termo": termo_norm,
        "escopo": escopo or "",
        "proveniencia": proveniencia or "",
        # "Estar no corpus" = ter ocorrência FORA de parênteses. Um termo que
        # só aparece entre parênteses não tem evidência utilizável, e tratá-lo
        # como atestado devolveria o score neutro 0 a um termo técnico.
        "no_corpus": (occ_o + occ_s) > 0,
        "somente_em_parenteses": (occ_o + occ_s) == 0 and (par_o + par_s) > 0,
        "original": {"ocorrencias": occ_o, "documentos": docs_o,
                     "ocorrencias_parenteses": par_o,
                     "freq_por_milhao": round(fpm_o, 2)},
        "simplificada": {"ocorrencias": occ_s, "documentos": docs_s,
                         "ocorrencias_parenteses": par_s,
                         "freq_por_milhao": round(fpm_s, 2)},
        "score_simplicidade": round(razao + bonus, 3),
        # Parcelas do score, abertas para inspeção/depuração.
        "razao_frequencias": round(razao, 3),
        "bonus_atestacao": round(bonus, 3),
        "atestado_simplificada": occ_s > 0,
        # Familiaridade: quão comum o termo é no corpus como um todo (palavras
        # frequentes tendem a ser mais conhecidas do leitor leigo).
        "freq_total_por_milhao": round((occ_o + occ_s) / (n_o + n_s) * 1e6, 2),
        "caracteres": len(termo_norm),
        "exemplo_simplificada": _exemplo(termo_norm, corpus["simplificada"]["textos"]),
        "exemplo_original": _exemplo(termo_norm, corpus["original"]["textos"]),
    }


def comparar(termo_a, termo_b, escopo=None, proveniencia=None):
    """Compara dois termos e aponta o mais simples, com a evidência usada.

    `escopo` (tema) e `proveniencia` restringem o corpus nos dois eixos (ver
    bulas.py e `estatisticas`). O recorte usado volta na resposta, em `corpus`.

    Cascata de decisão (campo `criterio` diz qual regra bateu o martelo):
      1. 'corpus'        — |Δscore| ≥ MARGEM_SCORE e o VENCEDOR tem evidência
                           no corpus: vence o de maior score. (Exigir isso do
                           vencedor evita que o score neutro 0 de um termo
                           nunca visto ganhe de um termo com evidência real de
                           ser técnico — mas deixa passar o caso oposto, em que
                           um termo atestado na versão validada ganha de um
                           termo sem nenhuma evidência.)
      2. 'familiaridade' — scores empatados ou vencedor sem evidência; vence o
                           mais frequente no corpus como um todo (dif. > 20%).
      3. 'heuristica'    — sem evidência de corpus suficiente; vence o termo
                           mais curto (nº de caracteres). É o último recurso, e
                           o mais fraco: comprimento não é simplicidade.
      4. 'empate'        — nada distingue os termos.
    """
    escopo = normalizar_escopo(escopo)
    proveniencia = normalizar_proveniencia(proveniencia)
    a = estatisticas(termo_a, escopo, proveniencia)
    b = estatisticas(termo_b, escopo, proveniencia)

    mais_simples, criterio = None, "empate"
    delta = a["score_simplicidade"] - b["score_simplicidade"]
    vencedor = a if delta > 0 else b
    if abs(delta) >= MARGEM_SCORE and vencedor["no_corpus"]:
        mais_simples = vencedor
        criterio = "corpus"
    elif a["no_corpus"] or b["no_corpus"]:
        fa, fb = a["freq_total_por_milhao"], b["freq_total_por_milhao"]
        if max(fa, fb) > 1.2 * min(fa, fb):
            mais_simples = a if fa > fb else b
            criterio = "familiaridade"

    if mais_simples is None and a["caracteres"] != b["caracteres"]:
        mais_simples = a if a["caracteres"] < b["caracteres"] else b
        criterio = "heuristica"

    return {
        "a": a,
        "b": b,
        "mais_simples": mais_simples["termo"] if mais_simples else None,
        "criterio": criterio,
        "delta_score": round(delta, 3),
        "corpus": resumo_corpus(escopo, proveniencia),
    }


# --- Interface de Terminal (CLI) ---
def _fmt_lado(est, lado):
    d = est[lado]
    par = f" [+{d['ocorrencias_parenteses']} entre parênteses, não contadas]" \
        if d["ocorrencias_parenteses"] else ""
    return (f"{d['ocorrencias']} ocorrências em {d['documentos']} documentos "
            f"({d['freq_por_milhao']}/milhão){par}")


def _imprimir_termo(rotulo, est):
    print(f"\n[{rotulo}] «{est['termo']}»")
    if est["somente_em_parenteses"]:
        print("  (termo só aparece ENTRE PARÊNTESES no escopo — sem evidência de uso corrido)")
    elif not est["no_corpus"]:
        print("  (termo NÃO encontrado no escopo — score não é confiável)")
    print(f"  Originais:      {_fmt_lado(est, 'original')}")
    print(f"  Simplificadas:  {_fmt_lado(est, 'simplificada')}")
    print(f"  Score simplicidade: {est['score_simplicidade']:+.3f}  "
          f"(razão {est['razao_frequencias']:+.3f} + atestação {est['bonus_atestacao']:+.3f})")
    print(f"  Familiaridade: {est['freq_total_por_milhao']}/milhão  "
          f"| caracteres: {est['caracteres']}")
    if est["exemplo_simplificada"]:
        print(f"  Ex. (simplificada): {est['exemplo_simplificada'][:120]}")
    elif est["exemplo_original"]:
        print(f"  Ex. (original): {est['exemplo_original'][:120]}")


def _imprimir_escopos(nos, recuo=0):
    for no in nos:
        por_prov = " · ".join(f"{pid}: {qtd}"
                              for pid, qtd in no["documentos_por_proveniencia"].items()
                              if qtd)
        print(f"  {'  ' * recuo}{no['id']:<24} {no['rotulo']:<24} "
              f"[{no['documentos']['simplificada']} documentos — {por_prov}]")
        _imprimir_escopos(no["filhos"], recuo + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara a simplicidade de dois termos usando o corpus "
                    "paralelo de bulas (original × simplificada).")
    parser.add_argument("termo_a", type=str, nargs="?",
                        help="Primeiro termo (pode ser multipalavra)")
    parser.add_argument("termo_b", type=str, nargs="?", help="Segundo termo")
    parser.add_argument("--escopo", type=str, default=None,
                        help="Recorte temático (ex.: bula, bula/oncologia). "
                             "Default: corpus inteiro.")
    parser.add_argument("--proveniencia", type=str, default=None,
                        choices=[p["id"] for p in proveniencias_disponiveis()],
                        help="Procedência do lado simplificado. "
                             "Default: qualquer (mistura as duas).")
    parser.add_argument("--listar-escopos", action="store_true",
                        help="Lista os escopos disponíveis e sai.")
    args = parser.parse_args()

    if args.listar_escopos:
        print("EIXO 1 — escopo/tema (--escopo; vazio = todos):")
        _imprimir_escopos(escopos())
        print("\nEIXO 2 — procedência do lado simplificado "
              "(--proveniencia; vazio = qualquer):")
        for pr in proveniencias_disponiveis():
            print(f"  {pr['id']:<24} {pr['rotulo']}")
        print("\nOs dois eixos se cruzam; combinações sem documento são recusadas.")
        raise SystemExit(0)
    if not args.termo_a or not args.termo_b:
        parser.error("informe os dois termos (ou use --listar-escopos)")

    try:
        res = comparar(args.termo_a, args.termo_b, args.escopo, args.proveniencia)
    except ValueError as exc:
        raise SystemExit(f"erro: {exc}")

    c = res["corpus"]
    id_escopo = f" ({c['escopo']})" if c["escopo"] else ""
    print("═" * 60)
    print(f"SIMPLICIDADE: «{res['a']['termo']}» × «{res['b']['termo']}»")
    filtro = " (filtro aplicado)" if c["proveniencia"] else ""
    print(f"Tema:        {c['rotulo']}{id_escopo}")
    print(f"Procedência{filtro}: "
          + "; ".join(p["rotulo"] for p in c["proveniencias"]))
    print(f"Documentos:  {c['documentos']['original']} originais × "
          f"{c['documentos']['simplificada']} simplificadas")
    if c["proveniencia_mista"]:
        print("  ATENÇÃO: este recorte mistura simplificação validada por "
              "humanos com simplificação\n  gerada por IA — o score sai de uma "
              "referência híbrida.")
    print("═" * 60)
    _imprimir_termo("A", res["a"])
    _imprimir_termo("B", res["b"])

    print("\n" + "─" * 60)
    if res["mais_simples"]:
        print(f"Mais simples: «{res['mais_simples']}» "
              f"(critério: {res['criterio']}, Δscore = {res['delta_score']:+.3f})")
    else:
        print(f"Empate — nada distingue os termos (Δscore = {res['delta_score']:+.3f}).")
