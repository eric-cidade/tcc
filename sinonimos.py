"""Dicionário de sinônimos PT do domínio das bulas (termo técnico <-> leigo).

Fonte ÚNICA usada de duas formas:
  - embedding.py: configura os sinônimos do Meilisearch (busca léxica passa a
    casar "cefaleia" com bulas que dizem "dor de cabeça").
  - search.py: expande a query antes do embedding (melhora o recall semântico
    do ChromaDB, que é fraco em sinônimo curto).

Cada grupo é um conjunto de termos equivalentes. Edite/expanda à vontade — é a
forma mais confiável de tratar sinônimos CONHECIDOS (o que a busca por
similaridade sozinha não garante).
"""

GRUPOS_SINONIMOS = [
    ["cefaleia", "dor de cabeça"],
    ["hipertensão", "hipertensão arterial", "pressão alta"],
    ["hipotensão", "pressão baixa"],
    ["náusea", "enjoo"],
    ["êmese", "vômito"],
    ["pirose", "azia"],
    ["astenia", "fraqueza", "cansaço"],
    ["mialgia", "dor muscular"],
    ["artralgia", "dor nas articulações"],
    ["dispneia", "falta de ar"],
    ["prurido", "coceira"],
    ["edema", "inchaço"],
    ["vertigem", "tontura"],
    ["epistaxe", "sangramento nasal"],
    ["dispepsia", "má digestão"],
]


def sinonimos_meili():
    """Mapa {termo: [outros termos do grupo]} no formato do Meilisearch."""
    syn = {}
    for grupo in GRUPOS_SINONIMOS:
        for termo in grupo:
            syn[termo] = [t for t in grupo if t != termo]
    return syn


def sinonimos_da_query(query):
    """Sinônimos conhecidos que seriam acrescentados à query (sem duplicatas)."""
    q = query.lower()
    extras = []
    for grupo in GRUPOS_SINONIMOS:
        if any(termo in q for termo in grupo):
            extras += [termo for termo in grupo if termo not in q]
    # dict.fromkeys preserva ordem e remove duplicatas
    return list(dict.fromkeys(extras))


def expandir_query(query):
    """Acrescenta à query os sinônimos dos termos conhecidos que ela contém.

    Ex.: 'cefaleia' -> 'cefaleia dor de cabeça'. Mantém a query original quando
    nada casa. Usado só na busca semântica (a léxica usa os sinônimos do Meili).
    """
    extras = sinonimos_da_query(query)
    if not extras:
        return query
    return query + " " + " ".join(extras)
