"""Parsing e chunking das bulas — núcleo compartilhado, sem efeitos colaterais.

Reúne o que `embedding.py` (indexação de produção) e `avaliar.py` (experimento
2×2 de modelo × chunking) precisam dividir, para que a estratégia de chunking
tenha UMA fonte de verdade e não haja risco de o experimento testar uma cópia
defasada. Importar este módulo não carrega modelo nem conecta em nada.

Estratégias de chunking:
  - `chunk_simples`     — agrupa linhas até ~600 chars (a versão "velha").
  - `chunk_estrutural`  — fatia por seção/subseção/lista, item-a-item nas listas,
                          com cabeçalho de seção (a versão "nova").
`preparar_chunks` unifica as duas: devolve, por chunk, o texto a ARMAZENAR e o
texto a EMBEDDAR (no estrutural o cabeçalho entra só no embedding, não no doc).
"""
import csv
import re

from caminhos import BASE_DIR, CORPUS_DIR


# Campos de metadados (do bloco <metadata> das bulas XML) que guardamos junto
# de cada entrada. Conjunto curado e útil para busca/exibição.
CAMPOS_METADADOS = [
    "doenca", "nome_medicamento", "nome_comercial",
    "medgenerico", "medreferencia", "fabricante", "apresentacao",
]

# Cada tipo de bula vive numa pasta própria, mas compartilham a mesma coleção.
# O `tipo` entra como prefixo do id (evita colisão ht_1 vs onco_1) e como metadado.
# Caminhos absolutos (via BASE_DIR): a indexação também roda a partir do systemd,
# onde o cwd não é a pasta do projeto.
TIPOS = [
    {
        "tipo": "hipertensao",
        "map_csv": str(BASE_DIR / "remedios_ht_map.csv"),
        "path_original": str(CORPUS_DIR / "ht" / "original"),
        "path_simplificada": str(CORPUS_DIR / "ht" / "simplificada"),
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_validada_limpo.txt",
    },
    {
        "tipo": "oncologia",
        "map_csv": str(BASE_DIR / "remedios_onco_map.csv"),
        "path_original": str(CORPUS_DIR / "onco" / "original"),
        "path_simplificada": str(CORPUS_DIR / "onco" / "simplificada"),
        "suffix_orig": "_original_limpo.txt",
        "suffix_simp": "_simplificada_limpo.txt",
    },
]


def ler_mapa(caminho):
    """Lê um CSV `id,nome` de medicamentos e devolve {id: nome}.

    O id fica como string de propósito: ele entra na composição dos ids do Chroma
    (ex.: 'hipertensao_7') e vira nome de arquivo, então não pode virar número.
    """
    with open(caminho, newline="", encoding="utf-8") as f:
        return {linha["id"]: linha["nome"] for linha in csv.DictReader(f)}


def parse_bula(raw):
    """Separa uma bula em (metadados: dict, texto: str).

    O arquivo tem um bloco <metadata>…</metadata> e um bloco <text>…</text>.
    Não dá para usar um parser XML padrão porque alguns nomes de tag têm espaço
    (ex.: <nome medicamento>) — XML inválido —, então extraímos por regex
    tolerante. Só o conteúdo de <text> é indexado; os metadados viram campos à
    parte. Se as tags não existirem, cai para o texto inteiro.
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


# --- Estratégia VELHA: agrupamento por tamanho ---

CHUNK_ALVO_CHARS = 600


def chunk_simples(texto, alvo_chars=CHUNK_ALVO_CHARS):
    """Quebra `texto` em trechos de ~alvo_chars, preservando quebras de linha.

    Cada linha não vazia é uma unidade; linhas muito longas são subdivididas por
    sentença. Unidades consecutivas são agrupadas até o tamanho-alvo. Esta é a
    estratégia anterior, mantida para o experimento comparativo.
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


# --- Estratégia NOVA: estrutural por seção, item-a-item nas listas ---

# Tamanho-alvo (em caracteres) dos chunks de PROSA. Seções em lista (reações,
# contraindicações, interações) NÃO usam isso: cada item vira um chunk próprio.
PROSE_ALVO_CHARS = 450

# Cabeçalho de seção numerada padrão ANVISA: "1. PARA QUE...", "8. Quais os...".
_RE_SECAO = re.compile(r"^\d+\.\s+\S")
# Faixas de frequência das reações adversas (introduzem listas de sintomas).
_RE_FREQ = re.compile(
    r"^-?\s*(muito comuns|comuns|incomuns|raros?|muito raros?|"
    r"desconhecidos?|frequência desconhecida)\b", re.IGNORECASE)
# Marcadores de item de lista.
_RE_ITEM = re.compile(r"^[-–•]\s+")


def _eh_subsecao(linha):
    """True se `linha` é um subcabeçalho dentro de uma seção (não conteúdo).

    Três formas vistas nas bulas: faixa de frequência ("Comuns (1 a cada 10
    pessoas):"), pergunta-título curta ("O que é pressão alta?", com ou sem "-"
    na frente) e linha em CAIXA ALTA que não é frase ("INFORMAÇÕES AO PACIENTE
    COM PRESSÃO ALTA"). Avisos em maiúsculas que terminam em pontuação (ex.:
    "...FORA DO ALCANCE DAS CRIANÇAS.") NÃO contam — são conteúdo.
    """
    if _RE_FREQ.match(linha):
        return True
    nucleo = linha.lstrip("-–• ").strip()
    if not nucleo:
        return False
    if nucleo.endswith("?") and len(nucleo) < 120:
        return True
    letras = [c for c in nucleo if c.isalpha()]
    if len(letras) >= 8 and nucleo == nucleo.upper() and nucleo[-1] not in ".!;:":
        return True
    return False


def chunk_estrutural(texto):
    """Fatia uma bula respeitando a estrutura (seções/subseções/listas).

    Devolve uma lista de dicts {texto, header, secao, subsecao}:
      - `texto`: conteúdo cru do chunk — é o que será ARMAZENADO no Chroma.
      - `header`: seção + subseção. Usado só como PREFIXO na hora de embeddar
        (dá contexto ao vetor: "Reações adversas: dor de cabeça"), sem poluir o
        texto exibido nem a reconstrução do documento completo.
      - `secao` / `subsecao`: rótulos para metadata e filtragem futura.

    Itens de lista viram um chunk cada (vetor limpo por sintoma, em vez da média
    borrada da lista inteira); prosa contígua é agrupada até ~PROSE_ALVO_CHARS.
    Um chunk nunca atravessa fronteira de seção/subseção.
    """
    chunks = []
    secao = subsecao = ""
    buffer = []  # prosa acumulada na (sub)seção atual

    def _header():
        return " — ".join(p for p in (secao, subsecao) if p)

    def _flush():
        if buffer:
            chunks.append({"texto": "\n".join(buffer), "header": _header(),
                           "secao": secao, "subsecao": subsecao})
            buffer.clear()

    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha:
            continue

        if _RE_SECAO.match(linha):          # nova seção numerada
            _flush()
            secao, subsecao = linha, ""
            continue

        if _eh_subsecao(linha):             # subcabeçalho dentro da seção
            _flush()
            subsecao = linha.lstrip("-–• ").strip()
            continue

        if _RE_ITEM.match(linha):           # item de lista → chunk próprio
            _flush()
            item = _RE_ITEM.sub("", linha).strip()
            chunks.append({"texto": item, "header": _header(),
                           "secao": secao, "subsecao": subsecao})
            continue

        # Prosa: acumula até o alvo, quebrando linhas longas por sentença.
        sentencas = ([linha] if len(linha) <= PROSE_ALVO_CHARS
                     else [s.strip() for s in re.split(r"(?<=[.!?;])\s+", linha) if s.strip()])
        for s in sentencas:
            atual = sum(len(x) + 1 for x in buffer)
            if buffer and atual + len(s) > PROSE_ALVO_CHARS:
                _flush()
            buffer.append(s)

    _flush()
    return chunks or [{"texto": texto.strip(), "header": "", "secao": "", "subsecao": ""}]


def preparar_chunks(texto, estrutural):
    """Normaliza as duas estratégias para o mesmo formato de saída.

    Devolve uma lista de dicts com:
      - `texto`  : o que ARMAZENAR em `documents` (texto cru, sem cabeçalho).
      - `embed`  : o que EMBEDDAR. No estrutural é `header + texto` (contexto de
                   seção no vetor); no simples é o próprio texto.
      - `header`, `secao`, `subsecao`: metadados (vazios no chunking simples).

    Centraliza a regra "cabeçalho entra no embedding mas não no documento", para
    que indexação e avaliação fiquem necessariamente consistentes.
    """
    if estrutural:
        out = []
        for c in chunk_estrutural(texto):
            embed = f"{c['header']}\n{c['texto']}" if c["header"] else c["texto"]
            out.append({"texto": c["texto"], "embed": embed, "header": c["header"],
                        "secao": c["secao"], "subsecao": c["subsecao"]})
        return out
    return [{"texto": t, "embed": t, "header": "", "secao": "", "subsecao": ""}
            for t in chunk_simples(texto)]
