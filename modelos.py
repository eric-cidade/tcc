"""Carregamento dos modelos de embedding — com os ajustes que cada um exige.

Fonte única de `SentenceTransformer(...)` para indexação, busca e avaliação, para
que os consertos abaixo não precisem ser repetidos (nem esquecidos) em cada script.

Duas coisas variam de modelo para modelo e ANULAM a qualidade da busca se ficarem
no default — as duas ficam declaradas em MODELOS:

1. PREFIXOS de instrução. Os `intfloat/multilingual-e5-*` foram treinados com
   "query: " nas consultas e "passage: " nos documentos; sem eles o retrieval
   degrada a ponto de o modelo parecer simplesmente ruim. Já bge-m3 e gte NÃO
   usam prefixo. Por isso ninguém deve chamar `model.encode()` cru: use
   `encode_query()` / `encode_docs()` daqui, que aplicam o prefixo do lado certo.
2. CONTEXTO máximo. bge-m3 e gte aceitam 8192 tokens; os e5 são XLM-R e param em
   512 — pedir 8192 neles é inválido.

Bug do gte-multilingual-base (arquitetura custom 'new-impl'): no carregamento
low-mem (meta device), os buffers NÃO-persistentes — que o __init__ deveria
preencher — ficam com LIXO de memória não inicializada:
  - `embeddings.position_ids`            -> deveria ser arange(0..N); vem lixo.
  - `rotary_emb.inv_freq`                -> vem `inf`.
  - `rotary_emb.cos_cached` / `sin_cached` -> vêm zerados.
O `position_ids` corrompido estoura com IndexError no primeiro encode; já o RoPE
corrompido é PIOR: não dá erro, mas as rotações posicionais viram ruído e o vetor
deixa de codificar o texto (busca devolve resultados quase aleatórios). Aqui
recomputamos esses buffers pelos próprios parâmetros do módulo. É inofensivo e
idempotente para o bge-m3 (que não tem esses buffers 1-D nem o rotary_emb).
"""
import torch
from sentence_transformers import SentenceTransformer

# Registro dos modelos suportados. Além de contexto e prefixos, guarda o limiar
# de corte, porque ele é PROPRIEDADE DO MODELO e não uma constante do sistema:
# cada modelo espalha os cossenos numa faixa própria.
#   max_seq  — teto real da arquitetura.
#   query/document — prefixos de instrução. Ausente = "", e string vazia tem um
#     significado ÚTIL aqui: o sentence-transformers só aplica o prompt que vem
#     no config do próprio modelo quando o nosso está vazio (SentenceTransformer
#     .py, `_load_sbert_model`: `if ... or not self.prompts[prompt_name]`). Ou
#     seja, deixar vazio = "use o que o modelo traz" (é o caso do
#     embeddinggemma, que define os seus); preencher = "use o meu" (caso dos e5,
#     que não trazem prompt no config e quebram sem prefixo).
#   min_score — corte default do cosseno na busca.
#
# Os limiares foram MEDIDOS no corpus completo (79 bulas), comparando o score do
# top-1 em 10 consultas legítimas do domínio contra 6 fora de domínio
# ("receita de bolo de cenoura", "como declarar imposto de renda", ...):
#
#   modelo     legítimas      fora de domínio   folga no limiar
#   bge-m3     0.519 - 0.712  0.317 - 0.440     0.079  (0.48: 10/10 e 6/6)
#   gemma-300m 0.300 - 0.613  0.176 - 0.319     0.061  (0.35:  9/10 e 6/6)
#   e5-base    0.839 - 0.899  0.809 - 0.816     0.023  frágil
#   e5-small   0.859 - 0.925  0.830 - 0.852     0.007  frágil
#
# "9/10 e 6/6" = mantém 9 das 10 consultas legítimas e rejeita os 6 ruídos. No
# gemma a única legítima sacrificada é "enjoo" (0.300), que fica abaixo de
# "receita de bolo de cenoura" (0.319) — é UM par, não uma sobreposição geral,
# então a margem crua (-0.019) engana; a folga real em torno de 0.35 é 0.061.
#
# Throughput medido na CPU (chunks/s, lote de 16, texto curto de bula), que no
# servidor sem GPU pesa tanto quanto a qualidade — indexação E latência de busca:
#   e5-small 243  |  embeddinggemma 39.4  |  bge-m3 19.2  |  granite-311m-r2 6.9
# gte-multilingual-base e granite-embedding-311m-r2 foram DESCARTADOS por aqui:
# arquiteturas sem caminho otimizado em CPU, ambas mais lentas que modelos duas
# vezes maiores (o gte levou 14m para indexar 10 bulas; o granite projetou 101m
# para o corpus completo, contra 5m do embeddinggemma).
#
# Consequência prática: no bge o limiar É um filtro de domínio (0.48 fica no meio
# da margem; o antigo 0.40 ficava ABAIXO do teto de ruído e deixava passar
# consulta fora de domínio). Nos e5 NENHUM limiar separa domínio; ali o valor é
# só um piso frouxo, e quem rejeita consulta fora de domínio é a lane léxica do
# Meilisearch (matchingStrategy="all").
#
# A compressão dos e5 é ANISOTROPIA DO MODELO, não efeito do nosso chunking.
# Duas medições descartam as explicações alternativas:
#   - truncamento: só 5 dos 12.520 chunks passam de 512 tokens (0,04%); a
#     mediana é 50 tokens, porque o chunking estrutural gera um chunk por item.
#   - corpus: frases curtas SEM RELAÇÃO entre si ("o gato dormiu no sofá" ×
#     "a bolsa de valores caiu") já dão cosseno 0.862 no e5-small contra 0.350
#     no bge. Nada a ver com bulas, sem truncamento possível.
# Centralizar os vetores (subtrair a média do corpus e renormalizar) foi testado:
# descomprime a escala para 0.22-0.52, como a teoria prevê, mas a margem fica
# NEGATIVA (-0.059) — descomprimir não cria o sinal de domínio que não existe.
# Não vale reimplementar.
MODELOS = {
    "BAAI/bge-m3":                       {"max_seq": 8192, "min_score": 0.48},
    "Alibaba-NLP/gte-multilingual-base": {"max_seq": 8192, "min_score": 0.55},
    "intfloat/multilingual-e5-base":     {"max_seq": 512, "min_score": 0.80,
                                          "query": "query: ", "document": "passage: "},
    "intfloat/multilingual-e5-small":    {"max_seq": 512, "min_score": 0.80,
                                          "query": "query: ", "document": "passage: "},
    # Sem prefixo (usa texto cru) e sem prompt no config. Contexto real de 32k,
    # capado aqui em 8192: nossos chunks têm mediana de 50 tokens e p95 de 138,
    # então o teto só serve para não truncar o outlier ocasional.
    "ibm-granite/granite-embedding-311m-multilingual-r2": {"max_seq": 8192},
    # Prompts VAZIOS de propósito: o embeddinggemma define os seus no config
    # ("task: search result | query: " / "title: none | text: ") e o
    # sentence-transformers só os aplica se os nossos estiverem vazios (ver nota
    # acima). Repo gated: exige aceitar a licença Gemma e HF_TOKEN no ambiente.
    "google/embeddinggemma-300m":        {"max_seq": 2048, "min_score": 0.35},
}

# Teto conservador para modelos fora do registro: 512 é o limite da maioria dos
# encoders BERT/XLM-R. Melhor truncar do que estourar as posições do modelo.
MAX_SEQ_PADRAO = 512
# Limiar conservador para modelo desconhecido: baixo o bastante para não esconder
# resultados de um modelo cuja escala não conhecemos.
MIN_SCORE_PADRAO = 0.40


def min_score_padrao(nome):
    """Limiar de cosseno adequado à escala de scores do modelo."""
    return MODELOS.get(nome, {}).get("min_score", MIN_SCORE_PADRAO)


# Par (modelo, coleção) usado quando o .env não diz outra coisa. Fica aqui, e não
# em search.py, porque a INDEXAÇÃO (embedding.py) precisa do mesmo par e não tem
# por que depender do módulo de busca. Os dois PRECISAM casar: cada modelo tem
# dimensão própria, então cada um tem a sua coleção.
#
# bge-m3 é o default. Escolhido sobre os candidatos mais leves por três motivos,
# nesta ordem:
#
# 1. QUALIDADE NA TAREFA QUE É O PROPÓSITO DO PROJETO. É o melhor em consultas
#    por sintoma em linguagem leiga (Context Relevance@5 = 1.000 com sinônimos e
#    0.750 sem; o embeddinggemma faz 0.867 / 0.706). Perde em busca por princípio
#    ativo (Context Precision 0.912 contra 0.938), mas ali a lane léxica já é
#    excelente sozinha — é tarefa quase lexical.
# 2. LICENÇA MIT. O embeddinggemma exige aceitar os Gemma Terms of Use, cujas
#    obrigações recaem sobre o site publicado (servir a busca na web é
#    "Distribution" via Hosted Service), além de um HF_TOKEN a administrar. Num
#    site institucional mantido por terceiros, isso é custo recorrente.
# 3. Continuidade com o que o TCC mediu, e a coleção já está indexada.
#
# O custo é latência: ~145ms por encode contra ~79ms do embeddinggemma. O que NÃO
# é mais custo é a indexação: com o indexador magro (embedding.py pede os vetores
# ao /embed da API) não há segunda cópia do modelo na memória, e o site fica no ar
# durante a reindexação mesmo com 4GB de RAM.
#
# O e5-small é o mais leve e rápido, mas foi descartado: é FRACO EM PORTUGUÊS
# (~40º de 93 no MTEB-BR) e seus scores são anisotrópicos demais para qualquer
# limiar (folga 0.007). A avaliação por princípio ativo não revela isso — aquela
# tarefa satura, todos os modelos acertam.
#
# Para trocar, ponha o par no .env (os dois PRECISAM casar):
#   EMBED_MODEL=google/embeddinggemma-300m
#   CHROMA_COLLECTION=corpop_saude_gemma_estr
EMBED_MODEL_PADRAO = "BAAI/bge-m3"
CHROMA_COLLECTION_PADRAO = "corpop_saude_bge_estr"


def carregar_modelo(nome, max_seq_length=None):
    """Carrega um SentenceTransformer pronto para uso, conforme o registro MODELOS.

    trust_remote_code é exigido pela arquitetura custom do gte e inofensivo para
    modelos sem código remoto (ex.: bge-m3). `max_seq_length` sobrescreve o valor
    do registro quando passado explicitamente.
    """
    cfg = MODELOS.get(nome, {})
    # Chaves fixas "query"/"document": são as que o encode_query()/encode_document()
    # do sentence-transformers procura por convenção.
    prompts = {"query": cfg.get("query", ""), "document": cfg.get("document", "")}
    model = SentenceTransformer(nome, trust_remote_code=True, prompts=prompts)
    model.max_seq_length = max_seq_length or cfg.get("max_seq", MAX_SEQ_PADRAO)
    _consertar_buffers(model)
    return model


def encode_query(model, texto, **kwargs):
    """Embeda o lado CONSULTA (aplica o prefixo de query do modelo, se houver)."""
    return model.encode_query(texto, **kwargs)


def encode_docs(model, textos, batch_size=None, **kwargs):
    """Embeda o lado DOCUMENTO (aplica o prefixo de passage do modelo, se houver).

    `batch_size=None` mantém o default do sentence-transformers; baixar esse valor
    é o principal controle de pico de RAM na indexação (ver EMBED_BATCH).
    """
    if batch_size:
        kwargs["batch_size"] = batch_size
    return model.encode_document(textos, **kwargs)


def _consertar_buffers(model):
    """Recomputa buffers não-persistentes corrompidos pelo carregamento low-mem."""
    # RoPE: reconstrói inv_freq (se não-finito) e a cache cos/sin. Chamamos o
    # próprio _set_cos_sin_cache do módulo, então subclasses (ex.: NTK) também
    # ficam corretas. Modelos sem rotary_emb (bge-m3) simplesmente não entram.
    for mod in model.modules():
        if (hasattr(mod, "inv_freq") and hasattr(mod, "_set_cos_sin_cache")
                and hasattr(mod, "dim") and hasattr(mod, "base")):
            if not torch.isfinite(mod.inv_freq).all():
                inv = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2).float() / mod.dim))
                mod.register_buffer("inv_freq", inv.to(mod.inv_freq.device), persistent=False)
            seq = getattr(mod, "max_seq_len_cached", None) or \
                getattr(mod, "max_position_embeddings", 8192)
            mod._set_cos_sin_cache(seq, mod.inv_freq.device, torch.get_default_dtype())

    # Buffer `position_ids` 1-D: deve ser um arange. Reinicializa se vier lixo.
    for mod in model.modules():
        for buf_nome, buf in list(mod.named_buffers(recurse=False)):
            if buf_nome == "position_ids" and buf is not None and buf.dim() == 1:
                esperado = torch.arange(buf.size(0), dtype=buf.dtype, device=buf.device)
                if not torch.equal(buf, esperado):
                    mod.register_buffer("position_ids", esperado, persistent=False)
