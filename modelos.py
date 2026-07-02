"""Carregamento dos modelos de embedding — com os ajustes que cada um exige.

Fonte única de `SentenceTransformer(...)` para indexação, busca e avaliação, para
que o conserto abaixo não precise ser repetido (nem esquecido) em cada script.

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


def carregar_modelo(nome, max_seq_length=8192):
    """Carrega um SentenceTransformer pronto para uso (gte ou bge).

    trust_remote_code é exigido pela arquitetura custom do gte e inofensivo para
    modelos sem código remoto (ex.: bge-m3).
    """
    model = SentenceTransformer(nome, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    _consertar_buffers(model)
    return model


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
