"""Extrai UMA coleção do chroma_db para um diretório novo e enxuto.

Serve para levar o índice pronto ao servidor em vez de reconstruí-lo lá — a
indexação com bge-m3 leva de 30 a 90 minutos, a cópia leva segundos.

O chroma_db de desenvolvimento acumula coleções de experimentos (comparações de
modelo, recortes de corpus), e todas dividem o mesmo `chroma.sqlite3`. Copiar o
diretório inteiro levaria centenas de MB de coisas que produção não usa; este
script grava só a coleção pedida, num banco limpo.

Não carrega modelo: apenas lê os vetores já gravados e os regrava.

Rodar:
  uv run python deploy/exportar_indice.py                      # coleção padrão
  uv run python deploy/exportar_indice.py --colecao NOME
  uv run python deploy/exportar_indice.py --saida /tmp/indice
"""
import argparse
import os
import sys

import chromadb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caminhos import CHROMA_PATH                      # noqa: E402
from modelos import CHROMA_COLLECTION_PADRAO          # noqa: E402

LOTE = 500


def tamanho(caminho):
    total = sum(os.path.getsize(os.path.join(raiz, f))
                for raiz, _, arqs in os.walk(caminho) for f in arqs)
    return f"{total / 1e6:.0f} MB"


def main():
    ap = argparse.ArgumentParser(description="Exporta uma coleção do Chroma para um diretório limpo")
    ap.add_argument("--colecao", default=CHROMA_COLLECTION_PADRAO,
                    help=f"coleção a exportar (default: {CHROMA_COLLECTION_PADRAO})")
    ap.add_argument("--origem", default=CHROMA_PATH, help="chroma_db de origem")
    ap.add_argument("--saida", default="chroma_db_export", help="diretório a criar")
    args = ap.parse_args()

    if os.path.exists(args.saida) and os.listdir(args.saida):
        raise SystemExit(f"❌ {args.saida} já existe e não está vazio. Apague antes.")

    origem = chromadb.PersistentClient(path=args.origem)
    try:
        fonte = origem.get_collection(args.colecao)
    except Exception:
        nomes = [c.name for c in origem.list_collections()]
        raise SystemExit(f"❌ Coleção {args.colecao!r} não existe.\n   Existem: {', '.join(nomes)}")

    total = fonte.count()
    if total == 0:
        raise SystemExit(f"❌ A coleção {args.colecao!r} está vazia — nada a exportar.")

    destino = chromadb.PersistentClient(path=args.saida)
    # Mesmo metadata da origem: o espaço de distância precisa ser idêntico, senão
    # a busca no servidor ordenaria por outra métrica.
    alvo = destino.create_collection(name=args.colecao,
                                     metadata=fonte.metadata or {"hnsw:space": "cosine"})

    print(f"Exportando {args.colecao!r}: {total} chunks")
    copiados = 0
    while copiados < total:
        pedaco = fonte.get(limit=LOTE, offset=copiados,
                           include=["embeddings", "documents", "metadatas"])
        if not pedaco["ids"]:
            break
        alvo.add(ids=pedaco["ids"], embeddings=pedaco["embeddings"],
                 documents=pedaco["documents"], metadatas=pedaco["metadatas"])
        copiados += len(pedaco["ids"])
        print(f"  {copiados}/{total}", end="\r", flush=True)

    print(f"\n✔ {alvo.count()} chunks em {args.saida}/ ({tamanho(args.saida)})")
    print(f"  origem: {tamanho(args.origem)} (com todas as coleções de experimento)")
    print("\nPara levar ao servidor:")
    print(f"  tar czf indice.tar.gz -C {args.saida} .")
    print("  scp indice.tar.gz usuario@143.54.25.141:/tmp/")
    print("  # no servidor:")
    print("  sudo -u corpop mkdir -p /opt/corpop-saude/chroma_db")
    print("  sudo -u corpop tar xzf /tmp/indice.tar.gz -C /opt/corpop-saude/chroma_db")


if __name__ == "__main__":
    main()
