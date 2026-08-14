"""Caminhos do projeto, ancorados no diretório do código (não no cwd).

Todos os caminhos eram relativos ("./chroma_db", "./csv_metadata/..."), o que só
funciona quando o processo é iniciado de dentro da pasta do projeto. Em produção
quem inicia é o systemd — e a reindexação pode ser disparada de qualquer lugar —,
então ancoramos tudo em BASE_DIR. Módulo sem dependências e sem efeitos
colaterais: pode ser importado por qualquer script.

CHROMA_PATH pode ser sobrescrito por env (CHROMA_PATH) para apontar o índice para
outro disco/volume sem mexer no código.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Corpus de entrada (bulas em XML já com metadados) e mapas id->nome.
CORPUS_DIR = BASE_DIR / "csv_metadata"

# Índice vetorial persistente do ChromaDB.
CHROMA_PATH = os.getenv("CHROMA_PATH") or str(BASE_DIR / "chroma_db")
