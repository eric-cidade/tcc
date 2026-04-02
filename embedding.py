import os
import re
import uuid
import meilisearch
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configurações Iniciais ---
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
meili_index = meilisearch.Client('http://localhost:7700', 'masterKey').index('corpop_saude')
chroma_coll = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("corpop_saude")

# Regex para identificar títulos de bulas (ex: "1. O QUE É...", "COMO USAR", etc.)
# Ajuste este padrão de acordo com o formato dos seus .txt
SECTION_PATTERN = r'(\n[0-9]+\.\s[A-ZÇÁÉÍÓÚ\s\?]+|\n[A-ZÇÁÉÍÓÚ\s]{5,}:)'

def processar_bulas(diretorio_txt):
    for filename in os.listdir(diretorio_txt):
        if not filename.endswith(".txt"): continue
        
        nome_remedio = filename.replace(".txt", "").replace("_", " ").title()
        caminho_completo = os.path.join(diretorio_txt, filename)
        
        with open(caminho_completo, 'r', encoding='utf-8') as f:
            conteudo_total = f.read()

        # 1. QUEBRA AUTOMÁTICA EM SEÇÕES (CHUNKING)
        # O split vai gerar uma lista alternando entre títulos e conteúdos
        partes = re.split(SECTION_PATTERN, conteudo_total)
        
        # Reconstruir os chunks unindo título + conteúdo
        chunks = []
        for i in range(1, len(partes), 2):
            titulo_secao = partes[i].strip()
            texto_secao = partes[i+1].strip() if i+1 < len(partes) else ""
            chunks.append(f"{titulo_secao}\n{texto_secao}")

        # 2. INDEXAÇÃO EM LOTE (BATCH)
        for idx, texto_chunk in enumerate(chunks):
            chunk_id = f"{filename}_{idx}"
            
            # --- Enviar para o Meilisearch (Palavra-Chave) ---
            meili_index.add_documents([{
                "id": chunk_id.replace(".", "_"), # IDs não podem ter pontos
                "remedio": nome_remedio,
                "secao": texto_chunk[:100], # Preview do título
                "conteudo": texto_chunk
            }])
            
            # --- Enviar para o ChromaDB (Vetorial) ---
            embedding = model.encode(texto_chunk).tolist()
            chroma_coll.add(
                embeddings=[embedding],
                documents=[texto_chunk],
                metadatas=[{"remedio": nome_remedio, "fonte": filename}],
                ids=[chunk_id]
            )

    print(f"Processamento concluído para o diretório: {diretorio_txt}")

# Chame a função passando sua pasta
# processar_bulas('./meus_textos_validados')