import os
import pandas as pd
import meilisearch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- 1. Configurações Iniciais ---
# Usando o modelo multilingue recomendado para português
print("Carregando modelo de embedding...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Modelo carregado com sucesso!")
# Inicialização dos Clientes
print("Conectando ao Meilisearch e ChromaDB...")
load_dotenv()
MEILI_KEY = os.getenv('MEILI_MASTER_KEY')
MEILI_URL = os.getenv('MEILI_URL', 'http://localhost:7700')
if not MEILI_KEY:
    print("❌ Erro: MEILI_MASTER_KEY não encontrada no arquivo .env")
    exit(1)
meili_client = meilisearch.Client(MEILI_URL, MEILI_KEY)
meili_index = meili_client.index('corpop_saude_ht')
print("Conexões meili estabelecidas com sucesso!")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_coll = chroma_client.get_or_create_collection(name="corpop_saude_ht")
print("Conexões chroma estabelecidas com sucesso!")
# Caminhos das pastas (ajuste se forem caminhos absolutos)
PATH_ORIGINAL = './csv/ht/original'
PATH_SIMPLIFICADA = './csv/ht/simplificada'
PATH_CSV_MAP = './remedios_ht_map.csv'

# --- 2. Carregar Mapeamento ---
# Forçamos o ID como string para evitar problemas com zeros à esquerda
df_map = pd.read_csv(PATH_CSV_MAP, dtype={'id': str})
mapa_remedios = dict(zip(df_map['id'], df_map['nome']))
print(f"Mapeamento carregado com {len(mapa_remedios)} medicamentos.")

def realizar_indexacao():
    print(f"Iniciando indexação de {len(mapa_remedios)} medicamentos...")

    for n_id, nome_remedio in mapa_remedios.items():
        # Construção dos nomes de arquivo conforme seu padrão
        file_orig = f"{n_id}_original_limpo.txt"
        file_simp = f"{n_id}_validada_limpo.txt"
        
        path_o = os.path.join(PATH_ORIGINAL, file_orig)
        path_s = os.path.join(PATH_SIMPLIFICADA, file_simp)

        if os.path.exists(path_o) and os.path.exists(path_s):
            with open(path_o, 'r', encoding='utf-8') as f_o, \
                 open(path_s, 'r', encoding='utf-8') as f_s:
                
                texto_orig = f_o.read().strip()
                texto_simp = f_s.read().strip()

                # --- Lógica da Busca Híbrida ---
                # Criamos um texto combinado para o embedding. Isso faz o vetor 
                # "entender" a relação entre o termo técnico e o simples.
                texto_hibrido = f"Medicamento: {nome_remedio}\nOriginal: {texto_orig}\nSimplificado: {texto_simp}"
                embedding = model.encode(texto_hibrido).tolist()

                # --- A. Indexar no ChromaDB (Busca Semântica/Sentido) ---
                chroma_coll.add(
                    embeddings=[embedding],
                    documents=[texto_simp], # O que será retornado na busca semântica
                    metadatas=[{"id": n_id, "nome": nome_remedio, "tipo": "hipertensao"}],
                    ids=[n_id]
                )

                # --- B. Indexar no Meilisearch (Busca Léxica/Palavra-Chave) ---
                meili_index.add_documents([{
                    "id": n_id,
                    "nome": nome_remedio,
                    "conteudo_original": texto_orig,
                    "conteudo_simplificado": texto_simp
                }])

                print(f" [OK] ID {n_id}: {nome_remedio} indexado.")
        else:
            print(f" [ERRO] Arquivos para o ID {n_id} ({nome_remedio}) não encontrados.")

if __name__ == "__main__":
    realizar_indexacao()