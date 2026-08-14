# Deploy — Busca Híbrida CorPop-Saúde

Subida do projeto no servidor **143.54.25.141** (Ubuntu 24.04, 4 vCPU, 4GB RAM,
40GB disco).

A restrição que define tudo aqui é **RAM**, não disco. Por isso: um worker de
uvicorn apenas, sem reranker, e o modelo de embedding escolhido pelo custo.

## O modelo

O escolhido é o **`BAAI/bge-m3`**, o mesmo do TCC. Quatro candidatos mais leves
foram avaliados no corpus completo (79 bulas, 12.520 chunks):

| Modelo | RAM (processo) | Indexação | Query | Sintoma¹ | Fármaco² | PT-BR³ |
|---|---|---|---|---|---|---|
| **bge-m3** | **2.076 MB** | 30,5min | 145ms | **1,000** | 0,912 | 32º/93 |
| embeddinggemma-300m | 1.193 MB | 14,7min | 79ms | 0,867 | **0,938** | **13º/93** |
| e5-small | 899 MB | 3,1min | 17ms | 0,800 | 0,935 | ~40º/93 |

¹ Context Relevance@5 nas consultas por sintoma — **a tarefa que é o propósito do
projeto**. ² Context Precision (MAP) nas consultas por princípio ativo.
³ posição no [MTEB-BR](https://arxiv.org/abs/2607.04581), 22 tarefas nativas
em PT-BR. Tempos na máquina de desenvolvimento; no servidor, ~2-3x mais.

O bge-m3 vence onde mais importa — busca por sintoma em linguagem leiga, que é o
público-alvo. Perde em busca por princípio ativo, mas ali a lane léxica já
resolve sozinha (é tarefa quase lexical). Some-se a **licença MIT**, sem
obrigações sobre o site publicado nem token a administrar.

Os 2GB de RAM deixam ~1,3GB livres no servidor, e **não impedem reindexar com o
site no ar** — ver a seção sobre adicionar bulas.

Descartados: `multilingual-e5-small` (fraco em português e scores anisotrópicos
demais para qualquer limiar); `gte-multilingual-base` e
`granite-embedding-311m-r2` (arquiteturas sem caminho otimizado em CPU, mais
lentas que modelos duas vezes maiores).

**Alternativa:** o `embeddinggemma-300m` é metade da RAM e 2x mais rápido, e vale
considerar se a latência incomodar. Custa aceitar os
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) — servir a busca na web
conta como *Distribution* via *Hosted Service*, exigindo repassar as restrições
de uso nos termos do site — e gerenciar um `HF_TOKEN`.

**Sobre o limiar de score:** cada modelo espalha os cossenos numa faixa própria,
então `min_score` acompanha o modelo (ver `MODELOS` em `modelos.py`; a API informa
o valor efetivo em `GET /config`). Com o bge-m3 em 0,48, as 6 consultas fora de
domínio testadas são rejeitadas e as 10 legítimas passam. O valor anterior (0,40)
ficava **abaixo** do teto de ruído medido (0,440) e não filtrava nada.

## Arquitetura

```
        :80                    127.0.0.1:8000          127.0.0.1:7700
navegador ──► nginx ──┬──► uvicorn (api:app) ──┬──► Meilisearch  (léxica)
                      │      1 worker          └──► ChromaDB     (semântica,
                      └──► frontend/index.html                    em disco)
```

Nginx serve o frontend e faz proxy da API na **mesma origem** — é o que dispensa
CORS. Meilisearch e uvicorn ficam em loopback, inacessíveis de fora.

## Subida do zero

### 1. Clonar e provisionar

```bash
sudo git clone -b deploy https://github.com/eric-cidade/tcc.git /opt/corpop-saude
sudo bash /opt/corpop-saude/deploy/setup.sh
```

O `setup.sh` é idempotente (pode rodar de novo) e faz: pacotes, usuário `corpop`,
**+3GB de swap** (o 1GB original não dá margem), Meilisearch + systemd,
`uv sync --no-dev`, units, nginx e ufw. Ele **não** indexa nem sobe a API.

### 2. Conferir o `.env`

O setup gera `/opt/corpop-saude/.env` com uma master key nova. Revise:

```bash
sudo -u corpop nano /opt/corpop-saude/.env
```

`EMBED_MODEL` e `CHROMA_COLLECTION` **precisam casar** — cada modelo tem dimensão
própria e por isso sua própria coleção. Os modelos suportados estão em
`modelos.py` (`MODELOS`), com o consumo de RAM de cada um.

### 3. Levar o índice pronto

Gerar os embeddings do corpus inteiro no servidor levaria de 30 a 90 minutos de
CPU. Como os vetores já existem na máquina de desenvolvimento, é mais rápido
copiá-los — e não se perde nada, porque bulas novas entram pelo modo incremental,
sem reconstruir o resto.

**Na sua máquina** — extrai só a coleção de produção (o `chroma_db` de
desenvolvimento tem várias coleções de experimento; 456MB viram 91MB, 59MB
compactados):

```bash
uv run python deploy/exportar_indice.py
tar czf indice.tar.gz -C chroma_db_export .
scp indice.tar.gz usuario@143.54.25.141:/tmp/
```

**No servidor** — extrai o índice vetorial e popula a busca léxica:

```bash
sudo -u corpop mkdir -p /opt/corpop-saude/chroma_db
sudo -u corpop tar xzf /tmp/indice.tar.gz -C /opt/corpop-saude/chroma_db

sudo -u corpop bash -c 'cd /opt/corpop-saude && HOME=/opt/corpop-saude \
    uv run --no-dev python embedding.py --apenas-meili'
```

> O `cd` não é decoração: o ChromaDB procura um `.env` no **diretório atual**, e
> se o comando for disparado de um diretório que o usuário `corpop` não pode ler
> — a sua home, por exemplo — o import falha com `PermissionError: '.env'`.

O `--apenas-meili` roda em **~1 segundo** e **não carrega modelo** — a busca
léxica é só texto. O ChromaDB não é tocado.

> **Alternativa: construir tudo no servidor.** Se preferir não copiar nada, rode
> `embedding.py` sem argumentos. Ele constrói as duas buscas do zero, carregando
> o modelo localmente (a API ainda não existe para servir o `/embed`). Reserve
> de 30 a 90 minutos.

### 4. Subir a API

```bash
sudo systemctl enable --now corpop-saude-api
```

O primeiro boot baixa os pesos do modelo do HuggingFace antes de responder — por
isso o `TimeoutStartSec=900` na unit. Acompanhe com
`journalctl -u corpop-saude-api -f`.

## Verificação

```bash
curl http://143.54.25.141/health                                  # {"status":"ok"}
curl "http://143.54.25.141/buscar?q=press%C3%A3o%20alta&n=3"
curl "http://143.54.25.141/simplicidade?a=cefaleia&b=dor%20de%20cabe%C3%A7a"
```

E no navegador: `http://143.54.25.141/` deve carregar o frontend e buscar sem
erro de CORS no console.

**O teste que mais importa** é a memória, não o HTTP:

```bash
systemctl status corpop-saude-api | grep Memory   # RSS da API
free -h                                  # RAM livre com tudo no ar
dmesg | grep -i oom                      # tem que estar vazio
```

Se o `Memory` da API estiver perto do `MemoryHigh=2900M` da unit, o modelo é
grande demais para este servidor — troque `EMBED_MODEL` por um mais leve (e
reindexe na coleção correspondente).

Confirme também que as dependências de desenvolvimento ficaram de fora:

```bash
/opt/corpop-saude/.venv/bin/python -c "import matplotlib"   # tem que FALHAR
du -sh /opt/corpop-saude/.venv                              # sem a stack CUDA
```

## Adicionar bulas depois

1. Coloque os arquivos em `csv_metadata/<tipo>/{original,simplificada}/`
2. Acrescente as linhas `id,nome` no `remedios_*_map.csv` correspondente
3. `sudo bash /opt/corpop-saude/deploy/reindexar.sh`

**O site continua no ar durante a indexação.** Duas escolhas de projeto tornam
isso possível num servidor de 4GB:

- **Incremental:** para cada medicamento, os chunks antigos dele são removidos e
  os novos gravados. A coleção nunca é apagada — a API não perde a referência a
  ela. (Rebuild completo só é necessário ao trocar de modelo ou de chunking.)
- **Indexador magro:** os embeddings são pedidos ao endpoint `/embed` da API, que
  já tem o modelo carregado. Sem isso seriam duas cópias do modelo na memória
  (~4,2GB com bge-m3), o que não caberia. O indexador sozinho usa ~160MB.

Se a API estiver parada, o script carrega o modelo localmente — é o que acontece
na primeira indexação, antes de a API existir.

### Quando reconstruir do zero

`sudo bash deploy/reindexar.sh --reconstruir` apaga e refaz a coleção. Só é
necessário ao **trocar de modelo** (dimensão diferente) ou **mudar o chunking**.
Nesse modo o script para a API antes, porque apagar a coleção invalida o handle
que ela mantém: as consultas passariam a falhar com `NotFoundError` até o
reinício, mesmo depois do índice novo ficar pronto.

## Atualizar o código

```bash
cd /opt/corpop-saude && sudo -u corpop git pull
sudo -u corpop env HOME=/opt/corpop-saude uv sync --no-dev --project /opt/corpop-saude
sudo systemctl restart corpop-saude-api
```

Se a atualização mexeu em chunking, modelo ou metadados, rode também o
`reindexar.sh` — o índice em disco não se atualiza sozinho.

## Operação

| Ação | Comando |
|---|---|
| Logs da API | `journalctl -u corpop-saude-api -f` |
| Logs do Meilisearch | `journalctl -u meilisearch -f` |
| Logs do nginx | `tail -f /var/log/nginx/corpop-saude.{access,error}.log` |
| Reiniciar tudo | `sudo systemctl restart meilisearch corpop-saude-api nginx` |
| Estado | `systemctl status meilisearch corpop-saude-api nginx` |

## Problemas comuns

**A API sobe mas toda busca dá 503.** O Meilisearch está fora do ar ou a master
key diverge. As duas units leem o mesmo `/opt/corpop-saude/.env`; confira que o arquivo
está legível pelo grupo `meilisearch` (`ls -l /opt/corpop-saude/.env` → `corpop
meilisearch`).

**A API não inicia e o log fala em coleção vazia.** Falta o passo 3 (indexar), ou
`CHROMA_COLLECTION` no `.env` não corresponde ao que foi indexado.

**O serviço morre sozinho / o servidor trava.** Falta de RAM. Confira
`dmesg | grep -i oom`. Reduza `EMBED_BATCH` no `.env` (8 → 4) ou troque para um
modelo menor.

**A busca responde 504 no navegador.** Consulta passou dos 120s do
`proxy_read_timeout`. Costuma ser sintoma de swap: veja `vm.swappiness` e a RAM
livre.

## Sem HTTPS

A subida é em HTTP. Habilitar HTTPS depende de decisões que ainda não foram
tomadas (qual nome usar e se o CPD da UFRGS tem processo próprio de emissão de
certificados), então fica fora do escopo desta configuração.

Quando for o caso, o frontend não precisará de alteração: ele usa caminho
relativo (`API_BASE` vazio) e acompanha o esquema da página automaticamente.
