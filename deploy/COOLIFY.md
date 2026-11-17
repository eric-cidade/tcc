# Deploy no Coolify — Busca Híbrida CorPop-Saúde (e5-small)

Alternativa ao deploy em `deploy/README.md`, que só é acessível de dentro da rede
da UFRGS. Aqui o projeto roda em containers no Coolify, com domínio público e
HTTPS automático. Só sobe o modelo **`intfloat/multilingual-e5-small`**, com a
coleção **`corpop_saude_e5_small_estr`** (segmentação estrutural).

## Arquitetura

```
            HTTPS
navegador ──────► Traefik (Coolify) ──► api:8000 ──┬──► meilisearch:7700 (léxica)
                                         │         └──► ChromaDB em /data/chroma
                                         └── serve também o frontend/index.html
```

| Arquivo | Papel |
|---|---|
| `docker-compose.yml` | serviços `api` e `meilisearch`, volumes e limites de memória |
| `Dockerfile` | imagem da API com o modelo já embutido |
| `deploy/entrypoint.sh` | indexa no 1º boot e depois sobe o uvicorn |
| `.dockerignore` | deixa de fora índices locais, monografia e avaliações |

O índice **não** vai pelo git. No primeiro boot o entrypoint constrói o índice a
partir de `csv_metadata/` e o grava no volume `chroma`. Nos boots seguintes ele já
está lá, e só o Meilisearch é ressincronizado (cerca de 1 s).

## Recursos

As estimativas abaixo devem ser substituídas pelos números medidos após o primeiro deploy.

| | Regime | Pico | Limite no compose |
|---|---|---|---|
| RAM da API (e5-small carregado) | ~0,9 GB | ~1,2–1,5 GB (indexação do 1º boot) | 1,5 GB |
| RAM do Meilisearch | 0,1–0,25 GB | ~0,5 GB | 512 MB |
| Disco: imagem da API | ~2–2,5 GB | | |
| Disco: volumes `chroma` + `meili` | ~0,1–0,3 GB | | |
| Disco: build e imagens antigas | +1–2 GB temporários | | |

Na EC2 do Coolify (medida em 16/11/2026) havia 103 GB de disco livres e 7 GiB de
RAM disponível, com 4 vCPU. Cabe com folga.

## Passo a passo

1. **Antes de tudo:** confirme com a empresa que pode hospedar o projeto e tenha um
   subdomínio apontando para o servidor do Coolify. Se não houver, use o domínio
   `sslip.io` que o próprio Coolify sugere. Faça o primeiro deploy fora do horário
   de pico, porque o build consome CPU por alguns minutos.
2. **Push** da branch `coolify` com estes arquivos para o GitHub.
3. **Criar o recurso:** Projects → (projeto) → *+ New* → *Resource*
   - repo público: *Public Repository* → `https://github.com/eric-cidade/tcc`
   - repo privado: *Private Repository (with GitHub App)*
   - Branch: `coolify` · Build Pack: **Docker Compose** · Docker Compose Location: `/docker-compose.yml`
4. **Variável de ambiente:** na aba *Environment Variables*, crie `MEILI_MASTER_KEY`
   com um valor aleatório longo (gere com `openssl rand -base64 32`). As outras
   variáveis já estão no compose.
5. **Domínio:** na configuração do serviço **`api`**, preencha *Domains* com
   `https://tcc.seudominio.com:8000`. O `:8000` informa ao Traefik a porta interna
   do container e não aparece na URL pública. Deixe o `meilisearch` **sem domínio**.
6. **Deploy.** No log, espere:
   - build da imagem: 5–10 min (torch e modelo);
   - `Índice vetorial vazio — indexando o corpus...`, com o progresso por medicamento: 3–10 min;
   - `Subindo a API`, e o container passa a *healthy*.
7. **Persistent Storage:** confira que os volumes `chroma` e `meili` aparecem. São
   eles que evitam reindexar a cada deploy.
8. **Auto-deploy (opcional):** ative o webhook para redeployar a cada push na `coolify`.

## Verificação

```bash
curl https://tcc.seudominio.com/health
curl https://tcc.seudominio.com/config   # modelo e5-small, coleção ..._e5_small_estr, min_score 0.85
curl "https://tcc.seudominio.com/buscar?q=press%C3%A3o%20alta&n=3"
curl -X POST https://tcc.seudominio.com/embed -H 'Content-Type: application/json' \
     -d '{"textos":["x"]}'               # tem que ser 403: /embed só aceita loopback
```

Depois abra a URL no navegador de um celular **fora** da rede da empresa e faça uma busca.

Recursos, pelo terminal do servidor:

```bash
docker stats --no-stream | grep -Ei 'api|meili'   # API ~0,9 GB; Meili < 250 MB
docker system df
```

Num redeploy sem mudança de chunking nem de modelo, o log deve mostrar
`Índice vetorial presente`, sem reindexar.

## Operação

- **Bulas novas:** acrescente os arquivos em `csv_metadata/` e os mapas
  `remedios_*_map.csv`, faça push e redeploy. Depois, no *Terminal* do container
  `api` no Coolify, rode `python embedding.py`. É incremental e pede os embeddings
  ao `/embed` da própria API, então não há segunda cópia do modelo na memória.
- **Reconstruir do zero** (troca de modelo ou de chunking): pare o recurso, apague o
  volume `chroma` em *Persistent Storage* e faça deploy de novo.
- **Logs:** aba *Logs* de cada serviço no Coolify.

## Problemas comuns

- **Busca responde 503:** o Meilisearch está fora ou a master key diverge. Os dois
  serviços leem a mesma `MEILI_MASTER_KEY`. Depois de trocar a chave, faça redeploy
  para que os dois containers reiniciem com o valor novo.
- **Container reiniciando em loop no 1º boot:** a indexação estourou o limite de
  1,5 GB. Diminua `EMBED_BATCH` de 8 para 4 no compose.
- **Container `unhealthy` durante o 1º boot:** é esperado enquanto indexa. O
  healthcheck tem 20 min de tolerância.
