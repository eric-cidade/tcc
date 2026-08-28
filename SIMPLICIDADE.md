# Cálculo de simplicidade lexical

Documentação do que [simplicidade.py](simplicidade.py) faz: dado um par de termos
(ex.: `cefaleia` × `dor de cabeça`), decidir qual é o **mais simples** usando como
evidência o próprio corpus paralelo de bulas (versão original × versão
simplificada), sem modelo de linguagem e sem motor de busca.

> **Revisão de agosto/2026 (após discussão com a equipe do CorPop).** Três
> mudanças, as duas primeiras detalhadas nas seções
> [4.2](#42-parcela-2--bônus-de-atestação-nas-simplificadas) e
> [4.3](#43-ocorrências-entre-parênteses-não-contam):
> 1. **estar presente na versão simplificada passou a valer pontos por si só**
>    (antes só a *razão* entre os lados contava);
> 2. **ocorrências entre parênteses não são mais contadas**, porque na bula
>    simplificada o parêntese é justamente onde o termo técnico reaparece —
>    «pressão alta (hipertensão)»;
> 3. **a contagem de sílabas foi removida** (§5). Era uma aproximação
>    ortográfica — contava grupos de vogais, errando ditongos e hiatos — e,
>    mesmo assim, decidia comparações. Medida aproximada não decide: o desempate
>    de superfície agora é só o número de caracteres, que ao menos é exato.
>
> E a comparação passou a aceitar um **recorte do corpus** (§6): comparar dentro
> de todas as bulas, só das de hipertensão, só das de oncologia — e, quando
> entrarem outros gêneros de documento, dentro deles.

## 1. Intuição

O corpus é paralelo: cada bula existe em duas versões, a **original** (técnica) e
a **simplificada**. Se um termo aparece proporcionalmente mais nas simplificadas
do que nas originais, foi ele que os simplificadores *escolheram* na hora de
reescrever — logo, é o termo mais acessível. É a abordagem clássica de
simplificação lexical por estatística de corpus: razão de frequências entre um
corpus complexo e um corpus simples.

> ⚠️ **O lado simplificado não é homogêneo.** As 30 bulas de hipertensão foram
> simplificadas e **validadas por linguistas**; as 49 de oncologia foram
> **simplificadas por IA, sem revisão profissional**. Como é esse lado que
> define o que o score chama de "simples", a procedência muda o estatuto da
> evidência — e, medido, muda o resultado. Ver §6.5.

A isso somam-se duas correções que vêm da natureza do corpus (não da
literatura), motivadas pela leitura das bulas validadas:

- **Aparecer na versão validada já é evidência**, mesmo que o termo continue
  frequente nas originais. Uma equipe de simplificação decidiu que aquela
  palavra podia ficar diante de um leitor leigo — isso é um dado, não ruído. A
  razão de frequências sozinha ignora esse fato: um termo usado 20 vezes nas
  simplificadas e 200 nas originais recebe score negativo como se nunca tivesse
  sido aprovado por ninguém.
- **O que está entre parênteses não é linguagem simples.** Na bula simplificada
  o padrão dominante é a glosa `termo leigo (termo técnico)` — a versão validada
  precisa continuar exibindo o termo técnico. Contar essas ocorrências faz
  exatamente o contrário do que a fórmula pretende: transforma a obrigação de
  mostrar o jargão em evidência de que o jargão é simples.

O número de caracteres entra só como desempate, quando o corpus não tem
evidência suficiente — e é o critério mais fraco da cascata (§5).

## 2. Notação

Para um termo $t$ (que pode ser multipalavra):

| Símbolo | Significado |
| --- | --- |
| $c_o(t)$ | ocorrências de $t$ no lado **original**, **fora de parênteses** |
| $c_s(t)$ | ocorrências de $t$ no lado **simplificada**, **fora de parênteses** |
| $p_o(t),\ p_s(t)$ | ocorrências **dentro** de parênteses (reportadas, não contadas) |
| $d_s(t)$ | nº de bulas simplificadas **distintas** com $t$ fora de parênteses |
| $N_o,\ N_s$ | total de tokens de cada lado, **fora de parênteses** |
| $D_s$ | total de documentos simplificados do corpus (79) |

Quando a comparação é feita num **recorte** (§6), todos esses valores — $c$, $d$,
$N$, $D$ — são os do recorte, não os do corpus inteiro.

Tokenização (`_tokenizar_marcado`): o texto é minusculizado e quebrado pelo
padrão `[a-záéíóúâêôãõàüç]+(?:-[a-záéíóúâêôãõàüç]+)*`, ou seja, sequências de
letras portuguesas aceitando hífen interno (`pós-operatório` é **um** token).
Números e pontuação somem. Cada token guarda também um sinalizador de estar ou
não dentro de parênteses. Termos multipalavra são contados como **subsequência
exata de tokens** dentro de cada documento (`_contar_no_lado`), o que ignora de
propósito pontuação e caixa entre as palavras.

No corpus atual (79 bulas de cada lado):

| | fora de parênteses ($N$) | dentro de parênteses | % dentro |
| --- | ---: | ---: | ---: |
| original | 215 279 | 23 896 | 10,0 % |
| simplificada | 95 819 | 9 680 | 9,2 % |

Repare que as simplificadas são **bem mais curtas** — daí toda normalização
abaixo ser por tamanho de lado, nunca por contagem bruta.

## 3. Frequência por milhão

Para poder comparar lados de tamanhos diferentes:

$$\text{fpm}_o(t) = \frac{c_o(t)}{N_o}\times 10^6
\qquad
\text{fpm}_s(t) = \frac{c_s(t)}{N_s}\times 10^6$$

E a **familiaridade** do termo, isto é, quão comum ele é no corpus como um todo
(palavras frequentes tendem a ser mais conhecidas do leitor leigo):

$$\text{fpm}_{\text{total}}(t) = \frac{c_o(t) + c_s(t)}{N_o + N_s}\times 10^6$$

## 4. A fórmula do score de simplicidade

O `score_simplicidade` é a soma de duas parcelas, ambas em **bits**:

$$
\text{score}(t) \;=\; \underbrace{\text{razão}(t)}_{\text{§4.1}} \;+\; \underbrace{\text{atest}(t)}_{\text{§4.2}}
$$

Leitura do resultado:

- **score > 0** → termo característico das bulas simplificadas ⇒ mais simples;
- **score < 0** → característico das originais ⇒ mais técnico;
- **score = 0** → termo sem evidência utilizável no corpus (`no_corpus: false`).

### 4.1. Parcela 1 — razão de frequências

A razão log₂ entre a frequência relativa do termo no lado simplificado e no
lado original, com suavização de Laplace:

$$
\text{razão}(t) =
\log_2 \frac{\dfrac{c_s(t) + 0{,}5}{N_s + 1}}{\dfrac{c_o(t) + 0{,}5}{N_o + 1}}
\qquad \text{se } c_o(t) + c_s(t) > 0
$$

$$\text{razão}(t) = 0 \qquad \text{se } c_o(t) + c_s(t) = 0$$

#### Por que a suavização `+0,5 / +1`

Sem ela, um termo com $c_o = 0$ ou $c_s = 0$ produziria divisão por zero ou
$\log_2 0 = -\infty$. O `+0,5` no numerador de cada frequência e o `+1` no
denominador são o ajuste de Laplace usual, e mantêm o score finito para termos
que só existem de um lado (ex.: `cefaleia`, com 21 ocorrências nas originais e 0
nas simplificadas).

#### Por que o caso ausente é fixado em 0 explicitamente

Este é o detalhe menos óbvio da fórmula. Se o termo não aparece em lado nenhum,
$c_o = c_s = 0$, e a expressão suavizada daria:

$$\log_2 \frac{0{,}5/(N_s+1)}{0{,}5/(N_o+1)} = \log_2 \frac{N_o+1}{N_s+1}
= \log_2 \frac{215\,280}{95\,820} \approx +1{,}17$$

Ou seja: um score **positivo espúrio**, herdado apenas da diferença de tamanho
entre os lados — e palavras inventadas pareceriam "simples". Por isso o código
zera o score nesse caso, por definição: ausência de dado não é evidência de
simplicidade. Com a regra dos parênteses (§4.3) isso passou a proteger também um
segundo caso: o termo que **só** aparece entre parênteses, que é tipicamente
técnico e não pode receber score neutro (a flag `somente_em_parenteses` marca
essa situação na resposta).

### 4.2. Parcela 2 — bônus de atestação nas simplificadas

É a resposta ao ponto levantado pela equipe do CorPop: *o simples fato de a
palavra aparecer no corpus simplificado já é indício forte de que ela é
simples.* A parcela vale zero para quem não aparece e cresce, com saturação, com
o número de bulas simplificadas **distintas** que usam o termo:

$$
\text{atest}(t) = \beta \cdot \frac{\log_2\big(1 + d_s(t)\big)}{\log_2\big(1 + D_s\big)},
\qquad \beta = \texttt{PESO\_ATESTACAO} = 1{,}5 \text{ bits}
$$

Três decisões embutidas aí:

**(a) Dispersão, não frequência bruta.** O bônus usa $d_s$ (em quantas bulas) e
não $c_s$ (quantas vezes). Doze ocorrências espalhadas por doze bulas
simplificadas são evidência muito mais forte do que doze ocorrências na mesma
bula — que podem ser só o efeito de um medicamento cujo tema é aquele termo.
Isso tem respaldo direto: Adelman, Brown & Quesada (2006) mostram que a
*diversidade contextual* (nº de documentos em que a palavra ocorre) prevê o
tempo de reconhecimento lexical **melhor** do que a frequência bruta; medidas de
dispersão são discutidas em Gries (2008).

**(b) Saturação logarítmica.** A primeira bula já vale boa parte do bônus e a
quadragésima acrescenta pouco — porque o que se quer capturar é *"passou pelo
crivo da simplificação"*, um fato quase binário, e não um segundo eixo de
frequência (que já é a parcela 1):

| $d_s$ | 0 | 1 | 3 | 10 | 17 | 51 | 79 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\text{atest}$ | 0,000 | 0,237 | 0,475 | 0,821 | 0,989 | 1,353 | **1,500** |

**(c) A normalização é pelo tamanho do recorte em uso** ($D_s$ do escopo, §6),
para que "presente em metade dos documentos" valha o mesmo em qualquer recorte.

**(d) O teto $\beta = 1{,}5$ bits.** O que decide uma comparação é a *diferença*
de bônus entre os dois termos, então o bônus quase se cancela quando ambos são
atestados e pesa justamente no caso que motiva a regra — um termo atestado na
versão validada contra outro que não é. O teto limita a influência: nenhum bônus
pode virar sozinho uma diferença de razão de frequências maior que 1,5 bit
(≈ fator 2,8 de frequência relativa). `PESO_ATESTACAO` é constante de módulo:
mexer nela é o parâmetro a variar se a equipe quiser calibrar de novo.

Exemplo do efeito, em `edema` × `inchaço`: os dois são atestados (17 e 56 bulas
simplificadas), os bônus são 0,99 e 1,38, quase se cancelam, e quem decide é a
razão — como antes. Já em `dor de cabeça` (atestado em 55 bulas) contra um termo
qualquer ausente das simplificadas, o bônus de 1,38 é o que separa os dois.

### 4.3. Ocorrências entre parênteses não contam

Todas as contagens acima — $c_o$, $c_s$, $d_s$ **e os totais $N_o$, $N_s$** —
descartam tokens dentro de parênteses. Descartar também dos totais é o que
mantém as frequências comparáveis: numerador e denominador falam do mesmo
subcorpus, o texto corrido.

A justificativa é a estrutura do gênero. Na bula simplificada o parêntese é o
lugar onde o termo técnico é reexibido depois da paráfrase leiga:

> Pressão alta também é chamada de hipertensão **(hipertensão arterial)**
> O médico prescreveu losartana potássica para tratar pressão alta **(hipertensão)**
> ou tratar coração enfraquecido **(insuficiência cardíaca)**

Os parentéticos mais frequentes do lado simplificado são, quase todos, jargão:
`(bradicardia)` 31×, `(hipertensão)` 26×, `(broncoespasmo)` 24×,
`(hipoglicemia)` 23×, `(insuficiência cardíaca)` 18×, `(feocromocitoma)` 17×.

O efeito na medição é grande, e no sentido esperado:

| termo | $c_s$ antes | $c_s$ (fora de parênteses) | dentro | razão antes | razão agora |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hipertensão` | 62 | 19 | **43** | −0,43 | **−2,01** |
| `hipotensão` | 21 | 3 | **18** | −1,28 | **−3,73** |
| `dispepsia` | 3 | 0 | **3** | −1,57 | **−4,19** |
| `edema` | 42 | 25 | 17 | −0,80 | −1,17 |

Ou seja: sem a regra, `hipertensão` aparecia quase neutro (−0,43) porque a bula
simplificada é obrigada a citá-lo entre parênteses; com a regra, ele volta a ser
classificado como o termo técnico que é.

Detalhes de implementação:

- `_spans_parenteses` varre o texto com uma pilha, então `(a (b) c)` conta como
  **um** trecho, e um `(` que nunca fecha é ignorado — assim um parêntese solto
  não engole o resto da bula.
- Uma ocorrência multipalavra que **encoste** em parêntese (qualquer um dos
  tokens dentro) é classificada como parentética: não é uso corrido limpo.
- As contagens descartadas não são jogadas fora, e sim devolvidas na resposta
  como `ocorrencias_parenteses`, por lado — é evidência para quem for auditar o
  resultado, e aparece na CLI e na interface.
- Os exemplos de uso (`exemplo_simplificada` / `exemplo_original`) também pulam
  ocorrências entre parênteses: a frase exibida tem que ilustrar o mesmo uso que
  o score contou.

### 4.4. Exemplo numérico completo

`pressão alta`: $c_o = 122$ (mais 55 entre parênteses), $c_s = 184$ (mais 14),
$d_s = 51$.

$$\text{razão} = \log_2 \frac{184{,}5 / 95\,820}{122{,}5 / 215\,280}
= \log_2 \frac{0{,}0019255}{0{,}0005690} = \log_2 3{,}384 \approx +1{,}759$$

$$\text{atest} = 1{,}5 \cdot \frac{\log_2 52}{\log_2 80} = 1{,}5 \cdot \frac{5{,}70}{6{,}32} \approx +1{,}353$$

$$\text{score} = 1{,}759 + 1{,}353 = \boxed{+3{,}111}$$

Contra `hipertensão` ($c_o = 176$, $c_s = 19$, $d_s = 17$):
razão $= -2{,}010$, atestação $= +0{,}989$, **score $= -1{,}021$**. O bônus
reconhece que `hipertensão` não é palavra proibida na versão validada (ela
aparece em 17 bulas fora de parênteses), mas não chega perto de compensar a
razão de frequências. Δscore $= 4{,}13$ a favor de `pressão alta`.

## 5. Cascata de decisão

`comparar(a, b)` calcula as estatísticas dos dois termos e aplica, em ordem, com
$\Delta = \text{score}(a) - \text{score}(b)$ e `MARGEM_SCORE = 0,5`:

| Ordem | `criterio` | Condição | Vencedor |
| --- | --- | --- | --- |
| 1 | `corpus` | $\lvert\Delta\rvert \ge 0{,}5$ **e o vencedor tem evidência no corpus** | maior score |
| 2 | `familiaridade` | pelo menos um no corpus e $\max(\text{fpm}_{\text{total}}) > 1{,}2 \times \min(\text{fpm}_{\text{total}})$ | maior $\text{fpm}_{\text{total}}$ |
| 3 | `heuristica` | nº de caracteres diferente | o mais curto |
| 4 | `empate` | nada distingue os termos | — |

Quatro escolhas de projeto valem nota:

- **A margem de 0,5 bit** existe porque diferenças menores são ruído de
  amostragem num corpus de ~80 pares de bulas. Abaixo dela, a decisão cai para
  familiaridade e superfície.
- **Não há mais contagem de sílabas** (removida nesta revisão). Ela vinha das
  fórmulas clássicas de legibilidade, mas o que o código fazia era contar grupos
  de vogais no texto escrito: ditongos viravam uma sílaba e hiatos ficavam
  subestimados (`saúde` saía com 2 em vez de 3). Silabação em português exige
  análise fonológica, não uma varredura de vogais — e uma medida aproximada não
  deveria estar decidindo qual termo é mais simples. Sobrou o número de
  caracteres: exato, ainda que grosseiro, e último da fila.
- **A guarda do critério 1 é assimétrica** (mudou nesta revisão). Antes exigia
  que os **dois** termos estivessem no corpus, para evitar comparar evidência
  real contra o score neutro (0) de um termo nunca visto. Mas essa versão
  bloqueava também o caso oposto — e ele é justamente o que a equipe do CorPop
  quer premiar: um termo **atestado na versão validada** contra um termo sem
  nenhuma evidência. Agora a exigência recai só sobre o **vencedor**: se quem
  ganharia é o termo fora do corpus (score 0 batendo um score negativo), a
  decisão passa adiante; se quem ganha é o termo com evidência, o corpus decide.
- **O bônus de atestação não substitui a cascata**: um termo pode ser atestado e
  ainda assim perder, se a razão de frequências for suficientemente negativa
  (é o caso de `hipertensão`).

O resultado ainda traz `exemplo_simplificada` / `exemplo_original`: a primeira
sentença de cada lado que contém o termo fora de parênteses (com mais de 15
caracteres), como evidência de uso.

## 6. Recorte do corpus: dois eixos

A comparação aceita dois filtros **independentes e cruzáveis**, que juntos
definem o subcorpus dentro do qual as frequências são contadas:

| eixo | parâmetro | valores | omitido |
| --- | --- | --- | --- |
| **tema / gênero** | `escopo` | `bula`, `bula/hipertensao`, `bula/oncologia` | todo o corpus |
| **procedência** do lado simplificado | `proveniencia` | `humana`, `ia` | qualquer (mistura) |

São eixos separados de propósito, e não dois níveis do mesmo caminho: "só o que
foi validado por humanos" é uma pergunta independente de "só oncologia". Hoje
eles coincidem por acidente do corpus — todo o material validado é de
hipertensão —, mas deixam de coincidir assim que a oncologia for revisada ou
outro gênero entrar. Ver §6.5.

### Por que

O score mede uma escolha editorial *situada*. As equipes que simplificaram bulas
de hipertensão e bulas de oncologia tomaram decisões diferentes, e a pergunta
"qual termo é mais simples?" só faz sentido em relação a um universo de
documentos. Sem recorte, o veredito do corpus inteiro é dominado pelo subcorpus
maior — hoje oncologia, com 49 das 79 bulas.

O caso mais nítido é `náusea` × `enjoo`:

| recorte | `náusea` | `enjoo` | vence |
| --- | ---: | ---: | --- |
| todo o corpus | +1,96 | +0,98 | `náusea` |
| bulas de hipertensão | +1,82 | **+3,79** | **`enjoo`** |
| bulas de oncologia | +2,10 | +0,22 | `náusea` |

Nas bulas de hipertensão, `enjoo` aparece 14 vezes nas simplificadas contra 3
nas originais — foi claramente a palavra escolhida na reescrita. Nas de
oncologia acontece o contrário (14 × 65), e é isso que o corpus inteiro reporta.
As duas respostas estão certas; o que faltava era poder perguntar separado.

### A taxonomia do eixo temático

Os documentos formam uma **árvore**, e o identificador de um escopo é o caminho
nela, separado por `/`:

```
(vazio)                       todo o corpus
└── bula                      Bulas de medicamento          79 + 79 documentos
    ├── bula/hipertensao      Hipertensão                   30 + 30
    └── bula/oncologia        Oncologia                     49 + 49
```

| recorte | documentos (por lado) | $N_o$ | $N_s$ |
| --- | ---: | ---: | ---: |
| todo o corpus / `bula` | 79 | 215 279 | 95 819 |
| `bula/hipertensao` | 30 | 64 143 | 42 814 |
| `bula/oncologia` | 49 | 151 136 | 53 005 |

A árvore mora em [bulas.py](bulas.py): cada entrada de `TIPOS` declara seu
`escopo` (o caminho), e `ROTULOS_ESCOPO` dá o nome legível de cada nó — indexado
pelo caminho **inteiro**, para que dois gêneros possam ter subgrupos homônimos.
O filtro (`escopo_casa`) é por **prefixo de caminho**, comparado segmento a
segmento: `bula` casa `bula/hipertensao`, e `bula/hiper` não casa nada.

Consequência prática: **a profundidade da árvore não está codificada em lugar
nenhum**. Acrescentar `consentimento/exames-de-imagem` — ou um terceiro nível,
`consentimento/exames-de-imagem/tomografia` — é declarar o escopo na entrada de
`TIPOS` e os rótulos dos nós novos. Nem o comparador, nem a API, nem o
front-end mudam: a interface monta o filtro a partir de `GET /escopos`.

### Receita para acrescentar um gênero ou subgrupo

1. Coloque os textos em pastas `original/` e `simplificada/` (ou os sufixos que
   o gênero usar) e acrescente uma entrada em `TIPOS` com `tipo`, `escopo`,
   `map_csv`, os dois `path_*` e os dois `suffix_*`.
2. Acrescente em `ROTULOS_ESCOPO` o rótulo de cada nó **novo** do caminho — o
   gênero e o subgrupo.
3. Pronto. `GET /escopos` já lista, a CLI já aceita `--escopo`, e o filtro da
   página já mostra a opção com a contagem de documentos.

### O que muda no cálculo

Tudo o que é "do corpus" passa a ser "do recorte": $N_o$, $N_s$ e $D_s$ (a
normalização do bônus de atestação), além das próprias contagens. A resposta traz
o bloco `corpus` com o recorte usado, seu tamanho e sua procedência — o score
não é interpretável sem isso à vista.

O filtro de procedência vale para os **dois lados**, não só para o simplificado:
o par original↔simplificada de uma bula é a unidade de evidência, e comparar o
lado simplificado de um subcorpus com o original de outro produziria uma razão
de frequências sem sentido.

Cruzamentos vazios (`bula/hipertensao` × `ia`, hoje) são **recusados** com 422 em
vez de devolver zeros — e a interface já desabilita a opção impossível, a partir
do `documentos_por_proveniencia` que `GET /escopos` devolve por nó.

### 6.5. Procedência: nem todo lado simplificado vale o mesmo

O recorte não é só um filtro temático. Os dois subcorpora foram produzidos de
maneiras diferentes, e isso está registrado na taxonomia (`proveniencia`, em
[bulas.py](bulas.py)), aparece em `GET /escopos`, no bloco `corpus` de cada
comparação, na CLI e na interface:

| recorte | pares | lado simplificado | `simp/orig` (tokens) |
| --- | ---: | --- | ---: |
| `bula/hipertensao` | 30 | simplificação **validada por linguistas** | 67 % |
| `bula/oncologia` | 49 | simplificação **gerada por IA, sem revisão profissional** | 35 % |

Duas consequências, uma óbvia e uma medida.

**A óbvia:** no recorte padrão (todo o corpus), 49 dos 79 documentos — 55 % dos
tokens do lado simplificado — não passaram por revisão humana. O score "do
corpus inteiro" é, em maioria, uma medida do vocabulário que a IA deixou passar.

**A medida:** os dois subcorpora concordam menos do que se esperaria. Tomando as
966 palavras com ao menos 10 ocorrências em ambos e calculando o score em cada
um separadamente:

- correlação entre os scores: **r = 0,73** (Spearman ρ = 0,73);
- **170 dessas 966 palavras (18 %) trocam de sinal** — uma diz "termo simples",
  a outra diz "termo técnico".

Parte disso é tema, não método: `glândula` pontua alto em hipertensão porque
aquelas bulas falam de suprarrenal e feocromocitoma. Mas há um grupo de
divergências que **não** é explicável por tema — palavras de registro formal,
presentes nas ORIGINAIS dos dois subcorpora (o que controla o efeito temático),
eliminadas pelos linguistas e mantidas pela IA:

| palavra | HT orig → simp | score HT | ONCO orig → simp | score ONCO |
| --- | ---: | ---: | ---: | ---: |
| `posologia` | 24 → **0** | −5,03 | 27 → **12** (em 12 bulas) | **+1,36** |
| `medicamentosas` | 23 → **0** | −4,97 | 39 → **13** | **+0,97** |
| `situações` | 18 → **0** | −4,63 | 23 → **10** | **+1,19** |
| `induzida` | 15 → **0** | −4,37 | 10 → **5** | **+1,11** |
| `diabéticos` | 12 → **0** | −4,06 | 7 → **4** | **+1,31** |
| `interações` | 23 → 7 | −0,16 | 62 → **51** (em 41 bulas) | **+2,67** |

O padrão é sistemático: o subcorpus validado por humanos **zera** o vocabulário
de bula formal, enquanto o gerado por IA o preserva — e, como a IA também
comprime muito mais (35 % contra 67 %), o que ela preserva ganha frequência
relativa alta e **pontua como palavra simples**. `posologia` com score +1,36 é o
caso emblemático: nenhum linguista deixaria "posologia" num texto para leigo, e
é exatamente isso que os 30 pares validados mostram.

É também a explicação do `náusea` × `enjoo` do começo desta seção: o veredito
"náusea" vem do subcorpus de IA e prevalece no corpus inteiro por maioria de
documentos; os 30 pares validados por linguistas dizem `enjoo`.

**Implicação para o TCC.** O subcorpus validado por linguistas é a única
evidência de escolha editorial humana que o projeto tem, e é o menor dos dois.
Misturá-lo com saída de IA não revisada num único score contamina a referência —
tanto que a mistura já inverte a resposta de pares conhecidos.

**Decisão tomada:** a procedência virou o **segundo eixo de filtro** (§6), e não
um novo nível da árvore temática. `proveniencia=humana` responde "o que os
linguistas escolheram", `proveniencia=ia` responde "o que a IA deixou passar", e
omitir o filtro mistura as duas — caso em que toda resposta vem com a flag
`proveniencia_mista` e a interface destaca o aviso. O default segue sendo o
corpus inteiro, para não fingir que 30 pares são o corpus todo.

Duas saídas continuam abertas, para discutir com a equipe:

1. **Mudar o default** para `proveniencia=humana`, tratando o restante como
   corpus de cobertura. Score mais confiável, base bem menor (30 pares) — e,
   como mostra o fim desta seção, 30 pares já deixam alguns pares conhecidos
   sem evidência.
2. **Ponderar a evidência pela procedência** dentro da própria fórmula (peso
   menor a ocorrências vindas de texto não revisado), em vez de separar em
   recortes. É o caminho mais invasivo e o que mais exige justificativa
   metodológica na escrita.

Uma ressalva sobre a comparação acima: HT e ONCO diferem em procedência **e** em
área temática, então nenhuma das duas coisas pode ser isolada com todo o rigor.
O controle usado — exigir que a palavra apareça nas originais dos dois lados —
reduz o efeito de tema, mas não o elimina. Um teste limpo exigiria o mesmo
conjunto de bulas simplificado das duas maneiras.

### O preço: menos evidência

Recorte menor significa margem de decisão mais otimista do que ela realmente é —
`MARGEM_SCORE = 0,5` foi pensada para ~80 pares, não para 30. Na verificação da
§8, `bula/hipertensao` (30 pares) é o único recorte em que pares como
`mialgia` × `dor muscular` e `epistaxe` × `sangramento nasal` deixam de ser
decididos pelo corpus e caem na heurística de comprimento — que erra os dois. Não é
defeito do recorte, é falta de dado: nesse subcorpus os quatro termos aparecem
pouquíssimo. A resposta sinaliza a situação (`no_corpus: false` por termo,
critério `heuristica`, tamanho do recorte em `corpus`), e a leitura correta é
"este recorte não sabe responder", não "o termo técnico é mais simples".

## 7. Uso

CLI:

```bash
uv run python simplicidade.py cefaleia "dor de cabeça"
uv run python simplicidade.py náusea enjoo --proveniencia humana
uv run python simplicidade.py náusea enjoo --escopo bula/hipertensao
uv run python simplicidade.py --listar-escopos   # os dois eixos e seus tamanhos
```

API (ver [api.py](api.py)):

```
GET /simplicidade?a=cefaleia&b=dor%20de%20cabe%C3%A7a
GET /simplicidade?a=n%C3%A1usea&b=enjoo&escopo=bula/hipertensao
GET /simplicidade?a=n%C3%A1usea&b=enjoo&proveniencia=humana
GET /escopos      → os dois eixos, com o nº de documentos de cada cruzamento
```

Um cruzamento sem documentos (ex.: `escopo=bula/hipertensao&proveniencia=ia`)
responde **422**, não um resultado vazio.

Resposta (resumida):

```json
{
  "a": { "termo": "cefaleia", "no_corpus": true, "somente_em_parenteses": false,
         "original":     { "ocorrencias": 21, "documentos": 15,
                           "ocorrencias_parenteses": 0, "freq_por_milhao": 97.55 },
         "simplificada": { "ocorrencias": 0,  "documentos": 0,
                           "ocorrencias_parenteses": 0, "freq_por_milhao": 0.0 },
         "score_simplicidade": -4.258,
         "razao_frequencias": -4.258, "bonus_atestacao": 0.0,
         "atestado_simplificada": false, "caracteres": 8 },
  "b": { "termo": "dor de cabeça", "no_corpus": true, "somente_em_parenteses": false,
         "original":     { "ocorrencias": 97, "documentos": 48,
                           "ocorrencias_parenteses": 16, "freq_por_milhao": 450.58 },
         "simplificada": { "ocorrencias": 65, "documentos": 55,
                           "ocorrencias_parenteses": 0, "freq_por_milhao": 678.36 },
         "score_simplicidade": 1.972,
         "razao_frequencias": 0.594, "bonus_atestacao": 1.378,
         "atestado_simplificada": true, "caracteres": 13 },
  "mais_simples": "dor de cabeça",
  "criterio": "corpus",
  "delta_score": -6.23,
  "corpus": {
    "escopo": "", "rotulo": "Todo o corpus", "proveniencia": "",
    "documentos": { "original": 79, "simplificada": 79 },
    "tokens": { "original": 215279, "simplificada": 95819 },
    "tokens_parenteses": { "original": 23896, "simplificada": 9680 },
    "proveniencias": [
      { "id": "humana", "rotulo": "Simplificação validada por linguistas",
        "curto": "validada por linguistas" },
      { "id": "ia", "rotulo": "Simplificação gerada por IA, sem revisão profissional",
        "curto": "IA sem revisão" }
    ],
    "proveniencia_mista": true
  }
}
```

Os `.txt` são lidos e tokenizados uma única vez (`_carregar_docs`); as contagens
de cada recorte são montadas sob demanda e memoizadas (`_indice`), de modo que
comparar em vários escopos na mesma sessão não relê disco. Nada de modelo de
embeddings ou ChromaDB/Meilisearch entra nesse cálculo.

## 8. Verificação empírica

Não há conjunto de referência anotado, mas os 15 grupos de sinônimos
técnico↔leigo de [sinonimos.py](sinonimos.py) (construídos independentemente,
para a busca) servem como teste de sanidade: em cada par, o termo leigo deveria
vencer. Resultado: **15 dos 17 pares** decididos a favor do termo leigo.
(Lembrando a §6.5: no recorte padrão, a maioria da evidência vem de
simplificação não revisada — o teste mede o corpus como ele é, não como deveria
ser.) As duas
exceções:

- `hipertensão` × `hipertensão arterial` — os dois são técnicos; a expectativa
  do teste não se aplica (o grupo lista três variantes, e a leiga,
  `pressão alta`, vence as duas).
- `náusea` × `enjoo` — `náusea` vence, e vencia também antes desta revisão. Não
  é efeito das mudanças: as bulas validadas simplesmente usam mais `náusea`
  (63 ocorrências em 46 bulas, fora de parênteses) do que `enjoo` (28 em 22).
  Restringindo o recorte, porém, o quadro se inverte nas bulas de hipertensão
  (§6) — é o exemplo de por que o filtro por corpus faz falta.

Repetindo o mesmo teste por recorte:

| recorte | acertos | observação |
| --- | ---: | --- |
| todo o corpus | 15/17 | — |
| `bula` | 15/17 | idêntico (só há bulas no corpus hoje) |
| `bula/oncologia` | 16/17 | o "acerto" extra é o par `hipertensão` × `hipertensão arterial`, cuja expectativa não se aplica — não é melhora real |
| `bula/hipertensao` | 14/17 | 30 pares: `mialgia` × `dor muscular` e `epistaxe` × `sangramento nasal` caem na heurística por falta de dado |
| `proveniencia=humana` | 14/17 | idêntico ao anterior — hoje os dois eixos coincidem (§6.5) |

## 9. Limitações

- **Corpus pequeno** (79 pares): scores com $\lvert\Delta\rvert$ pequeno não são
  confiáveis — daí a margem.
- **Domínio fechado**: as frequências refletem bulas de hipertensão e oncologia;
  fora desse domínio o score perde sentido.
- **A regra dos parênteses é uma aproximação.** Nem todo parêntese é glosa
  técnica: há também glosa na direção contrária (`edema (inchaço)`), faixas de
  valores (`(15°C a 30°C)`, `(1% a 10%)`) e apostos comuns. Descartar tudo
  perde alguma evidência a favor de termos leigos, mas a perda é pequena porque
  o termo leigo tem farto uso corrido: `inchaço` mantém 135 das 175 ocorrências
  nas simplificadas, `pressão alta` 184 de 198. Um refinamento possível é
  classificar o conteúdo do parêntese (glosa técnica × glosa leiga × valor
  numérico) em vez de descartar em bloco.
- **Procedência mista do lado simplificado** (§6.5): 49 dos 79 pares foram
  simplificados por IA sem revisão profissional, e é esse lado que define a
  referência de "simples". O recorte padrão continua híbrido — o eixo de
  procedência permite isolar, e a flag `proveniencia_mista` avisa, mas quem não
  filtrar recebe um score de referência mista.
- **Recorte pequeno amplifica o ruído.** A margem de 0,5 bit é a mesma em
  qualquer escopo, mas 30 pares de bulas dão muito menos evidência que 79 (§6).
  Se a árvore ganhar subgrupos pequenos, o caminho é tornar a margem função do
  tamanho do recorte — ou trocá-la pelo teste $G^2$ já sugerido abaixo, que
  incorpora o tamanho da amostra por construção.
- **O bônus de atestação é binário demais para casos raros**: um termo que
  aparece uma única vez, numa única bula simplificada, já leva 0,237 bit.
- **Sem lematização**: `comprimido` e `comprimidos` são tokens distintos.
- **Termos fora do corpus** recebem score 0 e são decididos só por superfície.

## 10. Fundamentação: de onde vem cada peça

O cálculo não é inventado aqui — cada componente tem precedente na literatura de
linguística de corpus e de simplificação lexical. O que é original deste projeto
é a *combinação* das peças e a calibração para o corpus de bulas.

| Componente | Precedente na literatura |
| --- | --- |
| $\log_2$ da razão de frequências relativas entre dois corpora | **Log Ratio**, Hardie (2014) — é literalmente a mesma fórmula, usada como medida de *keyness* / tamanho de efeito. Ver também Evert (2022) e Gabrielatos & Marchi (%DIFF, equivalente). |
| Razão de frequências entre corpus complexo e corpus simples como medida de complexidade lexical | **Biran, Brody & Elhadad (2011)**: definem *corpus complexity* $C_w = f_{w,\text{English}} / f_{w,\text{Simple}}$ (Wikipedia × Simple Wikipedia). Nosso score é o **inverso em log₂** disso (medimos simplicidade, eles complexidade). |
| Somar $0{,}5$ para tratar frequência zero | Correção de **Haldane–Anscombe** (adicionar 0,5 às células de uma tabela 2×2 para o log da razão não divergir). O AntConc (Anthony) aplica uma versão *quasi*-Haldane–Anscombe exatamente assim no cálculo de Log Ratio. |
| Familiaridade = frequência total ⇒ simplicidade | Linha clássica desde **Devlin & Tait (1998)**, que ranqueiam sinônimos pela frequência no banco psicolinguístico Kučera–Francis; consolidada no survey de **Paetzold & Specia (2017)**: "quanto mais comumente uma palavra ocorre, mais familiar ela é ao leitor". |
| Familiaridade aplicada a **texto médico para leigos** | **Leroy, Kauchak & Mork (2013)** estimam *term familiarity* por frequência de corpus para simplificar texto de saúde ao consumidor — o cenário mais próximo do das bulas. |
| **Dispersão / diversidade contextual** (nº de documentos) em vez de frequência bruta — base do bônus de atestação | **Adelman, Brown & Quesada (2006)**: a diversidade contextual prevê o desempenho em leitura melhor que a frequência. **Gries (2008)** formaliza medidas de dispersão em linguística de corpus. |
| Glosa entre parênteses como estratégia de simplificação (o que motiva descartá-la da contagem) | *Elaborative simplification* — acrescentar definição/aposto em vez de substituir o termo: **Yano, Long & Ross (1994)**; no PT-BR, a operação de "explicitação/elaboração" do **PorSimples** (Aluísio & Gasperin, 2010). |
| Comprimento como medida de dificuldade | *Lexical complexity* $L_w = \lvert w \rvert$ (nº de caracteres) em Biran et al. (2011). A outra medida clássica desse eixo — a contagem de sílabas das fórmulas de legibilidade tipo Flesch — foi **removida** deste projeto: sem silabação fonológica ela não é calculável com precisão (§5). |
| Corpus paralelo original × simplificado como fonte de evidência | **Coster & Kauchak (2011)** e **Horn, Manduca & Kauchak (2014)** (Wikipedia/Simple Wikipedia); em português, **PorSimples** (Aluísio & Gasperin, 2010). |

### O que é decisão deste projeto (sem precedente direto)

Quatro pontos são escolhas de engenharia calibradas para este corpus e devem ser
apresentados como tal, não como método consagrado:

1. **Fixar o score em 0 para termos ausentes.** Ferramentas de *keyness*
   normalmente apenas descartam itens de frequência zero; aqui o termo precisa
   receber *algum* valor porque a API responde sobre qualquer par de termos, e
   0 (mais a flag `no_corpus`) é o valor neutro coerente.
2. **A margem de 0,5 bit.** A literatura reconhece que medidas de tamanho de
   efeito como o Log Ratio são enviesadas para itens de baixíssima frequência e
   recomenda combiná-las com um filtro de significância — tipicamente
   log-likelihood ($G^2$) (Evert, 2022; Pojanapunya & Watson Todd, 2018). Nossa
   margem fixa é uma aproximação grosseira desse filtro. **Um caminho natural de
   melhoria para o TCC é trocar a margem por um teste $G^2$ de fato**, ou pelo
   log-odds ratio com prior de Dirichlet informativo de Monroe, Colaresi & Quinn
   (2008), que já embute a incerteza no próprio score.
3. **O bônus de atestação e o teto $\beta = 1{,}5$ bits.** A *forma* (dispersão,
   com saturação logarítmica) tem respaldo em Adelman et al. (2006); somá-la ao
   Log Ratio como segunda parcela e o valor de $\beta$ são calibração local.
   Alternativa mais principiada, se houver tempo: modelar as duas evidências
   num único log-odds bayesiano com prior informativo (Monroe et al., 2008), em
   que a atestação entra como massa de prior em vez de um termo aditivo.
4. **Descartar tokens entre parênteses dos dois lados e dos totais.** É uma
   decisão de pré-processamento motivada pelo gênero "bula simplificada", com a
   evidência quantitativa da §4.3; não é prática padrão em linguística de corpus.
   O refinamento natural é classificar o conteúdo do parêntese em vez de
   descartar em bloco.

### Contexto institucional

Este trabalho se insere na linha de **Acessibilidade Textual e Terminológica
(ATT)** da UFRGS, coordenada por Maria José Bocorny Finatto (projeto
CorPop-Saúde / PorPopular, ferramenta MedSimples), que tem como referência de
português popular escrito o corpus **CorPop** (Pasqualini, 2018). As duas
mudanças de agosto/2026 documentadas aqui saíram de discussão com essa equipe.

## 11. Referências

- Adelman, J. S.; Brown, G. D. A.; Quesada, J. F. (2006). *Contextual Diversity,
  Not Word Frequency, Determines Word-Naming and Lexical Decision Times.*
  Psychological Science, 17(9):814–823.
- Aluísio, S. M.; Gasperin, C. (2010). *Fostering Digital Inclusion and
  Accessibility: The PorSimples project for Simplification of Portuguese Texts.*
  In: Proceedings of the NAACL HLT 2010 Young Investigators Workshop on
  Computational Approaches to Languages of the Americas, pp. 46–53.
  <https://aclanthology.org/W10-1607/>
- Biran, O.; Brody, S.; Elhadad, N. (2011). *Putting it Simply: a Context-Aware
  Approach to Lexical Simplification.* In: Proceedings of ACL-HLT 2011,
  pp. 496–501. <https://aclanthology.org/P11-2087/>
- Coster, W.; Kauchak, D. (2011). *Simple English Wikipedia: A New Text
  Simplification Task.* In: Proceedings of ACL-HLT 2011.
- Devlin, S.; Tait, J. (1998). *The use of a psycholinguistic database in the
  simplification of text for aphasic readers.*
- Evert, S. (2022). *Measuring Keyness.*
  <https://www.stephanie-evert.de/PUB/Evert2022.pdf>
- Gries, S. Th. (2008). *Dispersions and adjusted frequencies in corpora.*
  International Journal of Corpus Linguistics, 13(4):403–437.
- Hardie, A. (2014). *Log Ratio: an informal introduction.* ESRC Centre for
  Corpus Approaches to Social Science (CASS), Lancaster University.
  <http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/>
- Horn, C.; Manduca, C.; Kauchak, D. (2014). *Learning a Lexical Simplifier
  Using Wikipedia.* In: Proceedings of ACL 2014.
- Leroy, G.; Kauchak, D.; Mork, J. (2013). *User Evaluation of the Effects of a
  Text Simplification Algorithm Using Term Familiarity on Perception,
  Understanding, Learning, and Information Retention.* Journal of Medical
  Internet Research, 15(7):e144. <https://www.jmir.org/2013/7/e144/>
- Monroe, B. L.; Colaresi, M. P.; Quinn, K. M. (2008). *Fightin' Words: Lexical
  Feature Selection and Evaluation for Identifying the Content of Political
  Conflict.* Political Analysis, 16(4):372–403.
- Paetzold, G. H.; Specia, L. (2017). *A Survey on Lexical Simplification.*
  Journal of Artificial Intelligence Research, 60:549–593.
  <https://www.jair.org/index.php/jair/article/view/11091>
- Pasqualini, B. F. (2018). *CorPop: um corpus de referência do português
  popular escrito do Brasil.* Tese (Doutorado), UFRGS.
  <https://lume.ufrgs.br/handle/10183/177566>
- Pojanapunya, P.; Watson Todd, R. (2018). *Log-likelihood and odds ratio:
  Keyness statistics for different purposes of keyword analysis.* Corpus
  Linguistics and Linguistic Theory, 14(1).
- Yano, Y.; Long, M. H.; Ross, S. (1994). *The effects of simplified and
  elaborated texts on foreign language reading comprehension.* Language
  Learning, 44(2):189–219.
