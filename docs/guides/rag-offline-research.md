<a id="offline-retrieval-and-selective-intervention-research"></a>

# Pesquisa offline de recuperação e intervenção seletiva

[Índice da documentação](../README.md) · [Configuração](../../configs/rag-research.json)

O [estudo concluído](../results/rag-offline-research.md) não encontrou candidato
que atendesse aos critérios para uma avaliação paga. Mantenha o padrão histórico
de stacking sem chamadas ao GPT.

Este fluxo não adiciona comando de execução GPT. Inspeciona artefatos congelados,
extrai representações locais, compara recuperação e reavalia respostas GPT
existentes. Não modifica SVM, LR, NorBERTo, Qwen, stacking, suas partições de
treino, pacotes históricos ou identidades dos caches de respostas.

O pacote independente `rag_research` preserva as identidades históricas do código
verificadas por `rag_cascade` e `cascade_classifier`. Uma cópia do código e
identidades SHA-256 acompanham os experimentos medidos. O código de pesquisa
não é promovido automaticamente aos padrões do ensemble.

<a id="data-boundaries"></a>

## Limites dos dados

- Banco: somente as 16,093 linhas originais de treino dos modelos-base. As 3,999
  linhas reservadas ficam fora do ajuste da recuperação. O seletor herdado
  deduplica textos normalizados, remove duplicatas com rótulos conflitantes e
  limita exemplos brutos a 1,600 caracteres; restam 11,562 exemplos.
- Consultas: as 1,972 linhas de meta-treinamento com saídas aninhadas OOF de
  stacking fixadas por checksum. As 2,027 linhas de calibração não são
  reutilizadas como consultas independentes.
- Desenvolvimento da recuperação: folds externos originais 0/1/2. Auditoria:
  folds 3/4. Grupos de similaridade não cruzam essa fronteira. O avaliador
  consome o gabarito só após selecionar exemplos, nunca o recuperador.
- Pesquisa de intervenções: apenas as 48 consultas do piloto e as 188 da
  confirmação têm respostas GPT para os quatro contextos pareados. Resultados
  não consultados são desconhecidos; a reavaliação do roteamento restringe-se
  explicitamente à cobertura registrada.
- Nenhuma entrada ou gabarito de teste1/teste2 é usado. A exposição histórica
  na seleção do checkpoint Qwen e no desenvolvimento do roteador OOF continua
  sendo uma limitação herdada.

<a id="frozen-representations-and-caches"></a>

## Representações congeladas e caches

NorBERTo é um ModernBERT de 22 camadas, dimensão oculta 768, com o adaptador
LoRA original ajustado de posto 4 em Wqkv. O SHA-256 verificado começa por
`d693f6e2`. O classificador original usa agregação por média; o tokenizador
real insere [CLS], ID 3, como primeiro token. Avaliam-se média mascarada e
primeiro token nas camadas 11 e 22, em FP32/SDPA e com a entrada início/fim
inalterada de 512 tokens.

As quatro representações vêm de uma única passagem congelada por lote. A
representação final usa `last_hidden_state` **após** a normalização final,
enquanto `hidden_states[-1]` do modelo instalado fica antes dela. Estados
intermediários são indexados pelo número de camadas concluídas. A média exclui
preenchimento e inclui tokens especiais atendidos, conforme o contrato do
classificador.

Cada vetor é finito, normalizado em L2 e verificado por checksum em cache SQLite
separado. O espaço de chaves inclui identidades da base/adaptador, código de
extração, tokenização, agregação/camadas, ambiente numérico, dispositivo e lote.
Valores com cache frio/quente preservam os mesmos bits armazenados. Gravações
por lote permitem retomar após interrupção. Arrays NPZ exportados usam
`allow_pickle=False`, IDs originais, hashes de texto e identificação separada
de banco/consultas.

O cache antigo `.cache/norberto-logistic/` contém atributos da base sem adaptação;
não podem ser apresentados como vetores do modelo ajustado. Ele é preservado.
O cache E5 existente é reutilizado, codificando localmente textos ausentes,
sem downloads.

<a id="retrieval-methods-and-ablations"></a>

## Métodos de recuperação e ablações

`ConfigurableRetriever` suporta TF-IDF, E5, NorBERTo, TF-IDF+E5,
TF-IDF+NorBERTo, E5+NorBERTo e os três juntos. Vetores densos usam similaridade
cosseno. As fusões usam posições recíprocas com pesos iguais
(`61 / (60 + rank)`), preservando empates e a escala do híbrido histórico.
Não se calcula a média dos escores brutos TF-IDF/E5/NorBERTo.

A configuração seleciona camadas/agregações do NorBERTo e o ambiente numérico.
`SelectionPolicy` configura independentemente peso de relevância, bônus por
classe ausente, limite por classe e quantidade de candidatos. O experimento
fixo compara:

| Política | Peso da relevância | Bônus de classe | Máximo por classe | Candidatos |
|---|---:|---:|---:|---:|
| legacy | 0.8 | 0.04 | 2 | 120 |
| no_bonus | 0.8 | 0 | 2 | 120 |
| no_cap | 0.8 | 0.04 | sem limite | 120 |
| no_bonus_or_cap | 0.8 | 0 | sem limite | 120 |
| relevance | 1 | 0 | sem limite | 120 |
| no_redundancy | 1 | 0.04 | 2 | 120 |
| pool40 | 0.8 | 0.04 | 2 | 40 |
| pool240 | 0.8 | 0.04 | 2 | 240 |
| weight50 | 0.5 | 0.04 | 2 | 120 |

O método atual de duas fontes é reproduzido como braço `tfidf_e5`/`legacy`.
A grade completa contém 171 combinações de métodos, quatro representações do
NorBERTo e políticas, mais referências aleatória e por comprimento, sem acesso
aos rótulos. Todas mantêm proteções de duplicatas/grupos. Consultas idênticas,
cosseno lexical >=0.95, quase duplicatas por fragmentos de caracteres e
componentes de treino repetidos são excluídos das demonstrações.

As saídas incluem Hit@1, cobertura de classes em três exemplos, pureza de rótulos,
revocação por documento, frequências das classes recuperadas, diversidade,
médias de cosseno, posições dos candidatos e diagnósticos por classe. Revocação@3
por classe significa ao menos um exemplo da classe verdadeira;
a revocação por documento divide pelo tamanho dessa classe no banco. São
medidas auxiliares de recuperação, não macro-F1 do GPT.

A proposta selecionada só no desenvolvimento maximiza Hit@1 balanceado e depois
pureza balanceada. A auditoria separada estima se a proposta se sustenta.
O relatório também expõe compromissos entre cobertura de classes e diversidade;
não presume superioridade de exemplos homogêneos ou diversificados.

<a id="distinct-routing-and-acceptance-decisions"></a>

## Decisões distintas de roteamento e aceitação

Atributos anteriores à chamada usam rótulos nativos, probabilidades calibradas
locais/do stacking, margem, entropia, divergência e evidência opcional da
recuperação original. Após a chamada, podem acrescentar a classe real retornada
pelo GPT e sua concordância com votos-base, stacking e exemplos recuperados.
Nenhum recebe o rótulo verdadeiro da consulta como atributo de inferência.

Os modelos de utilidade estimam `P(corrected) - P(harmed)` por regressão logística
regularizada. Essas pequenas políticas novas não reajustam o classificador
logístico base do projeto. Codificação por dicionário, escalonamento,
regularização, limiares e seleção de famílias são ajustados somente nos folds
agrupados de treino. Cinco folds externos originais avaliam o seletor completo,
com três internos para selecionar parâmetros. Uma seleção somente no piloto
é reavaliada na confirmação. Regras e resultados por família são descritivos,
ao lado do seletor conjunto aninhado principal.

As previsões OOF do stacking anterior permanecem fixas. O meta-modelo histórico
de uma linha de treino da política pode ter usado rótulos de um fold de
avaliação da política; a nova política nunca ajusta grupos de avaliação.
Portanto, essa reavaliação condicional não é uma nova estimativa aninhada
imparcial de ponta a ponta. Regenerar ajustes anteriores exigiria estudo
autorizado separadamente; nenhum resultado positivo por família é usado para
afirmar ou promover uma cascata melhor.

Todas as comparações restringem chamadas às 236 consultas históricas com
resultados GPT conhecidos. Manter stacking nos alvos não consultados integra
a política restrita de reavaliação, não é resposta GPT imputada. Novo roteamento
fora dessa cobertura não pode ser validado sem novas respostas. Respostas em
cache nunca são associadas a novos exemplos do NorBERTo como se o GPT tivesse
recebido esses contextos.

`rule_can_change_prediction` pode evitar uma chamada com segurança quando
nenhuma classe GPT válida poderia passar pela regra posterior e mudar stacking.
Essa equivalência matemática reduz consultas mantendo as saídas da regra.
Não estabelece melhora de qualidade. A economia por alvo precisa ser
reagrupada em lotes para estimar a redução real de chamadas à API.

<a id="execute-locally-without-gpt"></a>

## Execução local sem GPT

Use novos diretórios para preservar saídas anteriores. O script usa a imagem
existente `pln-rocm:6.4.1` para extração com rede desativada. A extração em GPU
usa a compatibilidade existente da RX 6650 XT; altere `device` para `cpu` em
configuração separada para omitir dispositivos GPU.

```bash
bash scripts/rag/offline.sh encode \
  --output-dir results/rag-offline-new/features

bash scripts/rag/offline.sh retrieval \
  --embeddings results/rag-offline-new/features \
  --output-dir results/rag-offline-new/retrieval

bash scripts/rag/offline.sh interventions \
  --output-dir results/rag-offline-new/interventions

bash scripts/rag/offline.sh report \
  --retrieval results/rag-offline-new/retrieval \
  --interventions results/rag-offline-new/interventions \
  --output-dir results/rag-offline-new/report
```

`RAG_RESEARCH_CONFIG`, `RAG_RESEARCH_PYTHON`, `RAG_RESEARCH_RUNTIME` e
`NORBERTO_IMAGE` configuram o script. O ambiente CPU local padrão é
`.cache/cascade-runtime:.cache/rag-runtime`; outro ambiente Python compatível
pode usar `python -m rag_research` diretamente. `rag-research-ep2` fica disponível
após instalar o projeto. A CLI não expõe opção de execução GPT.

O recuperador mantém as interfaces `prepare_targets` e
`retrieve(Text, method=..., count=3)`. Seus exemplos contêm texto original,
rótulo verdadeiro de treino, hash, IDs de linha/grupo e similaridades; são
compatíveis com `rag_cascade.prompt.build_request`. Um planejador futuro pode
injetar o novo recuperador em outro estudo sem editar pacotes históricos congelados.

<a id="screens-and-limitations"></a>

## Critérios de seleção e limitações

Uma proposta de recuperação só justifica um teste GPT pareado e limitado no
futuro se sua configuração escolhida no desenvolvimento melhorar Hit@1 na
auditoria com limite inferior positivo no bootstrap pareado de grupos e não
reduzir a pureza média frente ao híbrido antigo. Relevância, cobertura e
diversidade ainda exigem revisão; um critério offline não prova melhora da
classificação GPT. Um seletor aprendido de intervenção precisa de evidência
positiva de macro-F1 fora da amostra para justificar sua complexidade.
As melhores famílias retrospectivas isoladas não atendem a esse critério.

Nenhuma execução paga, comando de teste final ou promoção de padrão decorre
automaticamente. O relatório registra quais estratégias passam ou falham
nesses critérios.

<a id="measured-evidence-and-verification"></a>

## Evidência medida e verificação

Os artefatos concluídos estão em `results/rag-offline-20261106/`:

- `features/`: quatro representações congeladas verificadas por checksum e
  procedência do codificador/ambiente.
- `retrieval/`: rastreamentos de 171 variantes, dois controles, banco, métricas
  e cópia exata da implementação; todas as 236 sequências híbridas históricas
  são reproduzidas.
- `interventions-final/`: política conjunta aninhada principal, diagnósticos
  por família, reavaliação da confirmação com seleção no piloto e 288 registros
  de ajuste com grupos disjuntos.
- `report/`: Markdown e JSON consolidados; diretórios anteriores de relatórios
  e reavaliações permanecem preservados.
- `independent-verification.json`: métricas recalculadas independentemente para
  341,156 linhas recuperadas e nove políticas de intervenção, hashes das fontes,
  verificações de vetores unitários e separação dos grupos de ajuste.

Os controles finais aleatório/por comprimento têm cosseno lexical máximo com
as consultas abaixo de 0.95. Uma proteção explícita foi adicionada depois da
medição, sem mudar nenhum exemplo selecionado; o código exato medido permanece
copiado. Caches e evidências intermediárias são locais e ignorados pelo Git.
Agregados públicos e relatório Markdown são versionados. A suíte completa de
regressão simulada/offline passou em 229 testes. Casos fixos dos conjuntos
finais foram usados só para detectar regressões de compatibilidade, nunca
para escolher configurações da pesquisa; os comandos não carregam teste1/teste2.
