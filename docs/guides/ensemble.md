# Ensemble extensível

[Índice da documentação](../README.md)

Este guia cobre composição por maioria/consenso e arbitragem. Os padrões atuais
de votação suave e stacking baseados em probabilidades incluem Qwen como quarto
candidato/membro; veja o [fluxo específico](ensemble-qwen.md) e os
[resultados medidos](../results/ensemble-qwen.md).

## Objetivo e escopo

`src/ensemble_classifier/` combina classificadores das classes
`c1`, `c234` e `c5`, preservando suas configurações, pré-processamentos e pesos.
A primeira versão deve combinar arquivos de previsões; a execução dos modelos
entra como uma camada opcional, separada do combinador.

Os classificadores permanecem **congelados e já treinados**. O plano não prevê
retreino, ajuste de pesos, calibração ou treinamento de um metaclassificador.
A avaliação final usa `data/test1-rotulado-para-teste-estatistico.xlsx`;
o holdout do EP1 não participa dessa avaliação.

A primeira implementação fornece seleção de membros, combinação offline,
roteamento de divergências e importação das decisões do árbitro selecionado. O
[guia do pacote](../../src/ensemble_classifier/README.md) apresenta os comandos.
Execução automática dos modelos e estratégias adicionais permanecem extensões.

## Compatibilidade com os classificadores existentes

| Origem | Interface existente | Saída | Integração proposta |
|---|---|---|---|
| NorBERTo | `NorbertoClassifier.predict()` e CLI `predict` | Rótulo e probabilidades por nome de classe | Adapter para a API ou importação de `predictions.jsonl` |
| Qwen | `qwen_classifier.inference.predict()` | Rótulo, probabilidades e logits em listas | Importação de `predictions.jsonl`, usando a ordem de classes do modelo |
| SVM | `SVMClassifier.predict()` e CLI `predict` | Rótulo e margens de decisão | Importação de `predictions.jsonl`; sem probabilidades artificiais |
| Regressão logística | `RLClassifier.predict()` e CLI `predict` | Rótulo, probabilidades e scores | Importação de `predictions.jsonl` |
| GPT/Codex | `gpt_classifier.pipeline.run()` | Rótulo, sem probabilidades de classe | Adapter de desempate e importação de previsões válidas para os casos encaminhados |
| Novo classificador | Contrato comum ou adapter | Rótulo; probabilidades e logits opcionais | Registro explícito de um novo tipo de adapter |

O `validation-predictions.json` original do Qwen contém previsões de classe,
não probabilidades, e a exportação selecionada tinha um erro na ordem das
linhas causado pelo agrupamento por comprimento. Não use esse arquivo original
como entrada por linha dos ensembles. A exportação alinhada recuperada
separadamente serve apenas à análise de rótulos. Votação suave e stacking usam
probabilidades recém-inferidas do adaptador salvo, alinhadas à fonte; veja a
[auditoria da ordem](../results/qwen-ep1-model.md#per-row-export-ordering-audit).

Os presets de maioria usam **NorBERTo + SVM + RL**, com um voto de classe por
modelo. A maioria estrita decide; quando os três rótulos diferem, RL desempata.
Qwen é opcional nesses presets. O consenso anterior Qwen + NorBERTo com
arbitragem GPT permanece uma configuração histórica opcional.

## Seleção de membros

`configs/ensemble.json` declara `qwen`, `norberto`, `svm` e `rl`. A lista `models`
seleciona a composição e pode ser substituída por `--models`, por exemplo
`--models qwen norberto svm` ou `--models svm rl`. Os membros não selecionados
não exigem resultados disponíveis, exceto quando escolhidos como árbitro.
Qualquer identificador
adicional pode apontar para outra execução que siga o contrato de arquivos.

O padrão é `norberto svm rl`, estratégia `majority` e desempate `rl`.
A estratégia alternativa `consensus` encaminha toda divergência ao árbitro
selecionado. Mudar composição ou estratégia cria outra identidade e deve
preceder uma entrega oficial de teste2.

<a id="local-majority-configuration-and-exploratory-comparison"></a>

## Configuração de maioria local e comparação exploratória

A configuração dedicada `configs/ensemble-majority.json` seleciona NorBERTo,
SVM e RL, com um voto cada e RL como árbitro fixo nos empates de três classes.
`configs/ensemble-majority-test1.json` aponta para as previsões salvas de teste1.
Ambas fixam manifestos de modelos treinados somente no desenvolvimento e não
exigem configuração GPT. `ensemble.json` e `ensemble-test1.json` agora usam os
mesmos padrões de maioria local, após a exclusão explícita do Qwen pelo usuário.

Execute a comparação explicitamente exploratória com as previsões existentes:

```bash
PYTHONPATH=src python3 -m ensemble_classifier compare-majority \
  --config configs/ensemble-majority-test1.json \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/ensemble-majority-comparison-new
```

O comando avalia todos os subconjuntos não vazios, com cada modelo local como
árbitro fixo. Modelos isolados aparecem uma vez. O preset de três modelos gera
15 configurações, incluindo referências isoladas e pares. Previsões e
configurações são persistidas antes de ler o gabarito, e todas as linhas devem
alinhar. A acurácia define a classificação, com macro-F1 como critério secundário;
todos os empates de acurácia máxima são listados.

O comando grava evidência exploratória, não uma execução oficial de entrega;
não exige nem gera previsões de teste2. O protocolo oficial `combine` ainda
exige teste2 primeiro. Nenhum resultado é promovido automaticamente. A
[comparação medida](../results/ensemble-majority.md) registra 49.22% para o
melhor trio e 49.67% para RL isolada no teste1 já consultado.

<a id="default-training-partitions-and-artifact-selection"></a>

## Partições padrão de treinamento e seleção de artefatos

| Membro | Linhas selecionadas de treino | Linhas reservadas | Artefato |
|---|---:|---:|---|
| NorBERTo | 16,093 | 3,999 | Unchanged original EP1 delivery adapter |
| SVM | 16,093 | 3,999 | Original fixed-holdout SVM |
| RL | 16,093 | 3,999 | `models/rl-development/`, gerado com `bash scripts/rl/train.sh` |
| Qwen (opcional, excluído) | 16,093 | 3,999 | `models/qwen/adapter/` selecionado, divisão exata do EP1 |

As linhas reservadas servem à avaliação e não entram no treino. O pacote
histórico de RL com corpus completo fica em `models/rl/`, mas a ausência de
`rl-development/` interrompe a inferência em vez de selecionar o modelo antigo.

As duas configurações selecionam `results/svm-development-test1` ou `-test2`
e `results/rl-development-test1` ou `-test2`. NorBERTo mantém os caminhos
`results/norberto-ep1-test1`/`-test2`. Cada entrada também declara `manifest`,
que fixa a identidade do artefato antes de importar previsões. Previsões do
corpus completo são rejeitadas mesmo dentro do diretório configurado. A
verificação vale para membros selecionados e árbitros locais. O conjunto
padrão é NorBERTo + SVM + RL; Qwen fica excluído.

Prepare a RL somente quando necessário:

```bash
bash scripts/rl/train.sh --dry-run
bash scripts/rl/train.sh
```

O primeiro comando faz uma verificação prévia somente de leitura; o segundo
treina com C fixo apenas no desenvolvimento. O [guia da RL](../../src/rl_classifier/README.md)
documenta inferência e artefatos gerados. Não houve treinamento durante a
mudança desses padrões. O usuário o concluiu posteriormente, e o modelo de
desenvolvimento exportado está verificado. Gerar previsões para cada base
continua sendo uma etapa separada.

## Seleção do responsável pelo desempate

`tiebreaker`, ou `--tiebreaker` na CLI, seleciona o árbitro de votos não
resolvidos independentemente de `models`. As configurações distribuídas
selecionam `rl`; omitir o campo mantém a alternativa `gpt` por compatibilidade
histórica. Outros membros configurados podem ser selecionados explicitamente.

Exemplo: `--models qwen norberto --tiebreaker svm` mantém Qwen e NorBERTo
como membros e usa SVM para resolver suas discordâncias. `--tiebreaker-run`
permite informar o diretório de previsões do árbitro; para modelos locais,
o padrão é o caminho configurado em `members`. `--gpt-run` permanece como
alias exclusivo do GPT e não pode ser combinado com `--tiebreaker-run`.

Um membro também pode ser escolhido como árbitro. Sua previsão é reutilizada
como decisão final dos casos encaminhados, sem acrescentar outro voto ou
executar o modelo novamente. Um árbitro local externo aos membros exige uma
execução concluída e alinhada à fonte inteira, com identidade dos artefatos
conferida. Não se exige configuração GPT quando outro árbitro é selecionado.

O nome, os parâmetros e a identidade do árbitro ficam congelados no manifesto;
trocar o árbitro ou seus pesos invalida o vínculo de teste1 à entrega de teste2.
Os aliases antigos `consensus_gpt` e `majority_gpt` continuam aceitos para as
estratégias, sem sobrepor a seleção explícita de `tiebreaker`.

## Arquitetura

```mermaid
flowchart LR
    N[NorBERTo] --> A[Adapters e importadores]
    Q[Qwen] --> A
    S[SVM] --> A
    R[Regressão logística] --> A
    F[Novos classificadores] --> A
    A --> P[Previsões padronizadas e persistidas]
    P --> V[Validação de identidade e alinhamento]
    V --> C{Rótulos locais coincidem?}
    C -->|Sim| L[Manter classe local]
    C -->|Não| G[Árbitro selecionado / cache]
    L --> O[Previsões finais, métricas e exportação]
    G --> O
```

Estrutura implementada do núcleo offline:

```text
src/ensemble_classifier/
  config.py
  files.py
  arbitration.py
  strategies.py
  pipeline.py
  cli.py
```

`config.py` define a seleção e a configuração; `files.py` importa e valida
execuções concluídas; `arbitration.py` resolve a origem das decisões de
desempate. Os módulos de combinação não importam PyTorch,
Transformers, PEFT ou clientes remotos. A adição de um classificador pelo
contrato de arquivos exige somente sua configuração, sem alterar estratégias.

Uma extensão de adapters de execução poderá oferecer `describe()` e `predict(records)`, além
de liberação explícita de recursos quando aplicável. A entrada contém somente
identificador e texto; os rótulos verdadeiros ficam na camada de avaliação.
Cada adapter preserva o pré-processamento do modelo de origem.
A configuração distingue membros de base e árbitro; a inclusão de um novo membro
também exige declarar sua participação na estratégia de combinação.

## Contrato de previsões

Cada execução de um membro possui um manifesto com:

- Versão do esquema e identificador do classificador.
- Revisão e hashes dos artefatos, configuração e implementação.
- Vocabulário de classes e capacidades: rótulos, probabilidades e logits.
- Identidade da fonte, linhas esperadas, protocolo e partições de treinamento.
- Hash do arquivo de previsões e estado de conclusão.

O registro padronizado de inferência deve conter:

```json
{
  "excel_row": 2,
  "text_sha256": "<sha256>",
  "label": "c234",
  "probabilities": {"c1": 0.2, "c234": 0.6, "c5": 0.2}
}
```

`probabilities` e `logits` são opcionais e dependem das capacidades declaradas.
Listas são convertidas para nomes somente quando sua ordem estiver comprovada
pelo artefato. Rótulos de outro vocabulário exigem um mapeamento explícito,
completo e validado; não são convertidos por posição ou aproximação.

Antes de combinar, validar cobertura integral, IDs únicos, ordem, hash de cada
texto, fonte e conjunto de classes nos membros locais. Para GPT, exigir
cobertura integral **somente dos casos encaminhados por divergência**. A lista
desses casos deve ser persistida e conferida; a ausência de GPT nas linhas de
concordância é esperada. Textos repetidos em linhas diferentes
permanecem como registros distintos. Probabilidades devem ser finitas, estar
entre zero e um e somar um dentro de tolerância numérica documentada.

As previsões de validação importadas não devem propagar `gold` ao combinador.
Campos de uso, autenticação ou resposta remota não integram o contrato de
combinação. A procedência técnica fica no manifesto da execução.

## Regra inicial: concordância local e árbitro selecionado

1. Obter os rótulos dos membros selecionados para todas as linhas; as
   probabilidades, quando disponíveis, permanecem associadas ao modelo.
2. Se todos os rótulos coincidirem, usar essa classe sem consultar o árbitro.
3. Se divergirem, usar o rótulo do árbitro selecionado como decisão final.
   Com GPT, encaminhar somente o texto. O árbitro pode retornar qualquer classe do vocabulário
   `c1/c234/c5`, inclusive a terceira; atua como árbitro, não como voto
   permanente em uma maioria de três classificadores.
4. Registrar a origem da decisão, as classes locais, a indicação de
   encaminhamento e, quando aplicável, a classe e a identidade da execução do árbitro.

O request GPT não recebe o gabarito, as classes previstas pelos modelos locais
ou suas probabilidades. Preserva o contrato existente: alvo composto somente
por `resp_text`, com exemplos do treino quando previstos na configuração
congelada. Modelo, prompt, banco de demonstrações, recuperação e parâmetros
permanecem iguais entre teste2 e teste1.

O combinador offline gera a lista de divergências a partir dos arquivos locais.
As previsões do árbitro devem ser importadas ou produzidas em uma etapa explícita de
execução, com cache e orçamento. A combinação final exige todas as respostas
previstas; não inicia chamadas remotas automaticamente.

Probabilidades locais permanecem disponíveis para auditoria, associadas a
cada modelo. A saída do árbitro não recebe probabilidades artificiais. A
decisão final deve usar campos como `label` e `decision_source`, sem apresentar
a média local como distribuição de probabilidades da decisão do árbitro.

## Estratégias alternativas para extensão

As estratégias abaixo permanecem no desenho extensível, mas não definem a
decisão inicial com árbitro selecionável.

### Média de probabilidades

Para membros com probabilidades, calcular
`score(c) = sum(weight[m] * probability[m, c])`, com pesos não negativos e
soma igual a um. A classe final maximiza esse score. A alternativa inicial
documentada usa pesos fixos de 50% para Qwen e 50% para NorBERTo.

Probabilidades de modelos distintos não são automaticamente comparáveis em
calibração. Pesos iguais são um baseline, sem garantia de superar os membros.
Calibração, caso investigada, usa somente dados de desenvolvimento elegíveis;
seus parâmetros são congelados e incorporados à identidade do ensemble.

### Votação por rótulo

Para qualquer conjunto de membros, calcular
`score(c) = sum(weight[m] * indicator(label[m] == c))`.
Essa estratégia inclui modelos que fornecem somente rótulos.

Frequências de votos são scores de votação, não probabilidades calibradas.
Não transformar um rótulo GPT em uma probabilidade artificial para incluí-lo
na média de probabilidades. Uma estratégia que exija probabilidades deve
recusar membros sem essa capacidade.

### Empates nas estratégias alternativas

Definir um membro primário na configuração congelada. Entre classes empatadas,
usar a escolha desse membro quando ela estiver entre as empatadas; caso
contrário, usar uma ordem de classes previamente declarada. Com dois membros
de mesmo peso, a votação por rótulo apenas reproduz o membro primário nas
discordâncias; não oferece uma decisão majoritária adicional.

## Falhas e cobertura

Previsão local ausente ou inválida, ou resposta do árbitro pendente em uma divergência,
impede concluir a combinação. Limite de orçamento, recusa ou falha remota
preserva o estado pendente para retomada; não substitui o árbitro por uma classe
local. Não retirar silenciosamente membros nem renormalizar pesos por linha.
Uma composição ou política alternativa exige outra configuração e identidade.

## Avaliação final em teste1, sem retreino

NorBERTo selecionado e o novo Qwen selecionado usam as mesmas 16,093 linhas
congeladas de desenvolvimento e 3,999 reservadas. A divisão anterior do Qwen,
16,010/4,082, é histórica; entradas opcionais agora apontam ao adaptador salvo
com a divisão do EP1. Qwen permanece excluído do ensemble padrão NorBERTo/SVM/RL.
Seu conjunto reservado selecionou o melhor checkpoint, enquanto os pesos foram
ajustados só no desenvolvimento. SVM e regressão logística agora usam por
padrão as mesmas 16,093 linhas congeladas, reservando 3,999. Os parâmetros
fixos do EP1 permanecem `C=0.05` e `C=0.25`. O artefato de RL de desenvolvimento
concluiu treino e verificação; sua receita fixa pode ser reproduzida com
`bash scripts/rl/train.sh` e novo diretório. A RL histórica de 20,092 linhas
não é uma alternativa automática.

O protocolo inicial é:

1. Fixar artefatos treinados dos membros selecionados, modelo/configuração
   do árbitro, desempate e estratégias. Não selecionar pesos, limiares ou
   prompts por teste1.
2. Conferir procedência: textos, rótulos e grupos de teste1 não podem ter
   entrado no treino ou banco de demonstrações. Identificar e declarar
   sobreposição, mantendo todas as linhas oficiais.
3. Produzir teste2 como entrega, preservando linhas, textos e ordem.
4. Predizer teste1 com a mesma configuração congelada. O classificador recebe
   só `resp_text`; gabarito apenas na avaliação.
5. Medir acurácia, macro-F1, precisão macro, revocação macro, matriz e desempenho
   por classe nas 900 linhas. Comparar membros e ensemble nas mesmas linhas;
   usar a entrega original EP1 para McNemar.
6. Informar casos encaminhados ao árbitro, proporção de empates, consumo e
   tempo, separados da qualidade.

O holdout do EP1 e a validação do Qwen pertencem ao histórico de treinamento e
seleção dos membros; não substituem o teste1 rotulado como avaliação final.
Teste1 já foi analisado em experimentos anteriores. Essa exposição deve ser
declarada no relatório; seus resultados não devem orientar mudanças na
combinação e uma reavaliação não deve ser apresentada como teste independente.

A entrada de um novo membro exige outra configuração e identidade. Ajustar
pesos, calibrar probabilidades ou usar stacking seria uma extensão distinta,
com dados de desenvolvimento adequados e protocolo próprio. Essas etapas não
fazem parte da configuração inicial.

## Execução, recursos e integração com o EP2

- Combinar previsões persistidas deve exigir apenas CPU e não iniciar
  classificadores, instalar pacotes ou realizar chamadas remotas.
- Gerar previsões de modelos locais em processos separados e sequenciais,
  persistindo cada resultado e liberando recursos antes do próximo. Não manter
  Qwen e NorBERTo simultaneamente na GPU de 8 GB.
- Usar cache por identidade do modelo, configuração, fonte e texto. Mudanças
  na estratégia podem reutilizar previsões válidas, sem nova inferência.
  No GPT, a identidade inclui o request, o banco de demonstrações e o runtime;
  somente as divergências exigem resposta. Reutilizar uma resposta não dispensa
  validar a configuração e a procedência.
- Preservar as proteções públicas de inferência do Qwen. O adapter não deve
  contornar verificações chamando funções internas de baixo nível.
- Congelar composição, pesos fixos, desempate, artefatos e configuração do árbitro antes da
  entrega. A identidade do ensemble inclui todos esses elementos.
- Predizer teste2 primeiro. Teste1 exige vínculo à execução concluída de teste2
  com a mesma identidade do ensemble e de todos os seus membros.
- Exportar os campos `excel_row`, `text_sha256` e `label` no formato já aceito
  por `gpt_classifier.delivery` e `gpt_classifier.evaluation`. Manter scores e
  procedência em campos adicionais ou arquivos separados.
- Comparar teste1 com a entrega original do EP1; uma inferência nova do
  NorBERTo não substitui o baseline histórico. Resultados históricos isolados
  de GPT em teste1 servem a diagnóstico, não à seleção de pesos.

## Etapas e critérios de aceite

1. **Núcleo offline:** contratos, importadores, alinhamento, estratégias e
   manifesto. Aceitar arquivos de NorBERTo, Qwen, SVM, regressão logística e GPT sem importar suas
   dependências de execução; produzir e validar a lista de divergências.
2. **Integração dos adapters:** conectar as APIs existentes com carregamento
   sob demanda, preservando configurações e protocolos; sem API de retreino.
3. **Configuração congelada:** fixar os modelos existentes e a regra de
   desempate, conferir a procedência dos dados e preparar as previsões;
   sem seleção de pesos, calibração ou retreino.
4. **Entrega e comparação:** produzir teste2, vincular teste1 e reutilizar
   exportação, métricas e McNemar do projeto.

A verificação futura do núcleo deve usar fixtures, incluindo ordem de classes
diferente, empate, NaN, linhas faltantes, duplicatas, hashes divergentes,
capacidades incompatíveis e composição alterada. Validar a independência de
bibliotecas ML e o bloqueio de teste1 sem teste2 correspondente. A integração
real dos modelos constitui uma etapa separada dessa verificação. Cobrir também
concordância sem uso do árbitro, divergência com decisão do árbitro, escolha
da terceira classe, seleção independente dos membros, reutilização de um
membro como árbitro e retomada de desempates pendentes, sem fallback implícito.
