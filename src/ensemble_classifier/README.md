# Ensemble com seleção de classificadores

O pacote separado [`cascade_classifier`](../../docs/guides/stacking-gpt-cascade.md)
acrescenta decisões seletivas GPT ao stacking congelado de quatro membros.
Usa validação de roteamento OOF aninhada agrupada, cache persistente e
comparações explícitas 5/10/20%. Modelos e comandos existentes continuam disponíveis.

Combinação offline de previsões já produzidas, sem carregar modelos, retreinar
ou realizar chamadas remotas. A avaliação final usa teste1 rotulado após a
entrega de teste2. O [protocolo completo](../../docs/guides/ensemble.md) descreve as
regras e as extensões planejadas.

<a id="current-soft-voting-and-stacking-defaults"></a>

## Padrões atuais de votação suave e stacking

Os dois pipelines de probabilidades selecionam SVM, RL de desenvolvimento,
NorBERTo original EP1 e Qwen na divisão EP1 selecionado. Os artefatos padrão
são `models/ensemble-soft-qwen/model.json` e
`models/ensemble-stacking-qwen/model.json`. Votação suave busca 35 misturas
fixas do simplex em incrementos de um quarto e escolheu Qwen com peso zero.
Stacking usa as 12 probabilidades, incluindo Qwen, com C=0.1 selecionado por
validação agrupada.

As 2,027 linhas de calibração e 1,972 de pesos/meta-modelo permanecem disjuntas
e idênticas às partições originais. Qwen nunca ajustou pesos no conjunto
reservado, mas o usou para selecionar checkpoint. Sua exposição herdada é
registrada. Gabarito Teste1 nunca calibra ou ajusta o meta-modelo. Veja
[comandos atuais](../../docs/guides/ensemble-qwen.md) e
[comparação medida](../../docs/results/ensemble-qwen.md). Artefatos originais
de três modelos e configurações antigas continuam disponíveis explicitamente.

As seções seguintes descrevem o protocolo original. Presets de maioria
continuam independentes e mantêm três votantes.

<a id="original-three-model-stacking"></a>

## Stacking original de três modelos

Os comandos separados `fit-stacking`, `predict-stacking` e `evaluate-stacking`
treinam somente uma meta-LR nas linhas que não calibraram os modelos,
selecionando regularização por CV estratificada por grupos. Reutilizam os mesmos
calibradores, modelos-base e previsões do soft voting. Consulte o
[guia de stacking](../../docs/guides/ensemble-stacking.md) e o
[artefato congelado](../../models/ensemble-stacking/README.md).

<a id="original-three-model-weighted-soft-voting"></a>

## Votação suave ponderada original de três modelos

Os comandos `fit-soft`, `predict-soft` e `evaluate-soft` reutilizam os modelos
congelados com probabilidades alinhadas. O antigo holdout é dividido por grupos
em calibração e seleção de pesos por F1-macro, antes da avaliação nas 900 linhas.
Não alteram os comandos `combine`/`compare-majority` nem seus presets.
Consulte o [guia](../../docs/guides/ensemble-soft.md), o
[modelo congelado](../../models/ensemble-soft/README.md) e os
[resultados com matrizes de confusão](../../docs/results/ensemble-soft.md).

<a id="local-majority-ensemble"></a>

## Ensemble de maioria local

`configs/ensemble-majority.json` seleciona NorBERTo original EP1, SVM e RL
de desenvolvimento. Cada modelo vota uma vez; dois votos iguais decidem.
Se os três rótulos diferem, RL desempata. A configuração usa só previsões
locais salvas e manifestos fixados:

```bash
PYTHONPATH=src python3 -m ensemble_classifier combine \
  --config configs/ensemble-majority.json \
  --input data/test2.xlsx --output-dir results/ensemble-majority-test2-new
```

As execuções correspondentes de teste2 dos membros já devem existir.
Para comparar teste1 salvo sem nova inferência, use o comando exploratório explícito:

```bash
PYTHONPATH=src python3 -m ensemble_classifier compare-majority \
  --config configs/ensemble-majority-test1.json \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/ensemble-majority-comparison-new
```

Compara todos os subconjuntos não vazios, testando cada modelo local configurado
como árbitro, inclusive fora do par selecionado. Modelos isolados são avaliados
uma vez. Todas as previsões são gravadas antes de carregar o gabarito.
Alinhamento de linhas/textos e identidades são obrigatórios. Resultados incluem
acurácia, macro-F1, matrizes, empates, concordância pareada e todas as
configurações empatadas na maior acurácia.

A comparação grava relatório exploratório e nunca muda padrões de entrega.
`combine` oficial de teste1 ainda exige execução compatível concluída de
teste2. Resultados existentes são preservados. O melhor trio medido obteve
49.22%, contra 49.67% da RL; veja a
[comparação completa](../../docs/results/ensemble-majority.md).

## Escolher os membros

`configs/ensemble.json` declara membros disponíveis, enquanto `models`
seleciona votantes ativos. O padrão é `norberto`, `svm` e `rl`, com maioria
e arbitragem RL. Qwen é opcional nesses presets. `--models` substitui a lista
para execução alternativa explicitamente selecionada:

```bash
PYTHONPATH=src python -m ensemble_classifier combine \
  --config configs/ensemble.json \
  --models qwen norberto svm rl \
  --input data/test2.xlsx --output-dir results/ensemble-test2-plan
```

Somente os membros selecionados e o árbitro local configurado são lidos. É possível usar um único modelo
para obter um baseline, sem acionamento do GPT. Os caminhos da configuração
são relativos ao diretório de trabalho; execute a partir da raiz do EP2.

| Identificador | Pacote | Saída usada |
|---|---|---|
| `qwen` | `qwen_classifier` | Rótulo salvo |
| `norberto` | `norberto_classifier` | Rótulo salvo |
| `svm` | `svm_classifier` | Rótulo salvo, sem exigir probabilidades |
| `rl` | `rl_classifier` | Rótulo salvo |

Cada entrada `members` aponta para um diretório com `run.json` concluído e
`predictions.jsonl`. Todos precisam representar a mesma fonte e as mesmas
linhas, com identidade do modelo e hashes válidos. Para Qwen, `adapter` permite
confirmar a ordem das classes em `experiment.json`, pois seu `run.json` de
inferência original não contém esse campo.

Um novo classificador pode entrar com outro identificador e um diretório que
siga esse contrato, sem alterar o combinador. `identity_field` permite indicar
o nome do campo SHA-256 do modelo no manifesto, caso não seja um dos padrões
`artifact_sha256`, `adapter_sha256` ou `artifact_manifest_sha256`.

<a id="selected-model-variants"></a>

## Variantes selecionadas dos modelos

NorBERTo e SVM usam modelos originais de desenvolvimento de 16,093 linhas.
RL usa `models/rl-development/`, na mesma partição, preservando 3,999 reservadas.
Crie uma vez com `bash scripts/rl/train.sh`; pacote ausente nunca recorre à
RL histórica completa. O membro opcional Qwen seleciona `models/qwen/adapter/`,
treinado na mesma divisão EP1 16,093/3,999. É opcional na maioria e incluído
nos pipelines separados de votação suave e stacking.

O campo opcional `manifest` fixa contrato externo do modelo. Com
`identity_field=artifact_sha256`, a execução deve coincidir com o hash do
artefato do contrato. Com `identity_field=artifact_manifest_sha256`, deve
coincidir com o hash do manifesto selecionado. Nos dois casos, a identidade
do manifesto também integra a do ensemble. Configurações padrão rejeitam
modelos completos ou diferentes antes de combinar.

## Escolher o árbitro

`tiebreaker` na configuração define o responsável pelo desempate e assume
`gpt` quando omitido. `--tiebreaker` substitui esse valor na CLI, independentemente
de `--models`. Pode receber `gpt`, `qwen`, `norberto`, `svm`, `rl` ou outro
identificador declarado em `members`.

Por exemplo, Qwen e NorBERTo decidem os consensos e o SVM resolve as divergências:

```bash
PYTHONPATH=src python -m ensemble_classifier combine \
  --config configs/ensemble.json --models qwen norberto \
  --tiebreaker svm --tiebreaker-run results/svm-development-test2 \
  --input data/test2.xlsx --output-dir results/ensemble-test2-svm
```

O árbitro também pode ser um dos membros escolhidos. Nesse caso, sua previsão
é reutilizada nas divergências, sem acrescentar um segundo voto ou executar o
modelo novamente. Se houver outro diretório de arbitragem explícito para esse
membro, ele precisa preservar a mesma identidade do modelo.

Para um árbitro local, `--tiebreaker-run` é opcional: o padrão é o diretório
declarado em `members`. Sua execução deve estar concluída e cobrir a fonte
inteira, mesmo que somente algumas linhas exijam desempate. O artefato e a
configuração do árbitro entram na identidade do ensemble antes da decisão.
Sem esse resultado válido, a combinação falha explicitamente.

Com GPT, `--tiebreaker-run` informa as respostas já geradas e `--gpt-run` é
um alias compatível. As duas opções não podem ser usadas juntas; `--gpt-run`
é recusado para outro árbitro. `gpt_config` só é necessário e lido quando o
árbitro é GPT. Escolher um modelo local não carrega a configuração GPT nem
inicia chamadas remotas.

## Estratégias e respostas pendentes

- `consensus`: todos os membros selecionados precisam concordar.
  Qualquer divergência exige decisão do árbitro, mesmo com três ou mais membros.
- `majority`, default: aceita somente maioria estrita, acima de metade
  dos votos; sem essa maioria, exige decisão do árbitro.

Os nomes anteriores `consensus_gpt` e `majority_gpt` continuam aceitos como
aliases das estratégias; o parâmetro `tiebreaker` define o árbitro em ambos os
casos. GPT não entra em `models`. Nenhuma estratégia converte margens SVM ou
rótulos GPT em probabilidades.

Quando faltam respostas necessárias do árbitro GPT, o comando grava `routing.jsonl`,
`pending-targets.jsonl` e `run.json` com estado `pending_tiebreaks`, encerrando
com código 2. Os alvos pendentes contêm somente identificador e texto, sem
gabarito ou previsões locais. Não há uma previsão final substituta.

As respostas devem ser geradas separadamente pelo pipeline GPT existente ou
importadas de uma execução compatível. Não há executor remoto automático
nesse pacote. Para concluir, repetir a combinação em outro diretório com
`--tiebreaker-run`, reutilizando as previsões locais:

```bash
PYTHONPATH=src python -m ensemble_classifier combine \
  --config configs/ensemble.json --models qwen norberto svm rl \
  --input data/test2.xlsx --tiebreaker-run results/gpt-ensemble-test2 \
  --output-dir results/ensemble-test2
```

O importador verifica configuração, banco e runtime GPT, IDs e textos. Aceita
uma execução que cubra todos os casos encaminhados, inclusive uma execução
completa da mesma fonte; usa somente os rótulos necessários. GPT pode escolher
qualquer uma das três classes.

## Teste1 e saídas

Para teste1, usar outra configuração com os diretórios de previsões dos
mesmos membros para teste1, mantendo estratégia, artefatos e árbitro:

```bash
PYTHONPATH=src python -m ensemble_classifier combine \
  --config configs/ensemble-test1.json --models qwen norberto svm rl \
  --input data/test1-rotulado-para-teste-estatistico.xlsx \
  --phase statistical --test2-run results/ensemble-test2 \
  --gpt-run results/gpt-ensemble-test1 --output-dir results/ensemble-test1
```

`configs/ensemble-test1.json` contém os caminhos de exemplo dessa fase;
ajuste-os aos diretórios efetivamente produzidos. A identidade considera composição, estratégia,
artefatos, modelo de desempate, configuração GPT quando aplicável e código,
mas não os caminhos de saída de cada fase. Trocar os membros ou o árbitro em
teste1 invalida o vínculo à entrega de teste2.

Execuções `combine` concluídas gravam `predictions.jsonl`, `routing.jsonl` e
`run.json`. Cada previsão registra rótulo, fonte da decisão e rótulos dos
membros no formato existente de avaliação/exportação. `combine` não gera
XLSX nem métricas; `compare-majority` oferece avaliação exploratória separada.
Arbitragem ausente impede concluir. Saídas existentes são preservadas; use
novo diretório.

A integração de maioria foi verificada com previsões reais salvas e um
verificador independente de votos/métricas. Adaptadores ou configurações novas
ainda precisam de sua própria verificação de ambiente.
