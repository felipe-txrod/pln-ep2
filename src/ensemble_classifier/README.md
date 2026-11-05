# Ensemble com seleção de classificadores

Combinação offline de previsões já produzidas, sem carregar modelos, retreinar
ou realizar chamadas remotas. A avaliação final usa teste1 rotulado após a
entrega de teste2. O [protocolo completo](../../docs/guides/ensemble.md) descreve as
regras e as extensões planejadas.

## Stacking

Os comandos separados `fit-stacking`, `predict-stacking` e `evaluate-stacking`
treinam somente uma meta-LR nas linhas que não calibraram os modelos,
selecionando regularização por CV estratificada por grupos. Reutilizam os mesmos
calibradores, modelos-base e previsões do soft voting. Consulte o
[guia de stacking](../../docs/guides/ensemble-stacking.md) e o
[artefato congelado](../../models/ensemble-stacking/README.md).

## Weighted soft voting

Os comandos `fit-soft`, `predict-soft` e `evaluate-soft` reutilizam os modelos
congelados com probabilidades alinhadas. O antigo holdout é dividido por grupos
em calibração e seleção de pesos por F1-macro, antes da avaliação nas 900 linhas.
Não alteram os comandos `combine`/`compare-majority` nem seus presets.
Consulte o [guia](../../docs/guides/ensemble-soft.md), o
[modelo congelado](../../models/ensemble-soft/README.md) e os
[resultados com matrizes de confusão](../../docs/results/ensemble-soft.md).

## Local majority ensemble

`configs/ensemble-majority.json` selects the original EP1 NorBERTo,
development-only SVM, and development-only RL. Each model has one vote; two
matching votes decide. When all three labels differ, RL resolves the tie.
The configuration uses only saved local predictions and pinned manifests:

```bash
PYTHONPATH=src python3 -m ensemble_classifier combine \
  --config configs/ensemble-majority.json \
  --input data/test2.xlsx --output-dir results/ensemble-majority-test2-new
```

The corresponding member test2 runs must already exist. To compare the saved
test1 runs without new model inference, use the explicit exploratory command:

```bash
PYTHONPATH=src python3 -m ensemble_classifier compare-majority \
  --config configs/ensemble-majority-test1.json \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/ensemble-majority-comparison-new
```

This compares all nonempty model subsets, trying each configured local model
as tiebreaker, including models outside a selected pair. Singletons are scored
once. All candidate predictions are written before gold is loaded. Row/text
alignment and selected artifact identities are mandatory. Results include
accuracy, macro-F1, confusion matrices, tie counts, pairwise agreement, and
all configurations tied for maximum accuracy.

The comparison writes an exploratory report and never changes delivery
defaults. Official `combine` test1 execution still requires a matching completed
test2 run. Existing results are preserved. The measured best trio achieved
49.22%, versus 49.67% for RL alone; see the [full comparison](../../docs/results/ensemble-majority.md).

## Escolher os membros

`configs/ensemble.json` declares available members, while `models` selects
the active voters. The default is `norberto`, `svm`, and `rl`, using majority
voting and RL arbitration. Qwen is currently excluded from the active members.
`--models` overrides the list for an explicitly selected alternative run:

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

## Selected model variants

NorBERTo and SVM use their original 16,093-row development models. RL defaults
to `models/rl-development/`, fitted on the same partition with all 3,999
holdout rows reserved. Create it once with `bash scripts/rl/train.sh`; a
missing bundle never falls back to the historical full-corpus RL model.
The optional Qwen member now selects `models/qwen/adapter/`, trained on the
same 16,093/3,999 EP1 split; it remains excluded from default membership.

The optional `manifest` member field pins an external model contract. For
`identity_field=artifact_sha256`, the run must match that contract's artifact
hash. For `identity_field=artifact_manifest_sha256`, it must match the selected
manifest file hash. In both cases the selected manifest identity is also
recorded in the ensemble fingerprint. The default configurations use these
checks to reject full-corpus or otherwise different models before combining.

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

Completed `combine` runs write `predictions.jsonl`, `routing.jsonl`, and
`run.json`. Each prediction records its label, decision source, and member
labels in the existing evaluation/export format. The `combine` command does
not generate XLSX or calculate metrics; `compare-majority` provides separate
exploratory scoring. Missing arbitration responses prevent completion.
Existing outputs are preserved; use a new directory.

The majority integration was verified with real saved predictions and an
independent vote/metric checker. New adapters or arbitration configurations
still require their own runtime verification.
