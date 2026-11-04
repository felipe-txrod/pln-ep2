# Ensemble com seleção de classificadores

Combinação offline de previsões já produzidas, sem carregar modelos, retreinar
ou realizar chamadas remotas. A avaliação final usa teste1 rotulado após a
entrega de teste2. O [protocolo completo](../../docs/ensemble.md) descreve as
regras e as extensões planejadas.

## Escolher os membros

`configs/ensemble.json` declara os membros disponíveis e a lista `models`
seleciona quais entram na decisão. O padrão é `qwen` e `norberto`; `svm` e `rl`
também estão configurados. `--models` substitui essa lista para uma execução:

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

## Escolher o árbitro

`tiebreaker` na configuração define o responsável pelo desempate e assume
`gpt` quando omitido. `--tiebreaker` substitui esse valor na CLI, independentemente
de `--models`. Pode receber `gpt`, `qwen`, `norberto`, `svm`, `rl` ou outro
identificador declarado em `members`.

Por exemplo, Qwen e NorBERTo decidem os consensos e o SVM resolve as divergências:

```bash
PYTHONPATH=src python -m ensemble_classifier combine \
  --config configs/ensemble.json --models qwen norberto \
  --tiebreaker svm --tiebreaker-run results/svm-test2 \
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

- `consensus`, padrão: todos os membros selecionados precisam concordar.
  Qualquer divergência exige decisão do árbitro, mesmo com três ou mais membros.
- `majority`, opcional: aceita somente maioria estrita, acima de metade
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

Execuções concluídas gravam `predictions.jsonl`, `routing.jsonl` e `run.json`.
Cada previsão final contém classe, origem da decisão e rótulos locais, no
formato compatível com avaliação e exportação do EP2. O pacote não gera XLSX
nem calcula métricas automaticamente. Não publica uma execução concluída com
desempates faltantes. Saídas existentes são preservadas; use outro diretório.

A integração ainda requer validação em execução. A conferência estática do
código não comprova equivalência numérica dos adapters ou do ensemble.
