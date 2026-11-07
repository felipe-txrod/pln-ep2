<a id="script-entry-points"></a>

# Pontos de entrada dos scripts

[Documentação do projeto](../docs/README.md)

Execute na raiz do EP2. Os scripts são agrupados por modelo ou finalidade.
Caminhos de dados, modelos, caches e resultados são relativos à raiz.

## Entrega final de Teste2

```bash
bash scripts/ensemble/setup-local.sh
bash scripts/ensemble/deliver-test2.sh
```

Sem parâmetros, o setup prepara PyTorch CPU e as dependências neurais, e
a entrega executa em CPU. Para GPU, use `setup-local.sh --with-rocm` e
`deliver-test2.sh --device gpu`. `--minimal` ou `--without-rocm` instala só
a base e mantém PyTorch intacto; `--with-neural` usa um PyTorch existente;
`--with-cpu` explicita o padrão CPU. `--dry-run` apenas mostra o plano.
Os modelos lexicais sempre usam CPU. O segundo comando do exemplo aplica a
votação suave congelada e gera `deliveries/test2-soft-voting.xlsx`.
Repetir o comando recalcula todos os membros e substitui as previsões e o Excel
após validação. `--export-only` substitui apenas o Excel usando previsões concluídas.
O executor é nativo, resolve o caminho do projeto e permite `--check`,
`--export-only` e `--device gpu`. Consulte o
[guia de entrega](../docs/guides/final-delivery.md).

## Relatório do EP2

| Ponto de entrada | Finalidade |
|---|---|
| `report/explain.py --output ARQUIVO.json` | Extrair coeficientes e contribuições locais exatas da RL congelada, sem treinamento |

A interpretação requer as dependências lexicais e os dados/previsões identificados
no [relatório](../docs/report/relatorio-ep2.md).

## NorBERTo

O modelo ativo é o adaptador original da entrega EP1 em `models/norberto/`,
sem alterações. A predição verifica adaptador/base fixados e usa FP32/SDPA e
representação início/fim original de 512 tokens.

| Ponto de entrada | Finalidade |
|---|---|
| `norberto/test1.sh predict` | Predizer com adaptador original EP1 em GPU |
| `norberto/test1.sh compare` | Avaliar previsões salvas do modelo original em CPU |
| `norberto/evaluate-test1.py` | Implementação Python das duas etapas |

```bash
bash scripts/norberto/test1.sh predict \
  --output-dir results/norberto-ep1-test1-new

bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-ep1-test1-new \
  --output-dir results/norberto-ep1-test1-metrics-new
```

Use o [guia teste1](../docs/guides/norberto-test1.md) e o
[pacote original de inferência](../src/norberto_classifier/README.md).
O pacote ativo expõe só inferência. Correções experimentais de perda,
executores de treino, diagnósticos, avaliadores candidate-08 e variante
logística congelada foram desativados em 2026-11-05. As medições estão no
[arquivo](../docs/archive/README.md#norberto-training-investigations).

<a id="exploratory-grid-screening-candidate-08"></a>

### Candidate-08 exploratório da triagem da grade

O [fluxo separado de triagem candidate-08](../docs/guides/norberto-candidate08-screening.md)
carrega `best_adapter` da grade original, não o ajuste final posterior ou um
fold. Preserva o pacote NorBERTo selecionado e a decisão principal de votação
suave. Para executar a comparação retrospectiva pedida de 900 linhas de Teste1:

```bash
bash scripts/experiments/norberto/candidate08-test1.sh \
  --output-dir results/norberto-candidate08-screening-test1
```

## SVM

A investigação SVM está concluída. O
[relatório final](../docs/results/svm-final-optimization-final.md) mantém SVM
original 75% / NorBERTo 25% inalterado. O desenvolvimento congelou A antes de
Teste1; A tem macro-F1 49.5844% e 447/900, contra B=C com 49.5187% e 446/900.
Pesos aninhados continuam 75/25. Sem promoção de padrão ou acesso a Teste2.

| Ponto de entrada | Finalidade |
|---|---|
| `svm/finalize.sh` | Ajuste único, calibração/pesos aninhados, congelamento e avaliação separada de Teste1 executados; retomada valida artefatos |
| `svm/verify-finalization.py` | Auditar independentemente métricas e procedência salvas de desenvolvimento ou predição final |
| `svm/verify-confidence.py` | Reconstruir independentemente os 92 intervalos, sem ajuste/inferência |
| `svm/research-budgeted.sh` | Busca agrupada limitada de duas fases concluída: 96/20 candidatos, 288/100 folds |
| `svm/verify-search.py` | Auditar previsões salvas, cobertura de folds/grupos e decisões de um erro-padrão |
| `svm/research.sh` | Busca original preservada, interrompida após 129 folds por tempo excessivo |

Veja os guias de [finalização/verificação](../docs/guides/svm-finalization.md)
e [busca limitada](../docs/guides/svm-budgeted-optimization.md) para comandos
executados e retomada imutável. Experimentos longos são iniciados manualmente
a pedido do usuário; nenhum ajuste segue os rótulos finais. Artefatos originais
SVM/NorBERTo, resultados históricos e padrões são preservados.

Avalie um artefato novo do EP1 sem substituir o modelo congelado:

```bash
bash scripts/svm/test1.sh \
  --artifact results/svm-full/model.joblib \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/svm-full-test1
```

O comando offline só de CPU valida receita e metadados adjacentes, persiste
previsões só dos textos e depois lê rótulos para `metrics.json`. Inclui
acurácia, macro-F1, precisão, revocação e matriz. Use novo diretório.
`SVM_IMAGE` substitui a imagem existente.

<a id="logistic-regression"></a>

## Regressão logística

```bash
bash scripts/rl/train.sh --dry-run
bash scripts/rl/train.sh
```

O executor offline só de CPU ajusta a receita fixa EP1 nas 16,093 linhas
congeladas de desenvolvimento, excluindo 3,999 reservadas. Cria
`models/rl-development/` para o padrão RL selecionado. A RL histórica completa
é preservada e nunca usada como alternativa automática. Veja o
[guia RL](../src/rl_classifier/README.md).

## Qwen

| Ponto de entrada | Finalidade |
|---|---|
| `qwen/container.sh` | Executar CLI Qwen na imagem ROCm configurada |
| `qwen/evaluate-test1.py` | Avaliar adaptador treinado exportado em teste1 |

[Fluxo Qwen e compatibilidade](../docs/guides/qwen-finetuning.md).

Para retreinar nas mesmas linhas de desenvolvimento/conjunto reservado das
outras bases locais, passe `--ep1-split data/ep1-development-holdout.csv` a
`train` ou `prepare`. O [comando exato](../docs/guides/qwen-finetuning.md#retrain-with-the-exact-ep1-partition)
usa nova saída e verificação automática de compatibilidade. Qwen selecionado
participa de [votação suave e stacking](../docs/guides/ensemble-qwen.md);
presets de maioria mantêm três votantes.

<a id="model-artifacts"></a>

## Artefatos dos modelos

| Ponto de entrada | Finalidade |
|---|---|
| `models/prepare-base.py` | Restaurar/verificar a base pública NorBERTo fixada |
| `models/freeze.py` | Exportar artefatos existentes de inferência sem treino |

```bash
python3 scripts/models/prepare-base.py --verify-only
python3 scripts/models/freeze.py --help
```

[Formatos e restauração de artefatos](../models/README.md).

<a id="diagnostics-and-experiments"></a>

## Diagnósticos e experimentos

`diagnostics/audit-dataset-overlap.py` compara seis planilhas atuais EP1/EP2
em CPU, sem carregar modelos. Reporta identidade de arquivo/conteúdo,
sobreposição exata/normalizada e conflitos de rótulos. Veja a
[auditoria dos dados](../docs/results/dataset-overlap.md) para resultados e reprodução.

Diagnósticos antigos de treino NorBERTo são registros históricos, com fontes
locais em `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

`experiments/gpt/pilot-efficiency.py` contém o piloto pareado anterior GPT/Codex.
Prepara simulação por padrão; `--execute` habilita pedidos remotos. Veja o
[estudo GPT histórico](../docs/archive/gpt/first-result.md).

<a id="compatibility-commands"></a>

## Comandos de compatibilidade

`prepare-model-base.py` encaminha para `models/prepare-base.py`, preservando
as instruções emitidas pelos pacotes congelados. Invólucros removidos de
retreinamento NorBERTo só estão na cópia histórica do código.

Escolha novo destino para cada execução de predição ou comparação.

<a id="weighted-soft-voting-versus-the-ep1-delivery"></a>

## Votação suave ponderada contra a entrega EP1

`bash scripts/rag/offline.sh encode|retrieval|interventions|report` executa
extração NorBERTo congelada, ablações de recuperação, reavaliação GPT em cache
e relatórios. Não expõe execução GPT. Use novos diretórios; veja o
[guia offline](../docs/guides/rag-offline-research.md) e os
[resultados medidos](../docs/results/rag-offline-research.md).

Experimentos de contexto RAG usam `python -m rag_cascade`. Piloto e confirmação
terminaram e selecionaram roteamento zero GPT; testes finais ficam pendentes.
Classificar exige `--execute`; veja o [guia RAG](../docs/guides/rag-stacking.md)
para prévias, recuperação só no treino e proteções da ordem teste2 antes de teste1.

Stacking + GPT seletivo usa `python -m cascade_classifier`, sem novo invólucro.
O [guia da cascata](../docs/guides/stacking-gpt-cascade.md) documenta preparação,
ablação inicial de 10%, extensão em cache, seleção congelada do orçamento,
predição e avaliação final separada.

```bash
PYTHONPATH=src .venv/bin/python scripts/ensemble/compare-ep1.py \
  --run results/ensemble-soft-test1-20261105 \
  --baseline ../EP1/deliveries/test1.xlsx \
  --output-dir results/ensemble-soft-vs-ep1-new
```

A comparação offline em CPU verifica artefato congelado e previsões, alinha a
planilha real EP1 às 900 linhas do gabarito e reporta acurácia, macro-F1,
matrizes e McNemar pareado. Não faz inferência ou ajuste. É comparação isolada;
não certifica a ordem do enunciado teste2 antes de teste1. Veja
[resultados e limites do protocolo](../docs/results/ensemble-soft.md#conferência-contra-a-entrega-efetiva-do-ep1).

Comandos e artefatos atuais de quatro modelos com Qwen estão no
[guia dos ensembles Qwen](../docs/guides/ensemble-qwen.md).

Para stacking, use `python -m ensemble_classifier fit-stacking`,
`predict-stacking` e `evaluate-stacking`, sem executor adicional.
O [guia](../docs/guides/ensemble-stacking.md) dá comandos exatos e uso de escores
em cache; os [resultados](../docs/results/ensemble-stacking.md) incluem F1 por
classe, matrizes e linhas Excel corrigidas/prejudicadas.
