# Avaliação de SVM e regressão logística em teste1

[Documentation index](../README.md)

Historical evaluation: the RL result below uses the former full-corpus model
(20,092 training rows). The ensemble default now requires development-only
RL in `models/rl-development/`; these scores do not apply to that new model.
See the [current RL workflow](../../src/rl_classifier/README.md).

Avaliação dos artefatos já treinados do EP1 nas **900 respostas de teste1**,
usando `test1-rotulado-para-teste-estatistico.xlsx` como gabarito. Os modelos
receberam somente os textos de `test1.xlsx`; os rótulos foram lidos após a
inferência para calcular as métricas. Nenhum parâmetro foi ajustado.

| Modelo | Acurácia | Acertos | Macro-F1 | Macro-precision | Macro-recall |
|---|---:|---:|---:|---:|---:|
| LinearSVC, `C=0.05` | **49,44%** | **445/900** | 49,06% | 49,18% | 49,44% |
| Regressão logística, `C=0.25` | **49,33%** | **444/900** | 49,14% | 49,29% | 49,33% |

A diferença de acurácia é de uma resposta, ou 0,11 ponto percentual; isoladamente,
não demonstra superioridade. O SVM foi treinado nas 16.093 respostas de
desenvolvimento do EP1; a logística definitiva foi treinada nas 20.092 respostas.
São avaliações dos artefatos salvos, não de modelos reajustados nesta etapa.

## Concordância entre SVM e regressão logística

Os modelos concordaram em **780 das 900 respostas (86,67%)**. Entre essas
concordâncias, a classe estava correta em **398 casos (51,03%)** e incorreta
em **382 casos (48,97%)**. Acurácia condicional à concordância não equivale à
acurácia do conjunto inteiro.

| Classe prevista por ambos | Concordâncias | Acertos | Acurácia nesses consensos |
|---|---:|---:|---:|
| c1 | 228 | 120 | 52,63% |
| c234 | 234 | 105 | 44,87% |
| c5 | 318 | 173 | 54,40% |
| Total | 780 | 398 | 51,03% |

Nas **120 discordâncias (13,33%)**, somente o SVM acertou 47 vezes, somente
a logística acertou 46 e ambos erraram 27. Para uma composição de somente
SVM e logística com estratégia `consensus`, essas 120 linhas exigiriam árbitro.
Os 382 erros em concordância permaneceriam sem arbitragem nessa regra.

A análise reutilizou exclusivamente as previsões salvas, sem nova inferência.
Hashes, IDs e textos foram conferidos contra a avaliação original. A contagem
de consensos corretos mais acertos exclusivos reproduz os totais individuais:
398 + 47 = 445 para SVM e 398 + 46 = 444 para logística.

## Matrizes de confusão

Linhas representam classes verdadeiras e colunas representam classes previstas.

| SVM | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 142 | 82 | 76 |
| c234 | 82 | 118 | 100 |
| c5 | 54 | 61 | 185 |

| Regressão logística | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 133 | 100 | 67 |
| c234 | 77 | 130 | 93 |
| c5 | 47 | 72 | 181 |

## Execução e reprodução

A execução usou somente CPU, com limite de dois núcleos e 2 GiB de RAM,
sem dispositivos GPU expostos ao contêiner e sem importação de PyTorch,
Transformers, PEFT ou bitsandbytes. As dependências coincidiram com
`requirements-lexical.txt`. O tempo de carregamento e inferência foi de
aproximadamente 2,07 s para SVM e 2,69 s para logística, excluindo avaliação e
inicialização do contêiner; não é um benchmark comparativo controlado.

Os hashes dos artefatos, metadados, código, entradas e previsões, além das
métricas completas por classe, estão no
[agregado de resultados](data/lexical-test1-results.json). As previsões, os relatórios
e o script de avaliação foram preservados em
`results/lexical-test1-cpu-20261104-182046/`, nos subdiretórios `svm` e `rl`.
As 900 linhas foram mantidas, com IDs, ordem e textos conferidos contra o gabarito.

Esses resultados não avaliam o ensemble nem substituem sua entrega de teste2.
Não devem orientar ajuste de pesos, prompts ou regras de desempate usando o
gabarito de teste1. A exposição histórica a esse teste limita sua interpretação
como avaliação independente.
