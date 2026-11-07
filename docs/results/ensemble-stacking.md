<a id="stacking-versus-weighted-soft-voting-nas-900-linhas"></a>

# Stacking versus votação suave ponderada nas 900 linhas

Este documento preserva protocolo e resultados originais de **três modelos**.
Os padrões atuais de quatro incluem Qwen; veja o
[fluxo atual](../guides/ensemble-qwen.md) e os [resultados da integração](ensemble-qwen.md).

O stacking **não superou o weighted soft voting nesta avaliação**: acertou
442/900, contra 447/900. Acurácia caiu 0,56 ponto percentual e F1-macro caiu
0,55 ponto percentual. McNemar exato bilateral deu p=0,664645, sem evidência
de diferença significativa. Não houve novo ajuste com base no resultado.

Implementação, modelo e protocolo foram congelados no commit `b58ce09`, antes
da avaliação final do stacking, em 05/11/2026. O [guia](../guides/ensemble-stacking.md)
documenta o método e os comandos; o [agregado auditado](data/ensemble-stacking-test1.json)
guarda as métricas, comparações pareadas, hashes e linhas afetadas.

## Modelos e divisão efetiva

São os mesmos modelos do [experimento soft](ensemble-soft.md): SVM original,
LR de desenvolvimento já existente e NorBERTo da entrega do EP1. Nenhum foi
retreinado. A LR histórica ajustada nas 20.092 linhas continua excluída porque
viu o holdout. As transformações de texto permaneceram congeladas.

| Papel | Linhas | Grupos | c1 | c234 | c5 |
|---|---:|---:|---:|---:|---:|
| Treinamento original dos modelos-base | 16.093 | 12.956 | — | — | — |
| Calibração, reutilizada sem reajuste | 2.027 | 1.627 | 635 | 710 | 682 |
| Seleção dos pesos / treino da meta-LR | 1.972 | 1.654 | 622 | 707 | 643 |
| Avaliação final comum | 900 | — | 300 | 300 | 300 |

As 3.999 linhas do holdout mantêm a partição exata do soft voting, sem linhas
ou componentes de similaridade compartilhados entre calibração e meta-treino.
Os calibradores não viram as 1.972 linhas do stacking. Não se recalibraram
probabilidades nas 3.999 linhas depois de escolher C. Usar as mesmas 1.972
observações preserva o orçamento de desenvolvimento dos dois ensembles.

O soft voting continuou com SVM 0,75, LR 0 e NorBERTo 0,25, sem alterar seu
artefato ou suas previsões. O stacking recebe nove probabilidades calibradas,
na ordem `svm, rl, norberto` × `c1, c234, c5`. Modelos, classes, linhas e
hashes de texto são conferidos antes de ler o gabarito. Não há texto bruto
ou rótulo verdadeiro entre as entradas da meta-LR.

## Regularização selecionada no meta-treino

Meta-LR multinomial com intercepto, L2, `lbfgs`, sem pesos de classe ou escala
adicional. Cinco folds estratificados por grupos, semente 42; grupos ficam
inteiros em cada fold. Somente a meta-LR é ajustada na CV, preservando os
modelos-base e os calibradores. Nenhuma linha de calibração entra nesses folds.

| C | F1-macro médio de CV | Erro-padrão |
|---:|---:|---:|
| 0,01 | 39,46% | 1,52 pp |
| **0,1** | **46,08%** | **1,30 pp** |
| 1 | 46,59% | 1,18 pp |
| 10 | 46,20% | 1,40 pp |

A regra de um erro-padrão, definida antes da avaliação, escolheu **C=0,1**:
menor C acima de 45,40% (melhor média menos seu erro-padrão). Favorece maior
regularização entre resultados próximos; não constitui teste de significância.
Depois se ajustou apenas a meta-LR nas 1.972 linhas completas. F1-macro OOF
reunido foi 46,20%; serviu à seleção, não é estimativa independente de teste.
O modelo congelado tem SHA-256
`08b8433ce6407c5f08b4d0a3e94ad31f4dd68eb580dcc8abdf1d0ad500608e97`.

## Comparação final

As mesmas 900 linhas e posições foram usadas em todos os métodos. As previsões
do stacking foram persistidas antes de abrir o gabarito. Medidas em porcentagem;
modelos individuais usam suas decisões nativas, preservando a comparação com
EP1, enquanto ambos os ensembles recebem probabilidades calibradas.

| Método | Acertos | Acurácia | F1-macro | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|---:|---:|---:|
| SVM original | 445 | 49,44 | 49,06 | 49,13 | 42,07 | 55,98 |
| LR de desenvolvimento | 447 | 49,67 | 49,32 | 48,47 | 43,25 | 56,24 |
| NorBERTo original | 432 | 48,00 | 47,86 | 43,70 | 47,33 | 52,56 |
| Weighted soft voting | 447 | 49,67 | 49,58 | 49,38 | 44,63 | 54,75 |
| Stacking | 442 | 49,11 | 49,03 | 46,09 | 46,13 | 54,87 |

O stacking ganha dez acertos sobre NorBERTo, sem superioridade significativa
(p=0,520422). Perde três para SVM (p=0,865374) e cinco para LR (p=0,724854).
Todos os testes são bilaterais e descritivos. Não se selecionaram novos
parâmetros ou métodos com base nesses p-valores.

SVM após sigmoid: 440/900, acurácia 48,89%, F1-macro 48,77%, F1 por classe
48,97% / 42,57% / 54,78%. LR e NorBERTo após temperatura mantiveram rótulos e
métricas nativas. O agregado contém os oito conjuntos de métricas.

### Matrizes de confusão

Linhas = classe verdadeira; colunas = prevista; ordem `c1, c234, c5`.

```text
SVM original          LR de desenvolvimento
[[142,  82,  76],     [[135,  89,  76],
 [ 82, 118, 100],      [ 73, 125, 102],
 [ 54,  61, 185]]      [ 49,  64, 187]]

NorBERTo original     Weighted soft voting
[[118, 113,  69],     [[139,  93,  68],
 [ 68, 155,  77],      [ 74, 135,  91],
 [ 54,  87, 159]]      [ 50,  77, 173]]

Stacking              SVM calibrado
[[118, 124,  58],     [[142,  93,  65],
 [ 56, 155,  89],      [ 83, 126,  91],
 [ 38,  93, 169]]      [ 55,  73, 172]]
```

### Erros corrigidos e acertos prejudicados

| Resultado pareado | Linhas |
|---|---:|
| Ambos acertam | 402 |
| Apenas soft voting acerta (prejudicadas) | 45 |
| Apenas stacking acerta (corrigidas) | 40 |
| Ambos erram | 413 |

Entre os 413 erros comuns, 27 trocaram uma classe errada por outra errada.
No total, 112 previsões mudaram e 788 mantiveram o rótulo. McNemar usa os
discordantes 45 e 40: estatística corrigida 0,188235; p assintótico corrigido
0,664389 e p exato bilateral **0,664645**.

| Classe verdadeira | Corrigidas | Prejudicadas | Mudou, mas ambos erraram | Saldo de acertos |
|---|---:|---:|---:|---:|
| c1 | 0 | 21 | 14 | -21 |
| c234 | 31 | 11 | 0 | +20 |
| c5 | 9 | 13 | 13 | -4 |

O stacking previu `c234` em 372 linhas, contra 305 no soft voting; `c1` caiu
de 263 para 212 e `c5` de 332 para 316. O ganho em c234 não compensou as
perdas em c1 e c5. F1 de c234 subiu 1,50 ponto percentual; o de c1 caiu 3,28.
Isso descreve a mudança observada, sem estabelecer causas semânticas dos erros.

Os índices abaixo são linhas **reais do Excel**, contando o cabeçalho na linha
1. O arquivo local `results/ensemble-stacking-evaluation-20261105/changes-vs-soft.jsonl`
contém texto, gabarito, previsões e categoria. O agregado versionado guarda os
índices, sem duplicar respostas ou gabarito por linha.

```text
Corrigidas (40):
4, 16, 36, 45, 57, 75, 82, 89, 147, 148, 262, 275, 280, 289, 296,
333, 361, 370, 377, 420, 442, 469, 480, 502, 577, 588, 595, 614,
653, 669, 699, 727, 785, 789, 793, 819, 833, 870, 881, 899

Prejudicadas (45):
10, 73, 79, 99, 100, 155, 171, 215, 216, 221, 241, 245, 248, 274,
300, 306, 321, 341, 346, 364, 417, 445, 456, 518, 522, 527, 530,
592, 617, 627, 634, 640, 684, 697, 710, 723, 730, 759, 769, 781,
787, 810, 844, 855, 900
```

## Verificação, preservação e limites

**165 testes passaram e 8 opcionais foram ignorados**. Os novos testes cobrem
ordem das entradas, exportação multinomial, seleção de regularização, CV por
grupos, exclusão das linhas de calibração, proibição de ajuste na avaliação,
leitura tardia do gabarito e rejeição de arquivos alterados.

Uma conferência independente reconstruiu as 900 probabilidades do stacking
por multiplicação escalar dos nove atributos e softmax, sem funções de
inferência do scikit-learn/SciPy: todos os rótulos coincidiram, com diferença
máxima 1,11e-16. Também reconstruiu a mistura anterior e recalculou matrizes,
acurácia, F1 por classe/macro e p-valores por soma binomial. Grupos e folds
persistidos foram conferidos. Não houve chamada remota, treinamento de modelo
base, recalibração ou nova inferência neural. A execução foi curta em CPU.

Artefatos preservados em destinos novos:

- Modelo: `models/ensemble-stacking/`.
- Ajuste: `results/ensemble-stacking-20261105/`.
- Previsões: `results/ensemble-stacking-test1-20261105/`.
- Métricas, auditoria e mudanças: `results/ensemble-stacking-evaluation-20261105/`.
- Agregado: `docs/results/data/ensemble-stacking-test1.json`.

Holdout e teste1 já tiveram consultas históricas no projeto. Esta rodada não
usa os 900 rótulos no ajuste, mas não é um teste historicamente cego. Permanecem
as repetições herdadas da [auditoria](dataset-overlap.md), incluindo 105 textos
normalizados do teste1 presentes no corpus original. Não se filtraram linhas
ou retreinaram modelos para mudar essa condição. Repetições também limitam a
independência assumida pelo teste pareado. Esta avaliação isolada não comprova
a ordem oficial teste2 → teste1 nem constitui entrega de teste2.

A suíte preexistente também abre o gabarito real para conferir os 432 acertos
da entrega EP1. Ela foi executada após o ajuste da meta-LR, antes do commit
de congelamento; os bytes do modelo permaneceram intactos. Essa checagem não
participa do ajuste ou da seleção de C. A leitura tardia assegurada pelo novo
avaliador refere-se ao seu fluxo de execução, sem alegar bloqueio global de
acesso ao arquivo em todo o repositório.
