<a id="four-member-probability-stacking"></a>

# Stacking de probabilidades de quatro membros

Este é o pacote selecionado para SVM congelado, RL de desenvolvimento,
NorBERTo original EP1 e Qwen na divisão EP1. Referencia calibradores e partição
com grupos disjuntos de `models/ensemble-soft-qwen/`.

O meta-modelo logístico multinomial L2 recebe 12 probabilidades calibradas,
em ordem `svm, rl, norberto, qwen` por `c1, c234, c5`. Ajusta somente nas
1,972 linhas de meta-treino; exclui as 2,027 de calibração. Cinco folds
agrupados comparam `C=[0.01, 0.1, 1, 10]`; a regra de um erro-padrão escolhe
**C=0.1**. Os coeficientes aprendidos do Qwen são não nulos.

Teste1 obteve **440/900 (48.89%)**, macro-F1 **48.80%**. O stacking original
de três membros acertou 442/900 (49.11%). Qwen corrigiu 25 erros e prejudicou
27 acertos, perda líquida de dois. Nenhum modelo, parâmetro de calibração ou
regularização foi ajustado por esse resultado.

A CLI `predict-stacking` usa este pacote por padrão. O anterior permanece em
`models/ensemble-stacking/`, com pesos e procedência inalterados. Antes do
uso, o JSON é conferido contra manifesto, modelo suave compartilhado,
partição, ordem de classes/atributos e contrato numérico de probabilidades.

Veja os [resultados](../../docs/results/ensemble-qwen.md) e
[comandos](../../docs/guides/ensemble-qwen.md). A seleção do checkpoint Qwen
usou o conjunto reservado original; essas entradas de desenvolvimento têm
exposição histórica, não são amostra independente de validação de todo o processo.
