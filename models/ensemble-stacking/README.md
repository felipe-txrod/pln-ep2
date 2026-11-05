# Meta-LR congelada sobre probabilidades

Stacking dos mesmos SVM, LR de desenvolvimento e NorBERTo usados pelo soft
voting. Nenhum desses classificadores ou calibradores foi retreinado.

A meta-LR multinomial L2 usa nove probabilidades calibradas na ordem
`svm, rl, norberto` × `c1, c234, c5`. Foi ajustada nas 1.972 linhas fora da
calibração (2.027 linhas). Cinco folds estratificados por grupos e uma grade
fixa de quatro valores C selecionaram **C=0,1**, com a regra de um erro-padrão.
O ajuste não abriu teste1 ou teste2. Os resultados de CV serviram à seleção,
sem constituir estimativa independente de desempenho.

- `model.json`: coeficientes, interceptos, ordem de entradas/classes, código e referência imutável ao modelo soft.
- `frozen_manifest.json`: checksums do modelo e dos arquivos de proveniência.
- `partition.jsonl`: papéis de calibração/meta-treino e fold de validação por linha.
- `cv-search.json`: grade, cinco scores por C, regra e candidato selecionado.
- `meta-oof-predictions.jsonl`: previsões OOF somente das 1.972 linhas de meta-treino.
- `run.json`: partição efetiva, métricas de seleção, auditoria e origem dos escores.

Depende de `models/ensemble-soft/` e dos modelos-base intactos. Os arquivos
anteriores permanecem preservados. O [guia](../../docs/guides/ensemble-stacking.md)
descreve os comandos e as limitações do experimento. Este pacote não promove
automaticamente o stacking aos presets de entrega.

A [avaliação final](../../docs/results/ensemble-stacking.md) registra 442/900
acertos (49,11%), abaixo dos 447/900 do soft voting. O artefato foi preservado
sem ajuste a partir desse resultado.
