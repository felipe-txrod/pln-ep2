# Ensemble soft congelado

Pesos selecionados por F1-macro em 1.972 linhas do antigo holdout do EP1:
SVM 0,75, LR 0,00, NorBERTo 0,25. A outra parte, 2.027 linhas sem grupos
compartilhados, calibrou as probabilidades. Nenhum modelo-base foi retreinado.
LR é o pacote existente `models/rl-development/`, não a LR histórica ajustada
em toda a base. O peso zero é uma solução válida da grade, não uma omissão.

`model.json` contém pesos, pós-processadores numéricos, identidades dos três
modelos e do código. `frozen_manifest.json` protege esse arquivo. `run.json`
contém diagnósticos e proveniência do ajuste; `weight-search.json` registra
os 16 candidatos; `partition.jsonl` documenta os dois papéis de holdout.
Esses arquivos foram congelados antes da avaliação final nesta rodada.

Inferência: `python -m ensemble_classifier predict-soft --model
models/ensemble-soft/model.json --input ... --output-dir ...`.

O pipeline de votação por maioria não é substituído automaticamente.
Consulte o [protocolo e comandos](../../docs/guides/ensemble-soft.md).
