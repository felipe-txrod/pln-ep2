<a id="four-member-calibrated-soft-voting"></a>

# Votação suave calibrada de quatro membros

Este é o pacote selecionado para SVM, RL de desenvolvimento, NorBERTo original
do EP1 e Qwen na divisão EP1 selecionado. Todos os pesos-base ficam congelados.
Qwen usa `models/qwen/adapter/`; o adaptador histórico é excluído.

As 3,999 linhas reservadas originais mantêm os papéis disjuntos do método de
três membros: 2,027 para calibração e 1,972 para pesos. SVM usa sigmoide;
RL, NorBERTo e Qwen usam temperaturas escalares selecionadas em três folds
agrupados só na calibração. A temperatura Qwen é 1.515480709210816.

Trinta e cinco candidatos distintos do simplex, em incrementos de um quarto,
foram comparados por macro-F1. Pesos selecionados: **SVM 0.75, RL 0,
NorBERTo 0.25, Qwen 0**. Qwen foi avaliado, calibrado e incluído na busca,
mas não contribui na soma escolhida. Seu peso zero resulta da seleção no
conjunto reservado.

Teste1 obteve **447/900 (49.67%)**, macro-F1 **49.58%**. Os 900 rótulos
coincidem com a votação suave preservada de três membros. Nenhum parâmetro
foi atualizado pelo gabarito. Modelo, calibração, composição e identidade
do código ficam em `model.json`; `frozen_manifest.json` fixa o checksum.

A CLI `predict-soft` usa este pacote por padrão. O original de três membros
permanece em `models/ensemble-soft/`. Veja o
[relatório da integração](../../docs/results/ensemble-qwen.md) e o
[guia de uso](../../docs/guides/ensemble-qwen.md).

O conjunto reservado tem exposição histórica e selecionou o checkpoint Qwen.
É validação de desenvolvimento, não teste final inédito. A tarefa não produziu
entrega teste2 nem estabeleceu a ordem oficial teste2 antes de teste1.
