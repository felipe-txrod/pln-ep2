<a id="documentation"></a>

# Documentação

[README do projeto](../README.md)

Use **Guias** para comandos, **Resultados** para qualidade medida, **Referência**
para o material do enunciado e **Arquivo** para experimentos substituídos.
Todos os exemplos de comandos pressupõem a raiz do EP2, salvo indicação contrária.

Os pontos de entrada executáveis estão no [índice dos scripts](../scripts/README.md).

## Relatório do EP2

O [relatório do Grupo 3](report/relatorio-ep2.md) reúne a trajetória experimental,
o método final de votação suave, McNemar contra a entrega do EP1, interpretação
lexical e análise manual de erros.

<a id="accepted-decisions"></a>

## Decisões adotadas

- [Manter votação suave SVM 75% / NorBERTo 25% como abordagem principal](decisions/ensemble-strategy.md),
  preservando os experimentos com stacking, GPT e RAG e suas limitações.

<a id="guides"></a>

## Guias

| Assunto | Documento |
|---|---|
| Gerar o Excel de Teste2 com um comando, em CPU por padrão e GPU opcional, e validar a entrega 75/25 | [Entrega final](guides/final-delivery.md) |
| Um retreinamento do SVM selecionado, calibração/pesos aninhados, congelamento e avaliação separada em Teste1 | [Finalização do SVM](guides/svm-finalization.md) |
| Busca reduzida atual do SVM, quatro processos, limites rígidos e reutilização verificada | [Otimização do SVM com orçamento computacional](guides/svm-budgeted-optimization.md) |
| Busca do SVM somente no desenvolvimento, referência imutável e execuções longas manuais | [Otimização final do SVM](guides/svm-final-optimization.md) |
| Adaptador original selecionado da entrega do EP1, sem alterações, e inferência offline | [NorBERTo original](../src/norberto_classifier/README.md) |
| Predição com NorBERTo e avaliação separada de teste1 em CPU | [Fluxo de teste1](guides/norberto-test1.md) |
| Avaliação retrospectiva do adaptador original candidate-08 da triagem da grade | [Checkpoint de triagem](guides/norberto-candidate08-screening.md) |
| Instalação do Qwen, compatibilidade ROCm, QLoRA, treinamento e inferência | [Ajuste fino do Qwen](guides/qwen-finetuning.md) |
| Seleção de classificadores, contratos de predição e desempate | [Ensemble](guides/ensemble.md) |
| Votação suave e stacking atuais de quatro modelos com Qwen selecionado | [Ensembles com Qwen](guides/ensemble-qwen.md) |
| Roteamento OOF aninhado, seleção de orçamento e decisões seletivas do GPT | [Cascata stacking + GPT](guides/stacking-gpt-cascade.md) |
| Estudo de recuperação/contexto somente no treino; confirmação selecionou roteamento zero ao GPT | [Estudo de contexto RAG](guides/rag-stacking.md) |
| Recuperação com NorBERTo congelado, ablações MMR e pesquisa de roteamento/aceitação em cache | [Guia da pesquisa offline](guides/rag-offline-research.md) e [resultados](results/rag-offline-research.md) |
| Calibração original de três modelos, pesos e validação cruzada agrupada da meta-RL | [Votação suave](guides/ensemble-soft.md) e [stacking](guides/ensemble-stacking.md) |
| SVM, regressão logística, NorBERTo e GPT congelados | [Artefatos dos modelos](../models/README.md) |

As interfaces de inferência de cada pacote permanecem junto ao código-fonte:
[NorBERTo](../src/norberto_classifier/README.md),
[SVM](../src/svm_classifier/README.md),
[regressão logística](../src/rl_classifier/README.md) e
[ensemble](../src/ensemble_classifier/README.md).

<a id="results"></a>

## Resultados

O **[índice dos resultados](results/README.md)** separa as comparações de teste1
da validação no conjunto reservado e vincula cada relatório ao agregado técnico.
Execuções brutas e previsões individuais permanecem em `results/` na raiz do
projeto; os agregados públicos ficam em [`docs/results/data/`](results/data/).
A [investigação final concluída do SVM/votação suave](results/svm-final-optimization-final.md)
mantém o ensemble original 75/25 após desenvolvimento aninhado e avaliação
pareada congelada no Teste1; seu [agregado](results/data/svm-final-optimization.json)
preserva as auditorias independentes.

<a id="reference"></a>

## Referência

| Documento | Finalidade |
|---|---|
| [Enunciado](reference/assignment.md) | Tarefa oficial, critérios de avaliação e entregáveis |
| [Modelo de relatório](reference/report-template.md) | Estrutura exigida do relatório |

<a id="archive"></a>

## Arquivo

O **[índice do arquivo](archive/README.md)** reúne estudos anteriores com GPT/Codex,
receitas desativadas do NorBERTo, variantes logísticas, investigações de regressão
e diagnósticos. Esses registros preservam os resultados e as limitações originais;
seus comandos são referências históricas, não o fluxo padrão.
