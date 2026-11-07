<a id="decision-retain-weighted-soft-voting-as-the-primary-approach"></a>

# Decisão: manter votação suave ponderada como abordagem principal

Data: 2026-11-06. Estado: aprovado pela equipe do projeto.

[Índice da documentação](../README.md) · [Índice dos resultados](../results/README.md)

<a id="evaluation-of-alternative-ensemble-strategies"></a>

## Avaliação de estratégias alternativas de ensemble

Além da votação suave, o projeto investigou combinações mais complexas,
incluindo stacking e classificação seletiva assistida pelo GPT. Os experimentos
com GPT compararam contextos distintos, incluindo geração aumentada por
recuperação (RAG), com exemplos históricos rotulados de treino fornecidos na
inferência. A recuperação incluiu busca lexical, semântica e híbrida, além de
representações do checkpoint NorBERTo previamente ajustado para a tarefa.

Alguns experimentos exploratórios pareceram promissores, mas os ganhos não
persistiram nas avaliações seguintes. O GPT corrigiu erros do stacking e
introduziu erros em instâncias que o stacking acertava. Mudanças na recuperação,
no roteamento e nas políticas de aceitação não demonstraram melhora consistente.

A votação suave ponderada com **75% SVM e 25% NorBERTo** obteve **49.58% de
macro-F1 no Teste1**, superando as configurações de stacking avaliadas na mesma
base. Os resultados congelados comparados mantêm as 900 instâncias de Teste1:

| Método | Acertos / 900 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Votação suave ponderada | 447 | 49.67% | 49.58% |
| Stacking com SVM, LR e NorBERTo | 442 | 49.11% | 49.03% |
| Stacking com SVM, LR, NorBERTo e Qwen | 440 | 48.89% | 48.80% |

Os pesos da votação suave foram selecionados originalmente em validação.
Tanto o artefato de três modelos quanto o de quatro candidatos com Qwen dão
peso zero à LR; o segundo também dá peso zero ao Qwen e reproduz as mesmas
900 previsões. A decisão mantém a mistura existente de probabilidades
calibradas, em vez de criar uma combinação de escores não calibrados.

<a id="accepted-decision"></a>

## Decisão adotada

Manter **votação suave ponderada como abordagem principal de classificação do
projeto**, com os pesos congelados SVM 0.75 / NorBERTo 0.25. O desempenho medido,
a simplicidade da implementação e a ausência de chamadas adicionais a modelos
de linguagem sustentam essa escolha.

Preservar implementações, artefatos, caches e resultados de stacking, GPT
seletivo e RAG como parte da investigação. Os experimentos mostram que maior
complexidade do ensemble não melhora necessariamente a predição. Permanecem
disponíveis para reprodução e análise; esta decisão não autoriza novos treinos,
chamadas ao GPT ou execuções experimentais.

Este registro documenta a escolha sem alterar código dos classificadores,
pacotes congelados ou padrões dos comandos e sem gerar nova entrega de Teste2.

<a id="interpretation-and-limitations"></a>

## Interpretação e limitações

A conclusão reflete as partições registradas e o caráter exploratório de parte
da investigação. A comparação em Teste1 é retrospectiva: o conjunto já havia
sido examinado, e escolher o método principal após comparar esses resultados
não constitui nova avaliação independente. A vantagem numérica não afirma
superioridade estatisticamente significativa. Resultados de validação em
populações diferentes não são diretamente intercambiáveis com os das 900
linhas de Teste1 acima.

A seleção do checkpoint Qwen já havia usado o conjunto reservado original;
a reavaliação histórica de roteamento/aceitação manteve previsões OOF fixas dos
estágios anteriores e cobertura limitada de respostas GPT registradas. Essas
limitações continuam explícitas nos relatórios de origem. Melhoras na
correspondência de rótulos da recuperação são resultados auxiliares, não
estabelecem melhora do macro-F1 de classificação pelo GPT.

<a id="evidence-preserved"></a>

## Evidências preservadas

- [Protocolo original de votação suave calibrada e resultados de Teste1](../results/ensemble-soft.md).
- [Comparação dos ensembles de três e quatro modelos, pesos e limitações](../results/ensemble-qwen.md).
- [Experimento de stacking + GPT seletivo](../results/stacking-gpt-cascade.md).
- [Piloto exploratório de RAG](../results/rag-cascade-pilot.md) e
  [confirmação posterior](../results/rag-cascade-confirmation.md).
- [Investigação offline de recuperação pelo NorBERTo, MMR e intervenções](../results/rag-offline-research.md).
- Artefatos congelados da votação suave:
  [pacote original de três modelos](../../models/ensemble-soft/README.md) e
  [pacote de quatro candidatos](../../models/ensemble-soft-qwen/README.md).
