<a id="experiment-archive"></a>

# Arquivo de experimentos

[Índice da documentação](../README.md)

Os registros históricos são mantidos para rastreabilidade. Use os
[guias atuais](../README.md#guides) para os fluxos suportados e o
[índice dos resultados](../results/README.md) para as comparações atuais.

<a id="gpt-and-codex-studies"></a>

## Estudos com GPT e Codex

| Registro | Conteúdo |
|---|---|
| [Uso e fluxos anteriores](gpt/usage.md) | Instalação original, serviços remotos, desenvolvimento, exportação e comparação estatística |
| [Decisões e requisitos](gpt/decisions.md) | Protocolo inicial do GPT e análise dos dados |
| [Primeiro resultado](gpt/first-result.md) | Análise de erros, custos e constatações iniciais |
| [Recuperação híbrida](gpt/hybrid-retrieval.md) | Método de recuperação, piloto e comparação em 300 linhas |
| [Histórico de verificação](gpt/verification.md) | Verificações anteriores da implementação e marcos dos experimentos |

<a id="norberto-training-investigations"></a>

## Investigações de treinamento do NorBERTo

| Registro | Conteúdo |
|---|---|
| [Avaliação histórica em teste1 com o corpus completo](norberto/legacy-full-test1.md) | Comportamento original de acumulação, 20,092 linhas de treino |
| [Avaliação da receita 2 em teste1 com o corpus completo](norberto/recipe2-full-test1.md) | Perda média corrigida antes da compatibilidade das unidades do otimizador |
| [Avaliação da receita 2 no conjunto reservado de desenvolvimento](norberto/recipe2-development-holdout.md) | Instabilidade reproduzida na partição congelada de desenvolvimento |
| [Investigação da regressão](norberto/regression-investigation.md) | Verificações de dados, tokenizador, argumentos de treinamento e normalização da perda |
| [Auditoria do otimizador e da implementação computacional](norberto/instability-audit.md) | Gradientes em CPU/GPU, diferenças finitas e localização na primeira atenção |

O adaptador original da entrega do EP1, sem alterações, foi selecionado em
2026-11-05. Correções experimentais de treinamento, implementações de diagnóstico,
a variante de cabeça convexa e seus scripts foram removidos do código ativo.
Resultados históricos, adaptadores e caches permanecem intactos. Uma cópia local
do código e os hashes dos arquivos removidos estão em
`results/norberto-rollback-20261105/`.

| Fluxo desativado | Registro |
|---|---|
| Treinamento da receita 3 | [Guia histórico de treinamento](norberto/recipe3-training.md) |
| Correção das unidades do otimizador | [Implementação histórica](norberto/recipe3-optimizer-units.md) |
| Auditorias da amostra compartilhada de treino e do prefixo | [Resultados medidos de diagnóstico](norberto/recipe3-scale-audit.md) |
| Codificador congelado com cabeça logística | [Fluxo histórico e validação cruzada](norberto/frozen-logistic.md) |
| Avaliação do adaptador final candidate-08 | [Avaliação histórica](norberto/candidate08-evaluation.md) |

A [correção da receita 3](norberto/recipe3-optimizer-units.md) e sua
[avaliação no conjunto reservado](../results/norberto-recipe3-holdout.md) descrevem
a implementação posterior compatível com o otimizador. Os agregados públicos
dos experimentos atuais e arquivados permanecem em [`results/data/`](../results/data/).
