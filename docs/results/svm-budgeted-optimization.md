<a id="svm-compute-budget-amendment-completed-investigation"></a>

# Revisão do orçamento computacional do SVM: investigação concluída

O usuário pediu investigação mais rápida quando a busca original se tornou
impraticável. O processo antigo foi interrompido com SIGINT após 43 candidatos
completos / 129 folds. Fontes, planos, configuração, resultados e cache
permanecem inalterados. Nenhuma busca foi reiniciada automaticamente.

<a id="measured-reason-for-the-amendment"></a>

## Motivo medido para a revisão

Os 129 folds salvos continham:

| Normalização dos atributos | Folds concluídos | Tempo mínimo / máximo de ajuste em segundos | Avisos de convergência |
|---|---:|---:|---:|
| L1 | 48 | 0.473 / 12.258 | 0 |
| L2 | 72 | 1.397 / 31.162 | 0 |
| Sem normalização | 9 | 643.418 / 990.669 | 5 |

Os tempos vêm dos relatórios originais e incluem ajuste/cache do pipeline.
Justificam excluir computacionalmente atributos não normalizados, sem provar
inferioridade preditiva. Os desenhos reduzidos foram fixados por tempo/
convergência e pelo pedido de orçamento, sem seleção por desempenho ou rótulos finais.

<a id="new-fixed-budget-and-preserved-evidence"></a>

## Novo orçamento fixo e evidências preservadas

- Fase A: 96 candidatos e 288 folds, incluindo **72 importados** e **216 ajustes
  novos concluídos**. O plano original tinha 768 folds.
- Fase B: até 25 candidatos / 125 folds, contra limite original de 305.
- Quatro processos, uma thread numérica cada, no Ryzen 5 5600 verificado de
  seis núcleos/doze threads, aproximadamente 16 GiB RAM.
- Limite de 120 segundos por subprocesso do solucionador. Tempo esgotado ou
  não convergência tornam o candidato inelegível; nenhuma previsão padrão é inserida.
- Matrizes esparsas de treino/validação são calculadas uma vez por representação
  e fold agrupado, compartilhadas por mapeamento somente de leitura entre
  configurações. TF-IDF nunca aprende com textos de validação.

Os 72 folds importados têm atributos/estimador, ambiente, semente e IDs de
treino/validação idênticos. Cada NPZ foi conferido por checksum, cobertura de
linhas e índices de classes; métricas foram reconstruídas contra os rótulos
originais antes da importação preservando bytes. Os 129 folds históricos
continuam no diretório original, inclusive variantes excluídas.

O novo [guia de execução](../guides/svm-budgeted-optimization.md) descreve grade,
limites e retomada. A preparação está concluída em
`results/svm-final-budgeted-20261106/`: `plan.json`, `reuse.json`, auditoria,
referência e composição de treino copiadas, candidatos, os 288 pares de folds
concluídos e `report.md`. Identidades históricas de entradas/resultados
passaram novamente após a fase A. A fase B depois terminou e passou na
auditoria independente abaixo.

<a id="verified-phase-a-results"></a>

## Resultados verificados da fase A

Os **96 candidatos / 288 folds terminaram**, com **zero tempos esgotados e zero
avisos de convergência**. Todos são elegíveis. O verificador independente
`scripts/svm/verify-search.py` reconstruiu matrizes, acurácia, macro-F1 e
precisão/revocação/F1 por classe dos índices salvos. Conferiu ordem/cobertura
exatas, isolamento de grupos, checksums, médias/dispersão e decisão de um
erro-padrão. Evidência em
`results/svm-final-budgeted-20261106/verification-phase-a.json`; o
[agregado versionado](data/svm-budgeted-phase-a.json) registra os resultados principais.

| Configuração | Macro-F1 médio | Desvio-padrão dos folds | Macro-F1 OOF reunido |
|---|---:|---:|---:|
| Receita original SVM | 43.5384% | 0.7804 pp | 43.5419% |
| Maior média: palavras binárias + char_wb, C=.05, balanceado | 43.6550% | 0.3263 pp | 43.6755% |
| Escolha de um erro-padrão na fase A: char, C=.05, balanceado | 43.5432% | 0.9497 pp | 43.5590% |

O candidato de maior média usa TF-IDF binário, palavras (1,2), min_df=5,
max_df=.98, limite 60000; char_wb (3,6), min_df=3, max_df=.95, limite 50000;
ambos L2, perda hinge quadrática, C=.05 e classes balanceadas. O ganho médio
sobre a receita original é só **0.1166 pontos percentuais**. Diferenças nos
folds: -0.1545, -0.2168 e +0.7211 pp, melhora em só um de três. Previsões
reunidas corrigem 551 erros e prejudicam 537 acertos, saldo 14 em 16093
observações OOF. Diagnósticos posteriores à seleção não demonstram benefício robusto.

A regra fixa de um erro-padrão escolhe a representação mais simples de
caracteres (3,5), limite 50000, min_df=3, max_df=1, TF sublinear, L2, C=.05,
balanceada. Ganho médio 0.0048 pp; diferenças -0.5544, +0.6781, -0.1093 pp.
É escolha preliminar da fase A, não modelo final ajustado/calibrado ou
autorização para substituir o ensemble original.

O intervalo entre os horários do plano e resultado final da fase A é
**294.200 segundos (4 min 54 s)**. Inclui 216 ajustes novos e usa 72 folds
anteriores. As 216 execuções filhas somam 571.135 segundos entre processos
paralelos, máximo de 6.569 por filho; 36 construções de atributos somam
529.241 segundos entre processos. Tempos sobrepostos não devem ser somados
como duração total. O cache esparso ocupa cerca de 4.817 GiB. O intervalo
inclui pausas/retomadas não registradas, não é comparação controlada de
aceleração com a grade original maior e incompleta.

O gerador congelado da fase B produz **20 candidatos / 100 ajustes** em cinco
folds agrupados (semente 143), com referência original e duas representações
distintas de maior média. Ambas combinam palavras+char_wb: binária e sublinear
original. Valores C aproximados .0158114/.05/.158114, com pesos de classe,
verificações hinge e pesos de caracteres .5/1/2 predefinidos. A fase B refina
essas representações ordenadas por média; a escolha preliminar de um erro-padrão
não muda a regra retrospectivamente. A prévia completa está na verificação
e no agregado. Mantêm-se quatro processos e limite 120 segundos. Nenhum
experimento novo foi iniciado nesta auditoria.

<a id="verified-phase-b-results-and-fixed-final-selection"></a>

## Resultados verificados da fase B e seleção final fixa

O usuário concluiu **20 candidatos / 100 folds agrupados**. Os 20 são elegíveis,
com zero tempos esgotados e zero avisos de convergência. O verificador reconstruiu
cada previsão/métrica/grupo, agregado e decisão de um erro-padrão; identidades
de fontes/ambiente/resultados originais passaram. Evidência em
`verification-phase-b.json` na execução local e no
[agregado versionado](data/svm-budgeted-phase-b.json).

| Configuração | Macro-F1 médio | Desvio-padrão amostral dos folds |
|---|---:|---:|
| SVM original nos mesmos cinco folds | 43.7803% | 1.0230 pp |
| Maior média: união sublinear original, peso char .5, balanceado | 44.1344% | 0.8889 pp |
| Escolha fixa de um erro-padrão: união binária, sem pesos de classe | 43.8293% | 0.6075 pp |

A maior média ganha 0.3541 pp e melhora em quatro de cinco folds, com 517 erros
corrigidos e 451 acertos prejudicados OOF (saldo +66). O diagnóstico adaptativo
não substitui a regra fixa de seleção.

A regra de um erro-padrão seleciona `42c22ab6e0398e33b3ea`: C=.05, sem pesos
de classe, hinge quadrática; palavras binárias (1,2), min_df=5, max_df=.98,
limite 60000; char_wb binário (3,6), min_df=3, max_df=.95, limite 50000; ambos
L2, ramos com pesos iguais. Está a até um erro-padrão da maior média (limiar
43.7369%) e usa menos atributos. Ganho médio só 0.0490 pp; diferenças
-0.5167, +0.3444, +0.6292, +0.2782, -0.4902 pp. Corrige 489 e prejudica 491
(saldo -2). É mantido sem substituição retrospectiva pelo candidato de maior média.

A janela dos artefatos da fase B é **424.760 segundos (7 min 05 s)**, incluindo
pausas não registradas, não um teste controlado de hardware. Fases A e B usam
partições agrupadas distintas e não podem ser comparadas diretamente como
resultados pareados.

O [fluxo de finalização](../guides/svm-finalization.md) está concluído em
`results/svm-finalization-20261106/`. Retreinou a escolha de um erro-padrão só
nas 16093 linhas, selecionou sigmoide por calibração aninhada e manteve 75/25
nos cinco folds externos de pesos. A recomendação congelada é A. Teste1
reproduz A histórica exatamente: macro-F1 49.5844%, 447/900, contra 49.5187%,
446/900 de B=C; diferenças pareadas não estabelecem melhora significativa.
O [relatório completo](svm-final-optimization-final.md) e o
[agregado](data/svm-final-optimization.json) incluem a evidência final.

<a id="verification-and-limitations"></a>

## Verificação e limitações

Dezenove verificações unittest específicas passaram em 6.433 segundos, cobrindo
busca original e motor novo. Cobrem orçamento reduzido, equivalência de
previsões/margens (tolerância 1e-12), vocabulário local ao fold, cache, subprocesso
real, limite de tempo sem previsões, paralelismo/retomada em quatro processos,
reutilização e rejeição de adulteração. Uma descoberta de testes ampla demais
também encontrou módulo anterior exclusivo de pytest, sem importação por
pytest ausente no ambiente. A verificação explícita dos dois módulos passou;
sem instalação nem leitura de planilha final. Verificações básicas shell/CLI
e de espaços Git também passaram. Previsões reais salvas da fase A foram
verificadas depois, sem reajustar estimadores. A suíte não foi repetida nesta
auditoria de resultados somente de leitura.

A finalização foi verificada separadamente por 12 testes sintéticos (última
execução 2.599 segundos), além dos 19 da busca. Os 31 combinados passaram
antes da proteção final do checksum do gabarito; os 12 afetados passaram
novamente depois. Cobrem equivalência da calibração portável em escores
reservados, isolamento aninhado de grupos/rótulos, pesos padrão, bootstrap/
contagens, procedência de ajuste/retomada e limite de leitura tardia do gabarito
em 900 linhas sintéticas. Não usaram planilha final real nem novo ajuste real.
Pacotes/configurações imutáveis de busca continuam inalterados e válidos.

Os papéis exatos 16093/2027/1972, grupos originais, sementes, macro-F1 e regra
de um erro-padrão foram mantidos até a finalização. Auditorias independentes
passaram para composição do treino, papéis aninhados, métricas de probabilidade/
classe, contagens finais pareadas e os 92 intervalos. As 900 linhas de Teste1
foram mantidas; sobreposição conhecida e exposição anterior limitam conclusões
retrospectivas. Não houve ajuste após rótulos finais. Nenhum GPT, stacking,
RAG ou treino/inferência de outros modelos-base foi acrescentado; escores
NorBERTo foram reutilizados sem alterações. Teste2 e padrões ativos não mudam.
Mantenha o ensemble original 75/25.
