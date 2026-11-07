<a id="measured-results"></a>

# Resultados medidos

[Índice da documentação](../README.md)

Compare resultados somente na mesma base e preserve a receita de treino e a
partição registradas em cada relatório. A referência original do NorBERTo do
EP1 tem acurácia de 48.00% em teste1 e 45.31% no conjunto reservado congelado.

A [decisão adotada pelo projeto](../decisions/ensemble-strategy.md) mantém a
votação suave ponderada (SVM 75%, NorBERTo 25%) como abordagem principal:
macro-F1 de Teste1 49.58%, acurácia 49.67% (447/900). Estudos de stacking, GPT
e RAG ficam preservados abaixo, com suas limitações exploratórias e de partição.

<a id="final-svm-optimization-completed-retain-the-original-ensemble"></a>

## Otimização final do SVM concluída: manter o ensemble original

O [relatório final completo](svm-final-optimization-final.md) mantém A, o ensemble
original SVM 75% / NorBERTo 25% inalterado. O desenvolvimento congelou essa
recomendação antes de Teste1. Nas mesmas 900 linhas finais, A acerta 447 e
obtém macro-F1 49.5844%; B/C otimizados acertam 446 e obtêm 49.5187%. A seleção
aninhada mantém 75/25 nos cinco folds externos, tornando B e C idênticos.
Corrigem 21 erros de A e prejudicam 22 acertos; o intervalo pareado de macro-F1
inclui zero. Não se estabelece melhora consistente nem diferença significativa.

As buscas agrupadas terminaram: 96 candidatos / 288 folds na fase A e 20 / 100
na fase B, sem tempo esgotado ou aviso de convergência. Um SVM selecionado foi
retreinado, a calibração aninhada escolheu sigmoide, e todas as métricas finais
e 92 intervalos passaram na reconstrução independente. Teste1 é retrospectivo
e mantém todas as linhas apesar da sobreposição histórica. Teste2, artefatos
originais e padrões ativos permanecem inalterados.

Veja o [agregado](data/svm-final-optimization.json), a
[revisão do orçamento e evidências da validação cruzada](svm-budgeted-optimization.md),
o [registro original de preparação](svm-final-optimization.md) e o
[guia de execução/verificação](../guides/svm-finalization.md).

<a id="offline-retrieval-and-selective-acceptance-no-promotion"></a>

## Recuperação offline e aceitação seletiva: sem promoção

A [investigação offline](rag-offline-research.md) compara 171 variantes de
recuperação e duas referências em 1,972 linhas, reutilizando NorBERTo congelado,
E5 e respostas GPT históricas. A fusão tripla escolhida no desenvolvimento
não melhorou Hit@1 na auditoria: 37.27% contra 38.48% do híbrido atual.
Ablações de diversidade mostram o compromisso pureza/cobertura; ampliar o
conjunto de candidatos não traz benefício medido. O seletor conjunto aninhado
de intervenções obteve macro-F1 47.19%, contra 47.30% do stacking nas mesmas
1,972 linhas. A reavaliação tem cobertura registrada limitada e previsões OOF
anteriores fixas; resultados por família não estabelecem generalização de
ponta a ponta. Nenhum candidato passou no critério registrado para teste pago.
Zero novas chamadas GPT, sem reajuste dos modelos-base ou mudança de padrões.
O [agregado](data/rag-offline-research.json) e o
[guia de reprodução](../guides/rag-offline-research.md) preservam a evidência.

<a id="rag-context-confirmation-zero-gpt-routing-selected"></a>

## Confirmação de contexto RAG: roteamento zero ao GPT selecionado

A [confirmação concluída](rag-cascade-confirmation.md) avaliou as 1,884 linhas
OOF em quatro contextos e roteamento 0/5/10%. Stacking original venceu com
910 acertos, acurácia 48.30% e macro-F1 48.08%. Todos os candidatos com GPT
perderam macro-F1; o melhor usou previsões locais a 5%, com 47.84%. O ganho
do piloto híbrido RAG/local não persistiu. Houve 94 chamadas válidas, sem
falhas/fallbacks, com consumo conhecido estimado de USD 1.6595524 equivalente
à API. A avaliação não executou testes finais, mudanças em produção nem novas
chamadas remotas. O [agregado](data/rag-cascade-confirmation.json) inclui todas
as métricas e auditorias.

<a id="rag-context-pilot-exploratory-gain-not-confirmed"></a>

## Piloto de contexto RAG: ganho exploratório não confirmado

O [piloto concluído de 48 alvos](rag-cascade-pilot.md) comparou 14 braços com
recuperação TF-IDF/E5/híbrida somente no treino e três/cinco demonstrações.
A regra pareada escolheu recuperação híbrida com três exemplos. O braço
combinado com previsões locais acertou 19/48, macro-F1 36.90%, contra 16/48
e 27.59% do stacking nos mesmos casos difíceis: oito erros corrigidos e cinco
acertos prejudicados. O pequeno ganho exploratório não foi estatisticamente
significativo e não persistiu na confirmação posterior de 1,884 linhas.
O piloto usou 84 chamadas válidas em lotes e três tentativas falhas arquivadas,
com consumo conhecido estimado de USD 2.022988 equivalente à API. Nenhum teste
final ou padrão de produção mudou. O [registro original de preparação](rag-cascade-preparation.md),
[agregado](data/rag-cascade-pilot.json) e [guia](../guides/rag-stacking.md)
preservam marcos, contabilização da recuperação e protocolo teste2 antes de teste1.

<a id="stacking--selective-gpt-no-measured-improvement"></a>

## Stacking + GPT seletivo: sem melhora medida

O [experimento concluído da cascata](stacking-gpt-cascade.md) usa previsões OOF
agrupadas aninhadas 5x5 para escolher roteamento, contexto e orçamento, sem
previsões do meta-modelo nos próprios dados de ajuste. A ablação de 10% preferiu
previsões locais ao texto isolado, mas nenhuma melhorou stacking. A validação
selecionou **zero chamadas GPT** por macro-F1. As configurações congeladas
5/10/20% acertaram 435/434/436 em teste1, abaixo de 440/900 do stacking.
Macro-F1 foi 48.18%, 48.05%, 48.18%, contra 48.80%. Os 900 rótulos só foram
avaliados após a predição; resultados finais não mudaram a configuração.

O experimento usou 97 pedidos Codex válidos em lotes, incluindo 23 na comparação
final, sem falhas. Os tokens observados correspondem a USD 1.672858 pelos
preços padrão equivalentes à API; não é fatura ChatGPT. O
[pacote congelado](../../models/stacking-gpt-cascade/README.md) preserva orçamento
zero, comparações positivas opcionais, identidades e auditorias. Os 201 testes
passaram; cálculos independentes verificaram roteamento e métricas.

<a id="qwen-added-to-soft-voting-and-stacking"></a>

## Qwen adicionado à votação suave e ao stacking

A [integração de quatro modelos](ensemble-qwen.md) preserva a divisão original
do EP1 e as 2,027 linhas de calibração / 1,972 de meta-treino disjuntas.
Votação suave selecionou SVM 75%, RL 0%, NorBERTo 25%, Qwen 0% entre 35
candidatos fixos e reproduziu **447/900 (49.67%)**. Stacking usa doze
probabilidades e acertou **440/900 (48.89%)**, contra 442/900 do controle de
três modelos. Qwen isolado acertou **414/900 (46.00%)**. Nenhum parâmetro foi
ajustado em teste1. Os novos pacotes padrão coexistem com os originais; os
presets separados de maioria mantêm três votantes. Cálculos independentes
verificaram escores, pesos, previsões e métricas. O checkpoint Qwen usou o
conjunto reservado, mantendo a limitação herdada de exposição na validação.

<a id="selected-qwen-frozen-ep1-partition"></a>

## Qwen selecionado: partição congelada do EP1

A execução 16,093/3,999 concluída é o padrão Qwen em `models/qwen/adapter/`.
Concluiu três épocas e 3,018 atualizações; validação: **1,791/3,999 (44.79%)**,
macro-F1 **44.44%**. Todas as linhas, checkpoint, checksums e 225 tensores finitos
foram conferidos, seguidos de recarga real em GPU com textos sintéticos.
A exportação original por linha tinha erro de ordem pelo agrupamento por
comprimento; a recuperação exata do estado aleatório do checkpoint reproduziu
as métricas agregadas e salvou registros alinhados separadamente. A inferência
posterior de teste1 acertou 414/900; Qwen participa de votação suave e stacking.
Veja [modelo e auditoria](qwen-ep1-model.md).

<a id="original-three-model-stacking-versus-weighted-soft-voting"></a>

## Stacking original de três modelos contra votação suave ponderada

A [comparação de stacking](ensemble-stacking.md) reutiliza modelos, calibradores
e 1,972 linhas de desenvolvimento da seleção dos pesos suaves. Cinco folds
agrupados estratificados selecionaram C=0.1 da meta-RL entre quatro candidatos
fixos pela regra de um erro-padrão. Acertou **442/900 (49.11%)**, macro-F1
**49.03%**, abaixo da votação suave: 447/900 (49.67%), macro-F1 49.58%.
Corrigiu 40 erros e prejudicou 45 acertos; McNemar exato p=0.664645.
O relatório inclui F1 por classe, matrizes e linhas Excel afetadas. Nenhum
modelo-base foi reajustado, não houve ajuste pelos rótulos finais e artefatos
anteriores foram preservados. A comparação permanece exploratória, com
exposição histórica.

<a id="local-majority-ensemble-test1-comparison"></a>

## Ensemble de maioria local: comparação em teste1

A [comparação de votação suave ponderada](ensemble-soft.md) é um método separado
congelado: 2,027 linhas reservadas calibram e 1,972 escolhem pesos por macro-F1
entre 16 candidatos. Obtém 447/900 (49.67%) e macro-F1 49.58%, com SVM 0.75,
LR 0.00 e NorBERTo 0.25. O relatório inclui matrizes individuais, efeitos da
calibração e sobreposição herdada entre bases. A
[comparação direta com a entrega efetiva do EP1](ensemble-soft.md#conferência-contra-a-entrega-efetiva-do-ep1)
confirma que os 900 rótulos originais coincidem com as previsões salvas NorBERTo.
McNemar exato bilateral dá p=0.303327: o ganho numérico não é significativo
a 0.05. A ordem oficial teste2 antes de teste1 não foi estabelecida.

Os presets de maioria usam NorBERTo + SVM + RL, maioria estrita e arbitragem
RL; os padrões separados de votação suave/stacking agora também avaliam Qwen.
A [auditoria de isolamento e resultados do conjunto reservado](ensemble-holdout.md)
confirma que as 3,999 linhas e seus grupos ficaram fora do ajuste dos modelos
selecionados. NorBERTo acerta 1,812/3,999 (45.31%); RL e o melhor trio,
1,803/3,999 (45.09%); SVM, 1,792/3,999 (44.81%).

A melhor maioria NorBERTo + SVM + RL usou RL no desempate triplo: **443/900
(49.22%)**, abaixo de RL isolada com 447/900 (49.67%). Os três usam artefatos
somente de desenvolvimento, preservando o conjunto reservado. Quinze
configurações fixas foram comparadas em estudo explicitamente exploratório,
gerando seis sequências distintas. Veja [resultados e comando de reprodução](ensemble-majority.md)
e [agregado verificado independentemente](data/ensemble-majority-test1.json).

<a id="selected-norberto-unchanged-ep1-delivery"></a>

## NorBERTo selecionado: entrega original do EP1 sem alterações

O artefato selecionado é `models/norberto/final_model/`, SHA-256
`d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385`.
Mantém adaptador, cabeça treinada, tokenizador, revisão-base, representação
início/fim de 512 tokens, FP32/SDPA e parâmetros originais. Seu treino final
usou duas épocas em 16,093 linhas de desenvolvimento. Nenhuma correção de
perda, novo treino de adaptador ou substituição por cabeça convexa está ativa.

A restauração foi verificada predizendo novamente os 900 textos de teste1:
todos os rótulos coincidem com a entrega EP1, com 432 acertos (48.00%) e
47.86% de macro-F1. A diferença máxima de probabilidade contra o CSV original
é 8.60e-8. Um leitor ZIP/XML independente confirmou textos, rótulos, IDs,
métricas e identidades originais de adaptador/tokenizador/configuração.
A acurácia histórica no conjunto reservado continua 45.31%; não foi necessária
nova inferência nele.

A limpeza desativou 34 arquivos experimentais, preservando uma cópia local do
código e todos os diretórios de resultados/modelos/caches. Os comandos de
teste1 e o ensemble agora selecionam explicitamente o original. A suíte
disponível passou em 115 testes; dois módulos opcionais exigem o SDK `openai`
indisponível. Nenhum modelo foi retreinado nem predição/entrega de teste2 gerada.
Veja o [registro da restauração](data/norberto-ep1-restoration.json) e o
[fluxo atual](../guides/norberto-test1.md).

<a id="ensemble-defaults-development-models-with-holdout-reserved"></a>

## Padrões do ensemble: modelos de desenvolvimento com conjunto reservado preservado

NorBERTo e SVM selecionam os artefatos originais de desenvolvimento de 16,093
linhas. RL agora usa por padrão `models/rl-development/`. O usuário concluiu
o treino: C=0.25 fixo, 38 iterações, 25.08 segundos, excluindo as 3,999 linhas
reservadas. A verificação posterior conferiu as 16,093 identidades de treino,
hashes, ausência de sobreposição de grupos, carregamento padrão e inferência
sintética. Veja a [verificação da RL](data/rl-development-verification.json).
A RL histórica de 20,092 linhas fica arquivada em `models/rl/`, sem uso como
alternativa automática. A avaliação posterior está abaixo.

As duas configurações fixam manifestos NorBERTo/SVM/RL e rejeitam previsões de
artefatos diferentes. O Qwen histórico usou divisão 16,010/4,082; o pacote
novo selecionado usa EP1 16,093/3,999 e permanece excluído. Depois, o usuário
selecionou maioria NorBERTo + SVM + RL com arbitragem RL como padrão. A suíte
disponível passou em 121 testes, incluindo ajuste logístico sintético que
verifica exclusão de vocabulário exclusivo do conjunto reservado e recarga do
pacote exportado. Previsões SVM de teste1 foram regeneradas só para registrar
o ambiente atual: os 900 rótulos coincidem com o modelo de desenvolvimento
anterior. Não houve nova avaliação, inferência teste2 ou solicitação remota.

Veja o [registro da configuração](data/ensemble-development-defaults.json) e
o [guia de comandos da RL](../../src/rl_classifier/README.md).

<a id="development-only-rl-holdout-and-test1"></a>

## RL somente de desenvolvimento: conjunto reservado e teste1

O modelo logístico salvo de C=0.25 foi avaliado sem reajuste ou seleção de
parâmetros. O treino usou 16,093 linhas; as 3,999 reservadas e seus componentes
de similaridade ficaram fora do ajuste.

| Base | Acertos | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Conjunto reservado congelado do EP1 | 1,803 / 3,999 | 45.09% | 44.78% |
| Teste1 contra a planilha rotulada | 447 / 900 | 49.67% | 49.32% |

As previsões dos textos de ambas as bases foram salvas antes do gabarito.
Um leitor ZIP/XML independente verificou os 4,899 IDs, textos e rótulos,
recalculou manualmente matrizes e métricas e reconstruiu probabilidades dos
escores de decisão (erro máximo 2.23e-16). Cada rótulo reservado coincide com
a comparação logística lexical histórica do EP1. Todos os hashes permaneceram
inalterados. Teste1 ganha três acertos frente à RL histórica completa (444/900);
essa diferença descritiva não estabelece vantagem geral de menos dados de treino.

Resultados brutos, previsões, comparações por linha e cópias do avaliador/
verificador estão em `results/rl-development-evaluation-20261105/`. A execução
selecionada de teste1 é agora `results/rl-development-test1`, com identidade
do modelo verificada. Não houve predição teste2, chamada remota ou novo treino.
Conjunto reservado e teste1 já foram consultados; a avaliação é descritiva.
Veja o [agregado verificado](data/rl-development-evaluation.json).

<a id="test1-900-labeled-responses"></a>

## Teste1: 900 respostas rotuladas

| Modelo | Acurácia | Relatório |
|---|---:|---|
| SVM, retreinamento completo | 50.11% | [Avaliações verificadas no corpus completo](data/full-test1-verification.json) |
| RL, somente desenvolvimento com conjunto reservado preservado | 49.67% | [Avaliação verificada da RL](data/rl-development-evaluation.json) |
| SVM congelado do EP1 | 49.44% | [Avaliação dos classificadores em CPU](lexical-test1.md) |
| Regressão logística histórica do EP1, corpus completo | 49.33% | [Avaliação dos classificadores em CPU](lexical-test1.md) |
| NorBERTo original do EP1 | 48.00% | Referência documentada na [comparação Qwen](qwen-test1.md) |
| NorBERTo receita 3, treinado no desenvolvimento | 46.89% | [Métricas salvas de teste1](data/norberto-recipe3-test1-summary.json) |
| Qwen3-1.7B QLoRA | 46.67% | [Avaliação Qwen](qwen-test1.md) |
| NorBERTo receita 3, retreinamento completo | 38.56% | [Avaliações verificadas no corpus completo](data/full-test1-verification.json) |
| Vencedor da validação cruzada do EP1 candidate-08, adaptador final de desenvolvimento | 34.78% | [Avaliação do adaptador recarregado](data/norberto-ep1-candidate08-evaluation.json) |

A linha da receita 3 treinada no desenvolvimento registra 422/900 acertos
(46.8889%, arredondado a 46.89%), lidos da comparação concluída existente.
Seu macro-F1 é 46.85%. Nenhuma inferência nova foi feita para montar o índice.

A tabela lista resultados com agregados versionados. Novos adaptadores NorBERTo
são modelos distintos. Os comandos separados de predição/avaliação de teste1
estão no [guia](../guides/norberto-test1.md); resultado do conjunto reservado
não deve substituir resultado de teste1.

<a id="historical-norberto-experiments"></a>

## Experimentos históricos do NorBERTo

Os experimentos a seguir ficam preservados como evidência. Suas correções,
retreinamentos e fluxos de cabeça logística foram desativados e não são usados
pelo modelo EP1 selecionado.

<a id="frozen-norberto-logistic-ep1-cross-validation"></a>

### NorBERTo congelado com regressão logística: validação cruzada do EP1

O codificador público foi congelado em avaliação, e seus vetores de documentos
de 768 dimensões, agregados por média mascarada, foram normalizados em L2.
Só a cabeça logística foi ajustada. As 16,093 linhas usam **exatamente as
partições internas e externas originais do NorBERTo no EP1**; as 3,999
reservadas foram excluídas antes da tokenização. O protocolo tem três folds
externos e o primeiro conjunto interno reservado entre cinco divisões agrupadas,
sem reajuste no treino externo completo.

| Modelo nos mesmos folds externos de desenvolvimento | Acertos / 16,093 | Acurácia reunida | Macro-F1 reunido |
|---|---:|---:|---:|
| NorBERTo público congelado + regressão logística | 6,850 | 42.57% | 42.20% |
| NorBERTo LoRA original do EP1 | 6,995 | 43.47% | 43.22% |
| Logística lexical EP1, mesmas linhas de treino interno | 7,078 | 43.98% | 43.51% |
| Logística lexical EP1, treino externo completo | 7,080 | 43.99% | 43.54% |

Controles históricos reutilizam previsões OOF salvas após conferir IDs, rótulos,
grupos e folds. Não foram reexecutados. O novo candidato perde 145 acertos
(-0.90 pontos percentuais) frente ao LoRA. O intervalo descritivo do bootstrap
pareado por grupos de similaridade é [-1.84, +0.01] pontos; não estabelece
equivalência. Folds fixos e conjunto reservado historicamente reutilizado
tornam o experimento exploratório.

| Fold | C selecionado internamente | Acurácia externa | Macro-F1 externo |
|---|---:|---:|---:|
| 1 | 1 | 42.97% | 42.24% |
| 2 | 10 | 42.81% | 42.64% |
| 3 | 10 | 41.95% | 41.48% |

Os nove ajustes L-BFGS predefinidos convergiram em 46–372 iterações. O
classificador final de desenvolvimento usa a mediana predefinida C=10 e
convergiu em 134. Resultados externos não selecionaram esse parâmetro.
O codificador tinha zero parâmetros treináveis, zero atualizações e SHA-256
do estado carregado idêntico antes/depois da extração. Isso remove sua
otimização do pipeline, sem garantir acurácia monotônica com mais dados.

A extração codificou 14,469 textos únicos em 252.05 segundos, com pico de VRAM
alocada 1,300.62 MiB. A validação cruzada da cabeça levou 34.67 segundos e o
ajuste final 5.38, em CPU. A cabeça NPZ segura e o manifesto estão em
`results/norberto-frozen-logistic-ep1-20261105/model/`; modelos originais e
configuração do ensemble foram preservados.

Um leitor ZIP/XML independente verificou todos os textos-fonte OOF, rótulos,
IDs e métricas. A reconstrução direta das probabilidades das três cabeças
exportadas coincidiu exatamente com as OOF salvas. Inferência sem cache,
recarga segura, ordem do cache e CLI passaram com três respostas sintéticas.
A avaliação posterior do conjunto reservado está abaixo. O novo classificador
não foi avaliado em teste1 ou teste2.

A suíte de regressão disponível passou em 153 testes, incluindo o novo pipeline;
dois módulos não relacionados não puderam ser coletados por falta do SDK
opcional `openai` na imagem ROCm. Um teste de referência existente verificou
separadamente as previsões históricas EP1 de teste1; esses rótulos não entraram
na seleção, ajuste ou avaliação deste modelo.

Veja o [agregado técnico](data/norberto-frozen-logistic-ep1.json) e o
[guia de reprodução](../archive/norberto/frozen-logistic.md).

<a id="frozen-norberto-logistic-unchanged-classifier-on-ep1-holdout"></a>

### NorBERTo congelado com regressão logística: classificador inalterado no conjunto reservado do EP1

O classificador salvo de desenvolvimento C=10 foi avaliado nas 3,999 linhas
reservadas, sem reajuste, seleção de parâmetros ou carregamento de adaptador.
As 16,093 linhas e 12,956 componentes de desenvolvimento são disjuntos das
3,999 linhas e 3,281 componentes reservados. Hashes de fonte/divisão coincidem
com o treino. Cada previsão foi salva antes de construir registros rotulados
para avaliação. Hashes do classificador, manifesto e codificador não mudaram.

| Modelo nas mesmas linhas reservadas | Acertos / 3,999 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| NorBERTo público congelado + regressão logística | 1,694 | 42.36% | 42.08% |
| NorBERTo LoRA original do EP1 | 1,812 | 45.31% | 45.31% |
| Regressão logística lexical EP1 | 1,803 | 45.09% | 44.78% |
| SVM lexical congelado do EP1 | 1,792 | 44.81% | 44.47% |
| NorBERTo LoRA receita 3, treinado no desenvolvimento | 1,792 | 44.81% | 44.81% |

Controles reutilizam previsões salvas após conferir IDs e grupos/rótulos, ou
hashes dos textos no JSONL da receita 3. Nenhuma referência foi retreinada
ou reexecutada. O novo candidato perde 118 acertos (-2.95 pontos percentuais)
para LoRA original; o intervalo descritivo do bootstrap pareado por componentes
é [-5.14, -1.00] pontos (2,000 sorteios, semente 42). O conjunto reservado já
consultado continua exploratório.

A acurácia reservada é 0.20 pontos percentuais abaixo dos 42.57% OOF de
desenvolvimento. Essa proximidade não estabelece estabilidade com outro tamanho
de treino: a cabeça final não foi reajustada nesta avaliação. O codificador
fixo evita otimização do adaptador, mas a representação mostra qualidade de
classificação inferior à dos controles.

A matriz, com classes verdadeiras nas linhas e preditas nas colunas em ordem
`c1, c234, c5`, é `[[497,394,366], [396,477,544], [253,352,720]]`.
A revocação é 39.54%, 33.66% e 54.34%, respectivamente. Extrair 3,705 textos
únicos reservados levou 72.90 segundos, com pico de VRAM alocada 1,300.62 MiB;
a avaliação completa levou 75.00 segundos, offline. Parâmetros treináveis do
codificador e atualizações do otimizador foram ambos zero.

Um leitor ZIP/XML independente verificou cada texto, rótulo, ID e grupo
congelado. Os 3,999 vetores de probabilidades foram recalculados diretamente
da cabeça NPZ segura e dos embeddings salvos, com erro absoluto máximo zero.
Cálculos manuais de matriz, acurácia, F1, revocação e contagens pareadas
coincidiram com o relatório. Os artefatos estão separados em
`results/norberto-frozen-logistic-holdout-20261105/`; o manifesto original é
registro imutável da execução anterior só de desenvolvimento. Não houve nova
avaliação oficial nem ajuste completo.

Veja o [agregado do conjunto reservado](data/norberto-frozen-logistic-holdout.json)
e o [comando de reprodução](../archive/norberto/frozen-logistic.md#holdout-evaluation).

<a id="ep1-cv-winner-candidate-08-reloaded"></a>

### Vencedor da validação cruzada do EP1: candidate-08 recarregado

A validação salva dos três finalistas classificou `candidate-08` primeiro,
com acurácia média 43.73% (LoRA posto 16, alpha 32, taxa 0.0003). A pedido do
usuário, o adaptador final preservado foi recarregado e avaliado no conjunto
reservado e teste1. É o adaptador treinado nas 16,093 linhas por três épocas,
selecionadas pela mediana interna `2, 3, 3`; não é checkpoint de fold.
Não houve retreinamento, busca ou promoção.

| Avaliação | Acertos candidate-08 | Acurácia candidate-08 | Macro-F1 candidate-08 | Acurácia original EP1 |
|---|---:|---:|---:|---:|
| Conjunto reservado congelado, 3,999 linhas | 1,439 | 35.98% | 35.31% | 45.31% (1,812 acertos) |
| Teste1, 900 linhas | 313 | 34.78% | 34.30% | 48.00% (432 acertos) |

Teste1 foi avaliado contra `data/test1-rotulado-para-teste-estatistico.xlsx`
após conferir IDs e textos com a entrada não rotulada. Todas as previsões de
ambas as bases foram salvas antes dos registros com gabarito. Arquivos do
adaptador final e estado completo carregado permaneceram inalterados, com
zero parâmetros treináveis, gradientes ou atualizações na inferência.
Grupos de similaridade de treino e conjunto reservado eram disjuntos.

Leitura ZIP/XML independente, métricas manuais e reconstrução dos logits
verificaram as 4,899 linhas. Cada previsão reservada coincidiu com a avaliação
histórica candidate-08 (diferença máxima de probabilidade 2.97e-8). Controles
do modelo entregue foram recalculados das previsões salvas. Candidate-08 perde
373 acertos reservados (-9.33 pontos percentuais) e 119 em teste1 (-13.22 pontos)
frente ao original. A vantagem na validação cruzada não generalizou ao
adaptador final. O conjunto reservado já consultado continua exploratório.

A execução levou 200.64 segundos offline, sendo 152.12 no conjunto reservado
e 33.74 em teste1, incluindo tokenização. O pico de memória alocada na GPU
foi 954.91 MiB. Seis verificações de regressão relevantes passaram. Previsões,
métricas e comparações com gabarito por linha estão em
`results/norberto-ep1-candidate08-evaluation-20261105/evaluation/`.

Veja o [agregado verificado](data/norberto-ep1-candidate08-evaluation.json) e
o [guia de reprodução](../archive/norberto/candidate08-evaluation.md).

<a id="full-corpus-refits-verified-test1-results"></a>

## Retreinamentos completos: resultados verificados de teste1

Ambos os modelos treinaram nas 20,092 linhas. As métricas abaixo foram
recalculadas independentemente em CPU das previsões existentes, alinhando
os 900 textos/IDs da fonte e do gabarito. Hashes de artefatos, treino, previsões
e planilhas foram conferidos. A verificação não executou inferência nem
retreinamento.

| Modelo | Acertos | Acurácia | Macro-F1 | Precisão macro | Revocação macro |
|---|---:|---:|---:|---:|---:|
| SVM, C=0.05 fixo | 451/900 | 50.11% | 49.87% | 49.93% | 50.11% |
| NorBERTo receita 3 | 347/900 | 38.56% | 33.29% | 40.83% | 38.56% |

O SVM completo ganha seis acertos (+0.67 pontos percentuais) frente ao congelado
de desenvolvimento (445/900). Essa diferença descritiva não estabelece
significância estatística.

O NorBERTo completo perde 75 acertos (-8.33 pontos percentuais) frente à receita
3 treinada no desenvolvimento (422/900). As contagens preditas são `c1=216`,
`c234=62` e `c5=622`; o gabarito tem 300 por classe. Revocação: 30.33% em c1,
9.00% em c234 e 76.33% em c5.

NorBERTo concluiu 2,512 atualizações e duas épocas em 40.20 minutos sem parar
pelo orçamento. A conclusão não melhorou a classificação. A maior norma
registrada antes da limitação foi 1,196.86 em 1,120, contra 22.58 no
desenvolvimento; ambos usam escala do otimizador 4. O pico e o desequilíbrio
das previsões são compatíveis com otimização instável, sem estabelecer sua
causa. Nenhuma configuração de treino ou composição de ensemble mudou pelos
rótulos de teste1.

O [agregado da verificação](data/full-test1-verification.json) inclui métricas
completas, matrizes, resultados por classe e identidades dos modelos.

<a id="training-scale-audit-recipe-3"></a>

### Auditoria da escala de treinamento: receita 3

Este experimento desativado está na [auditoria histórica](../archive/norberto/recipe3-scale-audit.md).
A acurácia de 52% na amostra de 300 linhas de treino não era validação em
conjunto reservado. O modelo original da entrega do EP1 está selecionado;
nenhuma dessas mudanças de treino está ativa.

<a id="norberto-frozen-holdout-3999-responses"></a>

## Conjunto reservado congelado do NorBERTo: 3,999 respostas

| Adaptador | Acurácia | Macro-F1 | Relatório |
|---|---:|---:|---|
| EP1 original | 45.31% | 45.31% | Referência na comparação da receita 3 |
| Receita 3, compatível com o otimizador | 44.81% | 44.81% | [Validação concluída](norberto-recipe3-holdout.md) |
| Receita 2, gradientes em unidades da perda média | 35.66% | 35.59% | [Validação histórica](../archive/norberto/recipe2-development-holdout.md) |

Adaptadores do corpus completo incluem o antigo conjunto reservado no treino
e não podem usar essas linhas como validação inédita. Avaliações antigas de
teste1 e investigações de treino estão no [arquivo](../archive/README.md).

<a id="technical-aggregates"></a>

## Agregados técnicos

A [auditoria de sobreposição](dataset-overlap.md) verifica o corpus reutilizado,
a identidade de teste1 e a sobreposição exata/normalizada entre treino e ambos
os testes.

Os JSON em [`data/`](data/) contêm métricas, hashes, verificações de execução
e comparações pareadas registrados. Mantêm nomes e bytes originais. Cada
relatório detalhado vincula seu agregado. Previsões brutas, checkpoints e
caches são artefatos locais de execução e não são duplicados aqui.
