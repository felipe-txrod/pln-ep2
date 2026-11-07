<a id="offline-rag-and-selective-gpt-investigation"></a>

# Investigação offline de RAG e GPT seletivo

[Evidências em formato estruturado](data/rag-offline-research.json)

Este estudo fez **zero novas chamadas GPT**, não usou teste1/teste2 para otimizar
ou selecionar configurações e nunca retreinou SVM, LR, NorBERTo, Qwen ou
stacking. Pacotes históricos e caches de respostas foram preservados. Os novos
pequenos classificadores de utilidade são políticas só de pesquisa, não
substitutos da LR base treinada.

<a id="verified-historical-problem"></a>

## Problema histórico verificado

Nas 188 consultas da confirmação, hybrid-3-local corrigiu 29 erros e prejudicou
32 acertos do stacking. Dos prejuízos, 26 eram da classe verdadeira c234.
O primeiro exemplo recuperado coincidiu com o rótulo da consulta 60/188 vezes
(31.91%). RAG mudou 55 decisões GPT frente a previsões locais: 21 acertos
exclusivos contra 15 acertos perdidos.

O cosseno E5 médio do primeiro vizinho foi 0.9505 apesar das divergências.
Logo, alta similaridade semântica não estabeleceu concordância de notas neste
domínio. Semelhança temática e clareza percebida são sinais distintos; essa
explicação permanece hipótese, não causa comprovada.

| Caso descritivo de validação | Linha consultada | Gabarito | Stacking | GPT | Primeira linha de treino / rótulo | Cosseno E5 |
|---|---:|---|---|---|---|---:|
| harmed_c234 | 2218 | c234 | c234 | c5 | 15171 / c1 | 0.9507 |
| corrected | 1808 | c5 | c234 | c5 | 8942 / c1 | 0.9353 |
| highest_cosine_label_mismatch | 3767 | c5 | c234 | c1 | 3259 / c1 | 0.9964 |

São diagnósticos retrospectivos de erros, não exemplos de desenvolvimento para
novo prompt. Trechos de consulta/exemplo, as quatro previsões nativas, decisões
GPT só locais e os três rótulos recuperados estão no agregado; rastreamentos
completos ficam locais. A nota subjetiva de um exemplo não pode ser transferida
ao alvo só pela sobreposição de tema ou palavras. Pedido original e anexos não
estão disponíveis, portanto a revisão não determina por que o avaliador humano
atribuiu uma nota específica.

A divergência de maior cosseno compartilha um longo modelo administrativo de
encaminhamento de anexos, mas tem rótulos diferentes; o cosseno lexical fica
abaixo do limiar de exclusão de duplicatas. O caso c234 prejudicado tem três
votos-base c5 enquanto stacking acerta, mostrando que maioria não garante
substituição segura. No caso corrigido, RAG muda o GPT somente local de c234
para o c5 correto apesar do primeiro vizinho c1. Concordância dos rótulos
recuperados é insuficiente para explicar causalmente sucessos ou falhas individuais.

| Classe verdadeira da confirmação (somente análise) | Corrigidos | Prejudicados | Variação líquida de acertos |
|---|---:|---:|---:|
| c1 | 7 | 5 | +2 |
| c234 | 5 | 26 | -21 |
| c5 | 17 | 1 | +16 |

A concentração de prejuízos em c234 motiva regras diagnósticas de proteção,
mas não as valida. A regra na inferência usa a **classe predita pelo stacking**,
nunca a verdadeira do alvo.

<a id="frozen-norberto-extraction"></a>

## Extração com NorBERTo congelado

O checkpoint é ModernBERT de 22 camadas, dimensão oculta 768 e LoRA posto 4
em Wqkv. O adaptador é o artefato EP1 inalterado (`d693f6e2…`). O classificador
foi treinado com média mascarada. Extraímos média e primeiro token nas camadas
11 e 22 na mesma passagem congelada FP32/SDPA, com pré-processamento original
início/fim de 512 tokens, preservando maiúsculas/minúsculas, e normalização L2.

O tokenizador insere de fato [CLS] ID 3 e [SEP] ID 4. Atributos CLS usam o
primeiro token real. O modelo Transformers instalado expõe a última entrada
`hidden_states` antes de `final_norm`; a representação final usa
`last_hidden_state` depois dela, como a entrada do classificador. A camada
intermediária 11 usa seu estado anterior à normalização final.

As exportações mantêm 11,562 exemplos elegíveis de treino e 1,972 linhas de
validação. Textos ausentes são codificados uma vez para todas as representações,
e vetores em cache são reutilizados. Vetores E5 existentes também são
reutilizados. O cache NorBERTo-logistic antigo tem vetores da **base não adaptada**
e não representa este adaptador; foi preservado, sem renomeação enganosa ou sobrescrita.

<a id="retrieval-comparison"></a>

## Comparação da recuperação

Há 171 variantes predefinidas de método/representação/política e controles
aleatório e por comprimento. Desenvolvimento usa folds OOF originais 0/1/2
(1143 linhas), e auditoria usa 3/4 (829 linhas), com grupos disjuntos.
A proposta selecionada **só no desenvolvimento** é
`tfidf_e5_norberto--layer11_mean--relevance`. A tabela de sete métodos fixa a
representação escolhida pelo desempenho do NorBERTo isolado no desenvolvimento:
`layer22_mean`. Mantém a política MMR histórica fixa.

Hit@1 é a concordância da classe do primeiro vizinho. Revocação@3 por classe
significa ao menos um dos três exemplos da classe verdadeira; pureza@3 é a
fração que coincide. Usam gabarito só após recuperar. Revocação por documento
também é exportada com denominador igual ao banco da classe. Nenhuma mede
macro-F1 do GPT.

| Método | Hit@1 | Revocação@3 por classe | Pureza@3 | Classes distintas |
|---|---:|---:|---:|---:|
| tfidf | 37.83% | 92.09% | 35.46% | 2.688 |
| e5 | 38.29% | 100.00% | 33.33% | 3.000 |
| norberto | 38.64% | 100.00% | 33.33% | 3.000 |
| tfidf_e5 | 38.79% | 83.67% | 35.73% | 2.423 |
| tfidf_norberto | 38.13% | 85.34% | 36.36% | 2.419 |
| e5_norberto | 38.79% | 85.85% | 36.61% | 2.425 |
| tfidf_e5_norberto | 38.59% | 86.00% | 36.66% | 2.422 |
| random | 32.76% | 71.15% | 33.32% | 2.119 |
| length | 34.63% | 71.15% | 34.47% | 2.086 |

Essas linhas descrevem as 1,972 consultas de validação. A configuração usa só
desenvolvimento; os valores da auditoria da proposta escolhida aparecem
separadamente abaixo.

| Partição da proposta selecionada | Hit@1 balanceado | Pureza@3 balanceada | Revocação@3 por classe |
|---|---:|---:|---:|
| Desenvolvimento | 40.37% | 37.71% | 72.97% |
| Auditoria | 37.97% | 37.31% | 71.53% |

Frente à política atual TF-IDF+E5/legacy, a diferença de Hit@1 na auditoria é
-1.21%; intervalo descritivo de 95% por bootstrap de grupos [-4.41%, +1.99%].
Pureza@3 muda +1.93%. O intervalo descreve correspondência de rótulos da
recuperação, não melhora do GPT.

<a id="pooling-and-layer-ablation"></a>

## Ablação de agregação e camada

| Representação NorBERTo / política | Hit@1 balanceado no desenvolvimento | Hit@1 balanceado na auditoria | Pureza@3 da auditoria |
|---|---:|---:|---:|
| layer11_mean/legacy | 37.01% | 35.85% | 33.33% |
| layer11_mean/relevance | 37.01% | 35.85% | 35.67% |
| layer11_cls/legacy | 34.31% | 38.86% | 33.33% |
| layer11_cls/relevance | 34.31% | 38.86% | 37.88% |
| layer22_mean/legacy | 38.95% | 38.28% | 33.33% |
| layer22_mean/relevance | 38.95% | 38.28% | 37.07% |
| layer22_cls/legacy | 36.87% | 41.40% | 33.33% |
| layer22_cls/relevance | 36.87% | 41.40% | 39.24% |

| Representação do codificador | Cosseno médio de pares aleatórios | Desvio-padrão | Norma do vetor médio |
|---|---:|---:|---:|
| layer11_mean | 0.8068 | 0.5289 | 0.8812 |
| layer11_cls | 0.9248 | 0.0447 | 0.9619 |
| layer22_mean | 0.9534 | 0.0278 | 0.9761 |
| layer22_cls | 0.7960 | 0.1051 | 0.8936 |
| e5 | 0.8849 | 0.0212 | 0.9411 |

Não se presume que as quatro representações congeladas melhorem a recuperação.
Cosseno elevado de pares aleatórios e norma elevada do vetor médio indicam
anisotropia: pares de treino sem relação também podem ter cosseno alto.
A dispersão ampla de layer11-mean reforça a cautela ao transferir limiar
absoluto entre representações. E5 usa seu prefixo de consulta e truncamento
no início originais; NorBERTo usa a política início/fim original, portanto
comparar codificadores inclui diferenças de tokenização/truncamento.
Escolher layer22-CLS por seu número maior na auditoria violaria a seleção
restrita ao desenvolvimento.

<a id="mmr-and-diversity-ablation"></a>

## Ablação de MMR e diversidade

MMR histórico tem peso de relevância 0.8, bônus de classe ausente 0.04,
máximo de dois exemplos por classe e 120 candidatos. Todas as variantes
mantêm exclusão de duplicatas/quase duplicatas, grupos únicos e exemplos
somente de treino. A fusão usa posições recíprocas iguais, não mistura
escalas brutas de escores.

O bônus de classe é igual para todos na primeira escolha, ainda sem redundância.
Sozinho não explica o primeiro vizinho fraco. Bônus/limite afetam sobretudo
segundo e terceiro exemplos. Mudanças de posição e cosseno mostram quando a
diversidade substitui um vizinho mais próximo.

| Política atual TF-IDF+E5 | Hit@1 | Revocação@3 | Pureza@3 | Classes distintas | Posição média selecionada |
|---|---:|---:|---:|---:|---:|
| legacy | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| no_bonus | 38.79% | 75.15% | 36.19% | 2.158 | 2.44 |
| no_cap | 38.79% | 82.00% | 36.33% | 2.359 | 2.51 |
| no_bonus_or_cap | 38.79% | 70.23% | 37.58% | 1.974 | 2.27 |
| relevance | 38.79% | 70.33% | 37.64% | 1.975 | 2.24 |
| no_redundancy | 38.79% | 81.80% | 36.02% | 2.367 | 2.54 |
| pool40 | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| pool240 | 38.79% | 83.67% | 35.73% | 2.423 | 2.62 |
| weight50 | 38.79% | 89.00% | 35.58% | 2.578 | 2.87 |

| Recuperador denso / política | Revocação@3 | Pureza@3 | Classes distintas | Posição média selecionada |
|---|---:|---:|---:|---:|
| e5--none--legacy | 100.00% | 33.33% | 3.000 | 5.70 |
| e5--none--no_bonus | 77.84% | 37.10% | 2.174 | 3.62 |
| e5--none--no_cap | 100.00% | 33.33% | 3.000 | 5.70 |
| e5--none--no_bonus_or_cap | 72.52% | 38.07% | 2.001 | 3.43 |
| e5--none--relevance | 72.26% | 37.68% | 1.984 | 2.21 |
| e5--none--no_redundancy | 99.95% | 33.37% | 2.998 | 3.81 |
| norberto--layer22_mean--legacy | 100.00% | 33.33% | 3.000 | 5.44 |
| norberto--layer22_mean--no_bonus | 78.25% | 36.78% | 2.176 | 3.56 |
| norberto--layer22_mean--no_cap | 100.00% | 33.33% | 3.000 | 5.44 |
| norberto--layer22_mean--no_bonus_or_cap | 73.83% | 37.54% | 2.030 | 3.45 |
| norberto--layer22_mean--relevance | 72.16% | 37.34% | 1.995 | 2.20 |
| norberto--layer22_mean--no_redundancy | 100.00% | 33.33% | 2.999 | 3.88 |

Nos braços históricos E5 e NorBERTo isolado selecionado, toda consulta recupera
as três classes. Isso torna Revocação@3 exatamente 100% e pureza um terço;
é consequência da diversidade forçada, não de identificar a classe-alvo.
Remover só o limite não resolve; remover o bônus, sim. Cossenos densos são
muito concentrados, e bônus fixo 0.04 pode dominar pequenas diferenças de
relevância. A fusão por posições recíprocas altera a dispersão dos escores,
explicando por que a mesma política age diferentemente em braços densos e fundidos.

No híbrido atual, retirar bônus e limite aumenta pureza e reduz cobertura;
ampliar de 40 para 240 candidatos não traz benefício agregado aqui. Remover
só redundância não resolve a divergência. Não se presume superioridade de
demonstrações homogêneas ou diversificadas para GPT. São compromissos da
recuperação, não ganhos confirmados do GPT.

<a id="per-class-retrieval-diagnostics"></a>

## Diagnósticos da recuperação por classe

| Método | Classe verdadeira | Hit@1 | Revocação@3 por classe | Pureza@3 |
|---|---|---:|---:|---:|
| tfidf | c1 | 35.69% | 89.23% | 34.14% |
| tfidf | c234 | 33.80% | 91.37% | 33.76% |
| tfidf | c5 | 44.32% | 95.65% | 38.62% |
| e5 | c1 | 36.33% | 100.00% | 33.33% |
| e5 | c234 | 34.94% | 100.00% | 33.33% |
| e5 | c5 | 43.86% | 100.00% | 33.33% |
| norberto | c1 | 37.46% | 100.00% | 33.33% |
| norberto | c234 | 33.95% | 100.00% | 33.33% |
| norberto | c5 | 44.95% | 100.00% | 33.33% |
| tfidf_e5 | c1 | 36.50% | 77.49% | 32.26% |
| tfidf_e5 | c234 | 35.50% | 84.87% | 34.75% |
| tfidf_e5 | c5 | 44.63% | 88.34% | 40.18% |
| tfidf_norberto | c1 | 36.33% | 82.64% | 34.24% |
| tfidf_norberto | c234 | 34.23% | 85.15% | 35.12% |
| tfidf_norberto | c5 | 44.17% | 88.18% | 39.76% |
| e5_norberto | c1 | 41.64% | 82.96% | 35.74% |
| e5_norberto | c234 | 29.99% | 86.70% | 35.12% |
| e5_norberto | c5 | 45.72% | 87.71% | 39.09% |
| tfidf_e5_norberto | c1 | 38.59% | 82.48% | 34.62% |
| tfidf_e5_norberto | c234 | 33.10% | 86.99% | 35.64% |
| tfidf_e5_norberto | c5 | 44.63% | 88.34% | 39.76% |
| random | c1 | 29.58% | 64.95% | 29.05% |
| random | c234 | 29.70% | 69.45% | 32.30% |
| random | c5 | 39.19% | 79.00% | 38.57% |
| length | c1 | 21.06% | 72.51% | 30.55% |
| length | c234 | 47.10% | 77.79% | 41.02% |
| length | c5 | 34.06% | 62.52% | 31.05% |

| Método | c1 recuperados | c234 recuperados | c5 recuperados | Cosseno lexical médio | Cosseno E5 médio | Cosseno NorBERTo médio |
|---|---:|---:|---:|---:|---:|---:|
| tfidf | 31.66% | 32.74% | 35.60% | 0.4443 | n/a | n/a |
| e5 | 33.33% | 33.33% | 33.33% | 0.3399 | 0.9456 | n/a |
| norberto | 33.33% | 33.33% | 33.33% | 0.3066 | n/a | 0.9861 |
| tfidf_e5 | 29.77% | 34.09% | 36.14% | 0.4213 | 0.9450 | n/a |
| tfidf_norberto | 30.51% | 35.18% | 34.31% | 0.3948 | n/a | 0.9849 |
| e5_norberto | 30.93% | 34.48% | 34.58% | 0.3527 | 0.9428 | 0.9853 |
| tfidf_e5_norberto | 30.59% | 35.14% | 34.26% | 0.4019 | 0.9433 | 0.9831 |
| random | 29.43% | 32.74% | 37.83% | 0.0314 | n/a | n/a |
| length | 30.71% | 39.69% | 29.60% | 0.0381 | n/a | n/a |

Resultados de similaridade por classe estão no agregado estruturado e no
relatório local completo. Similaridades são reportadas no espaço próprio de
cada codificador; não são probabilidades comparáveis de confiança.

<a id="routing-versus-acceptance"></a>

## Roteamento versus aceitação

Antes da chamada, os modelos veem probabilidades calibradas de stacking/bases,
classes nativas, margem, entropia, divergência e, opcionalmente, estatísticas
dos exemplos originalmente recuperados. Depois, podem ver a classe GPT
retornada e sua concordância com stacking, votos-base e exemplos. O gabarito
não entra em nenhuma função de atributos. Novos contextos recuperados pelo
NorBERTo **não** são pareados com respostas antigas como se a API os tivesse visto.

O modelo de utilidade estima P(corrected)-P(harmed). É uma pequena regressão
logística regularizada ajustada em cada fold de treino. Cinco folds externos
originais com grupos disjuntos avaliam a política condicionalmente às previsões
históricas fixas dos estágios anteriores; três internos escolhem família,
regularização, limiar e orçamento. O seletor conjunto inclui a busca de família,
evitando escolhê-la nos mesmos resultados OOF usados para alegar ganho.

Não é avaliação aninhada nova de ponta a ponta: o preditor OOF histórico de
stacking de uma linha de treino da política pode ter usado rótulos de um fold
de avaliação da política. As previsões do próprio grupo continuam OOF, e
transformações/modelos novos excluem os grupos avaliados, mas a matriz OOF
anterior é fixa, sem regeneração por divisão da política. Exposição Qwen e
desenvolvimento anterior do roteador também permanecem. Essas restrições
impedem alegar nova estimativa imparcial de generalização da cascata, mesmo
com família positiva.

Só 236 consultas históricas têm respostas observadas. A reavaliação restringe-se
a elas; os outros 1736 alvos mantêm stacking porque a política não os consulta.
Não se identificam efeitos de novo roteador fora dessa cobertura sem novas
respostas GPT. Referências somente de aceitação usam todas as chamadas
registradas (11.97% da população retrospectiva), não garantem orçamento de
10% em produção. Políticas roteadas têm limites de 5/10% por fold e incluem zero.

| Política OOF agrupada | Acertos /1972 | Acurácia | Macro-F1 | Corrigidos | Prejudicados | Alvos solicitados na reavaliação |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 936 | 47.46% | 47.30% | 0 | 0 | 0 |
| unconditional_logged | 936 | 47.46% | 47.20% | 37 | 37 | 236 |
| rules | 930 | 47.16% | 46.97% | 10 | 16 | 236 |
| accept_local | 933 | 47.31% | 47.12% | 15 | 18 | 236 |
| route_local | 938 | 47.57% | 47.32% | 19 | 17 | 116 |
| accept_retrieval | 932 | 47.26% | 47.06% | 12 | 16 | 192 |
| route_retrieval | 937 | 47.52% | 47.30% | 15 | 14 | 89 |
| cascade | 936 | 47.46% | 47.26% | 9 | 9 | 72 |
| joint_nested_selection | 935 | 47.41% | 47.19% | 16 | 17 | 163 |

O intervalo descritivo de 95% por bootstrap de grupos da diferença de macro-F1
do seletor conjunto é [-0.71%, +0.45%]. Não estabelece benefício. Melhoras de
famílias isoladas são exploratórias; gerar hipóteses retrospectivamente nesses
resultados acrescenta otimismo.

<a id="fixed-rule-controls-and-safe-pre-call-skips"></a>

## Controles de regras fixas e omissão segura antes da chamada

| Regra fixa | Acertos | Macro-F1 | Corrigidos | Prejudicados | Alvos solicitados equivalentes | Alvos omitidos com segurança |
|---|---:|---:|---:|---:|---:|---:|
| protect_c234 | 941 | 47.50% | 11 | 6 | 71 | 165 |
| two_base_votes | 935 | 47.17% | 25 | 26 | 188 | 48 |
| three_base_votes | 937 | 47.33% | 5 | 4 | 28 | 208 |
| two_examples | 937 | 47.30% | 17 | 16 | 89 | 147 |
| votes_and_example | 936 | 47.22% | 24 | 24 | 162 | 74 |
| protect_and_votes | 940 | 47.48% | 5 | 1 | 45 | 191 |

Resultados de regras fixas são descritivos, sem seleção em teste novo.
Omitir a chamada é seguro em relação à regra posterior quando **nenhum dos
três rótulos GPT válidos pode passar na regra e mudar stacking**. Testes
sintéticos exaustivos e reavaliação registrada verificam equivalência das
previsões. Contagens são instâncias-alvo, não lotes de API; um plano futuro
precisa reagrupar para medir economia de chamadas. Isso não prova que a regra
supere stacking.

| Referência de incerteza com mesmo orçamento | Acertos | Macro-F1 | Alvos solicitados |
|---|---:|---:|---:|
| uncertainty_matched_to_route_local | 930 | 46.96% | 116 |
| uncertainty_matched_to_route_retrieval | 932 | 47.09% | 89 |

Uma verificação cronológica separada escolheu a política só nas 48 consultas
do piloto e avaliou as 1884 linhas restantes da confirmação. Selecionou
`two_examples`: 907 acertos, macro-F1 47.88%, contra 910 e 48.08% do stacking.

<a id="screening-for-any-later-paid-evaluation"></a>

## Critérios para eventual avaliação paga posterior

Proposta selecionada no desenvolvimento:
`tfidf_e5_norberto--layer11_mean--relevance`. Resultado offline: **não passa
no critério predefinido para teste pago**. Exige limite inferior positivo da
diferença de Hit@1 na auditoria e nenhuma queda da pureza média frente ao
híbrido atual. Cobertura/diversidade/relevância ainda exigem revisão. O relatório
não autoriza nem executa chamada paga.

Critério de limite inferior positivo da intervenção aprendida: **não atendido**.
Não promova regra por parecer melhor retrospectivamente nem acrescente rodadas
pagas para resgatar resultado não confirmado. O relatório não muda padrões.
Se um candidato de recuperação passar no critério auxiliar, um pequeno teste
pareado futuro deve congelar prompts/roteamento, manter controle texto+local
e comparar relevância pura com MMR restrito antes de alegar melhora GPT.

<a id="decisions-from-the-measured-evidence"></a>

## Decisões fundamentadas nas medições

- Manter o padrão histórico até validação autorizada separadamente estabelecer
  cascata melhor. Reavaliação com estágios anteriores fixos não promove padrão novo.
- Manter recuperação NorBERTo como componente opcional de pesquisa offline.
  A proposta escolhida no desenvolvimento falha no critério pago, sem promoção.
- Ampliação de candidatos e aceitação aprendida precisam de benefício demonstrado
  antes de acrescentar complexidade ao pipeline ativo.
- Manter opções de bônus/limite/redundância e verificações de equivalência para
  diagnósticos reproduzíveis. Efeito mensurável da diversidade não estabelece
  prompt GPT melhor.
- Nenhuma configuração nova justifica teste pago pelos critérios registrados.
  Estudo futuro autorizado exigiria nova validação agrupada, roteamento/prompts
  congelados, controle só local e limites explícitos de custo. Não escolha novo
  vencedor reutilizando esta auditoria.

<a id="reproducibility-and-boundaries"></a>

## Reprodutibilidade e limites

Reprodução da recuperação histórica: 236/236 sequências exatas de IDs dos
exemplos. A saída bruta contém 171 rastreamentos completos, controles, cópia
do código e hashes; pesos, espaços de cache, IDs e folds são fixados. Todos
os embeddings são finitos/unitários, com caches verificados. Checkpoints,
experimentos e respostas GPT históricos não foram sobrescritos.

Tokens da pesquisa, novo custo GPT e latência remota são zero. O tempo local
é exportado separadamente; quantidades reavaliadas de alvos solicitados não
representam novas chamadas nem redução medida de lotes cobrados.

Não houve otimização ou previsão final em teste1/teste2. OOF aninhada impede
ajuste do meta-modelo no próprio grupo, sujeita à limitação dos estágios
anteriores fixos. Nenhum classificador-base ou stacking original foi reajustado.
Avaliação GPT futura deve usar novas chaves válidas de cache para contextos
mudados; respostas antigas não demonstram melhora por novas demonstrações.

<a id="independent-verification-and-reproduction"></a>

## Verificação independente e reprodução

A suíte completa simulada/offline passou em **229 testes**. Um avaliador
independente recalculou as três métricas principais nos 173 métodos/controles
(341,156 consultas), verificou rótulos de treino e grupos únicos, conferiu os
quatro arrays finitos/unitários, confirmou que 288 ajustes de políticas
excluem grupos de avaliação e reproduziu os nove conjuntos de métricas de
intervenção. Casos fixos finais detectam apenas regressões de compatibilidade;
não selecionaram configurações. Fontes e artefatos históricos dos classificadores/
cascata não mudaram. A proteção lexical explícita dos controles aleatório/
comprimento adicionada após medir não exclui exemplos deste estudo; o código
exato medido fica preservado com hashes do protocolo.

A evidência local está em `results/rag-offline-20261106/`: `features/`,
`retrieval/`, `interventions-final/`, `report/` e `independent-verification.json`.
Relatórios/reavaliações anteriores permanecem. Veja o
[guia da pesquisa offline](../guides/rag-offline-research.md) para comandos,
ambiente e limites. Textos brutos, caches, atributos e 579 MB de evidência
intermediária ficam locais; o agregado compacto é versionado.
