<a id="frozen-stacking--selective-gpt-measured-experiment"></a>

# Stacking congelado + GPT seletivo: experimento medido

[Índice dos resultados](README.md) · [Comandos e protocolo](../guides/stacking-gpt-cascade.md) · [Agregado verificado](data/stacking-gpt-cascade.json)

**GPT seletivo não melhorou este experimento.** A validação escolheu stacking
local sem chamadas GPT antes da abertura do gabarito final. Os três orçamentos
positivos predefinidos também reduziram macro-F1 e acurácia em teste1.
A cascata permanece disponível com orçamentos configuráveis, cache persistente,
retentativas e fallback; o comportamento selecionado mantém stacking.
Nenhum classificador ou regra mudou em resposta aos resultados finais.

<a id="validation-only-model-and-budget-selection"></a>

## Seleção de modelo e orçamento somente na validação

As bases congeladas usam 16,093 linhas originais. As 3,999 reservadas mantêm
os papéis: 2,027 ajustam calibradores; 1,972 sustentam meta-treino e roteamento
por **previsões OOF agrupadas aninhadas 5x5**. Cada alvo externo e seu grupo
ficam fora do ajuste do meta-modelo e da escolha interna de C. Os cinco folds
escolheram C=0.1 com seus próprios treinos. Stacking de produção e todos os
artefatos-base permanecem idênticos em bytes.

O roteamento combina entropia normalizada, probabilidade máxima, margem das
duas maiores e divergência nativa pareada SVM/RL/NorBERTo/Qwen. A busca fixa
de doze candidatos capturou 122 erros OOF nas 197 linhas iniciais. Os doze
capturaram a mesma quantidade; o desempate determinístico escolheu peso de
confiança 0.9, limiar máximo de probabilidade 0.5 e de margem 0.05. Ordem e
elegibilidade finais dependem só das previsões, sem gabarito.

No orçamento inicial de 10%, os prompts foram comparados nas mesmas linhas:

| Método de validação | Acertos / 1,972 | Macro-F1 | Corrigidos | Prejudicados |
|---|---:|---:|---:|---:|
| Stacking OOF aninhado | 936 | 47.2977% | 0 | 0 |
| GPT somente com texto original | 925 | 46.5751% | 31 | 42 |
| GPT com previsões locais | 929 | 46.8407% | 22 | 29 |

Previsões locais ajudaram frente ao texto isolado, mas ambos ficaram abaixo
do stacking. Só o contexto local foi estendido a 20%, reutilizando as 197
decisões. O prompt congelado inclui texto, definições de classes, rótulos
nativos locais, probabilidades calibradas e de stacking; são identificados
como previsões falíveis, nunca verdade conhecida.

Comparação de orçamentos na validação:

| Orçamento GPT | Roteados | Acertos | Acurácia | Macro-F1 | Corrigidos | Prejudicados |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 936/1972 | 47.4645% | 47.2977% | 0 | 0 |
| 5% | 98 | 937/1972 | 47.5152% | 47.2762% | 16 | 15 |
| 10% | 197 | 929/1972 | 47.1095% | 46.8407% | 22 | 29 |
| 20% | 394 | 937/1972 | 47.5152% | 47.1016% | 56 | 55 |

Orçamentos 5% e 20% ganharam um acerto líquido cada, mas reduziram macro-F1,
a métrica principal. Assim, zero venceu a validação. São diagnósticos de
seleção, não estimativas independentes após ajustes.

<a id="final-comparison-all-900-original-rows"></a>

## Comparação final: as 900 linhas originais

Prompt, fórmula, limiares, opções de orçamento e identidades das fontes foram
congelados em `models/stacking-gpt-cascade/` antes da inferência final.
A união de 5/10/20% exigiu no máximo 180 alvos. Cada arquivo foi salvo antes
da abertura do gabarito pelo avaliador separado. Nenhum rótulo final escolheu
exemplos encaminhados ou modificou a configuração congelada.

Resultados medidos em teste1:

| Orçamento GPT | Roteados | Acertos | Acurácia | Macro-F1 | Corrigidos | Prejudicados |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 0 | 440/900 | 48.8889% | 48.8034% | 0 | 0 |
| 5% | 45 | 435/900 | 48.3333% | 48.1824% | 4 | 9 |
| 10% | 90 | 434/900 | 48.2222% | 48.0477% | 9 | 15 |
| 20% | 180 | 436/900 | 48.4444% | 48.1761% | 30 | 34 |

Orçamentos menores reutilizam respostas das mesmas seleções ordenadas
aninhadas. Outras mudanças de rótulo continuaram erradas: 8 a 5%, 11 a 10%
e 20 a 20%. Não houve chamadas falhas nem fallback nas execuções medidas.

| Orçamento GPT | F1 c1 | F1 c234 | F1 c5 |
|---|---:|---:|---:|
| 0% | 45.7594% | 46.4023% | 54.2484% |
| 5% | 45.3280% | 46.2687% | 52.9506% |
| 10% | 45.3280% | 45.5235% | 53.2915% |
| 20% | 45.6000% | 45.0704% | 53.8578% |

As matrizes usam classes verdadeiras nas linhas e preditas nas colunas,
em ordem `c1, c234, c5`:

```text
GPT 0%:
[116, 125, 59]
[55, 158, 87]
[36, 98, 166]

GPT 5%:
[114, 117, 69]
[53, 155, 92]
[36, 98, 166]

GPT 10%:
[114, 115, 71]
[53, 150, 97]
[36, 94, 170]

GPT 20%:
[114, 108, 78]
[51, 144, 105]
[35, 87, 178]

```

<a id="calls-tokens-cost-and-latency"></a>

## Chamadas, tokens, custo e latência

O ambiente foi `codex-cli 0.166.0`, solicitando `gpt-6.1-sol` com raciocínio
`low`. O texto original foi preservado, JSON estrito validado e ferramentas/
contexto do repositório desativados na CLI isolada. Não há controle de
temperatura. O modelo remoto é um alias solicitado; a reprodução usa respostas
registradas em cache, sem presumir determinismo de bytes em pedidos futuros.

| Etapa | Instâncias-alvo | Acertos de cache por item | Novas chamadas | USD equivalentes à API | Segundos decorridos |
|---|---:|---:|---:|---:|---:|
| validation_text_only_10pct | 197 | 0 | 24 | 0.374843 | 96.02 |
| validation_with_local_10pct | 197 | 0 | 25 | 0.450464 | 94.07 |
| validation_selected_prompt_extension_20pct | 394 | 197 | 25 | 0.455602 | 91.25 |
| final_frozen_budget_comparison | 180 | 0 | 23 | 0.391949 | 86.17 |

Total: **97 chamadas**, **883,532 tokens de entrada** (incluindo 147,840 em cache),
**18,669 de saída** e **USD 1.672858** de estimativa equivalente à API padrão.
A autenticação usou o login ChatGPT Codex existente; não é fatura medida da
assinatura. Preços conferidos em 2026-11-05 na
[página oficial GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol):
USD 2 entrada, 0.10 entrada em cache e 10 saída por milhão de tokens.

A comparação final completa levou 86.17 segundos com duas solicitações
simultâneas; a soma das latências foi 171.38 segundos. Cada solicitação
agrupou até oito alvos independentes. Os relatórios rateiam cada lote pelos
alvos únicos para contabilizar custo/latência dos orçamentos menores:

| Orçamento GPT final | Roteados | Recibos de lotes utilizados | USD alocados | Segundos de solicitações alocados |
|---|---:|---:|---:|---:|
| 0% | 0 | 0 | 0.000000 | 0.00 |
| 5% | 45 | 6 | 0.103906 | 44.47 |
| 10% | 90 | 12 | 0.205357 | 86.41 |
| 20% | 180 | 23 | 0.391949 | 171.38 |

São rateios do experimento compartilhado, não medições independentes de tempo
ou cobrança de cada orçamento reexecutado. O caminho de produção selecionado
com orçamento zero também foi executado: zero tentativas GPT e os 900 rótulos
idênticos ao stacking congelado. Reproduzir as 180 decisões finais com limite
de solicitações zero recuperou todos os rótulos, com 180 acertos de cache por
item e nenhuma tentativa nova.

<a id="verification-and-limitations"></a>

## Verificação e limitações

Os **201 testes disponíveis passaram**, incluindo módulos do SDK. Cobrem
limites percentuais, divergência nativa dos quatro membros, isolamento de
grupos externos/internos, preservação do texto, cache entre orçamentos/lotes,
retentativas limitadas, disjuntor, fallback explícito, saídas imutáveis e
leitura do gabarito só após persistir todas as previsões.

Um verificador independente da biblioteca padrão conferiu identidades das
planilhas, saídas dos coeficientes OOF aninhados, cálculos de roteamento,
limites percentuais, procedência GPT, seleção só na validação, classificações
finais, F1 por classe, matrizes e todas as correções/prejuízos. O erro máximo
de reconstrução OOF foi 1.67e-16. Evidência e script ficam em
`results/cascade-verification-20261105/`; validação do roteamento, predição
final e avaliação têm diretórios separados preservados.

A seleção do checkpoint Qwen usou as 3,999 linhas reservadas. Os arquivos
oficiais também repetem textos normalizados entre teste1 e desenvolvimento
(85 linhas), calibração (11) e meta-treino (9), conforme a
[auditoria anterior](ensemble-soft.md). Nenhum novo rótulo de teste1 entrou
no ajuste, mas OOF do meta-modelo não remove essa exposição sem mudar modelos/
dados originais. As 900 linhas continuam avaliadas. É comparação exploratória
com exposição histórica, não nova amostra final inédita.

Nenhum modelo-base foi retreinado, experimento original sobrescrito,
configuração remota ajustada após avaliação final ou entrega teste2 gerada.
