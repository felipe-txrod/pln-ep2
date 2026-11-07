<a id="norberto-evaluation-after-correcting-loss-normalization"></a>

# Avaliação do NorBERTo após corrigir a normalização da perda

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

O adaptador da receita versão 2 obteve **45.33% de acurácia (408/900)** e
**43.83% de macro-F1** em teste1 rotulado. São 67 acertos a mais que no
retreinamento completo histórico, mas 24 a menos que na entrega original do
EP1. Não houve reconfiguração do ensemble nem substituição de modelo.

<a id="training-completion-and-provenance"></a>

## Conclusão do treinamento e procedência

`results/norberto-full-normalized/` concluiu **duas épocas e 2,512 atualizações**
nas **20,092 respostas originais de treino do EP1**, incluindo o conjunto
reservado histórico. O treino levou **2,334.80 segundos (38.91 minutos)** sem
atingir o orçamento de tempo. Os 272,643 parâmetros treináveis salvos são
finitos; os 22 tensores LoRA B contêm atualizações não nulas. Os hashes dos
arquivos exportados passaram na verificação.

O manifesto registra receita versão 2 e
`loss_normalization=sample_count_across_accumulation`. Os SHA-256 do invólucro
de treinamento e da implementação da perda coincidem com os arquivos corrigidos.
Os hiperparâmetros do otimizador mantêm a receita original de duas épocas:
posto 4, alpha 8, dropout 0.05, `Wqkv`, contexto 512 com truncamento início/fim,
FP32, SDPA, taxa de aprendizado 0.001, lote 4, acumulação 4, decaimento de pesos
0.01, proporção de aquecimento 0.1 e semente 42.

SHA-256 dos pesos do adaptador:

```text
f43ebe2b9cc1670057bdaa5c7d43b6235d002837bd8d68dbac50b243a1485d71
```

Todos os IDs, hashes dos textos e rótulos de treino salvos foram conferidos com
a planilha original. A revisão da base pré-treinada continua sendo
`db73446f89c96044863ea05a39f680524b84bccb`.

<a id="test1-metrics"></a>

## Métricas de Teste1

A inferência usou somente os textos de `data/test1.xlsx`. As previsões foram
persistidas antes de abrir a planilha rotulada. As 900 linhas originais foram mantidas.

| Métrica | Resultado |
|---|---:|
| Acurácia | 45.33% |
| Acertos / erros | 408 / 492 |
| Macro-F1 | 43.83% |
| Precisão macro | 44.95% |
| Revocação macro | 45.33% |

| Classe | Precisão | Revocação | F1 | Acertos / total |
|---|---:|---:|---:|---:|
| `c1` | 45.34% | 47.00% | 46.15% | 141 / 300 |
| `c234` | 43.35% | 25.00% | 31.71% | 75 / 300 |
| `c5` | 46.15% | 64.00% | 53.63% | 192 / 300 |

Matriz de confusão: as linhas são as classes verdadeiras e as colunas, as previsões.

| Verdadeiro / predito | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 141 | 59 | 100 |
| `c234` | 101 | 75 | 124 |
| `c5` | 69 | 39 | 192 |

As contagens das classes preditas foram 311 `c1`, 173 `c234` e 416 `c5`.

<a id="comparison-of-the-three-norberto-adapters"></a>

## Comparação dos três adaptadores NorBERTo

| Adaptador | Linhas de treino | Receita da perda | Acertos / 900 | Acurácia |
|---|---:|---|---:|---:|
| Entrega original do EP1 | 16,093 | Histórica | 432 | 48.00% |
| Retreinamento completo histórico | 20,092 | Histórica | 341 | 37.89% |
| Retreinamento completo normalizado | 20,092 | Normalizada por amostra | 408 | 45.33% |

Em relação ao retreinamento completo histórico, os acertos por classe mudaram
em +85 em `c1`, −72 em `c234` e +54 em `c5`, totalizando +67. Em relação à entrega
original do EP1, as mudanças foram +23, −80 e +33. A principal fraqueza atual é
a recuperação da classe intermediária, com revocação de 25.00%.

As comparações pareadas reutilizam previsões salvas dos outros adaptadores;
nenhum foi executado novamente:

| Referência | Somente a referência acerta | Somente o normalizado acerta | p exato bilateral de McNemar |
|---|---:|---:|---:|
| Entrega original do EP1 | 145 | 121 | 0.158346 |
| Retreinamento completo histórico | 154 | 221 | 0.000634651 |

A primeira comparação não estabelece diferença significativa a 0.05 nem
equivalência. A segunda favorece nominalmente o adaptador corrigido. A inspeção
anterior de teste1 e textos repetidos limitam a interpretação confirmatória.
Estes resultados não selecionaram hiperparâmetros nem regras adicionais de ensemble.

<a id="training-behavior-after-correction"></a>

## Comportamento do treinamento após a correção

| Intervalo de épocas | Perda média normalizada por amostra | Norma média do gradiente antes da limitação |
|---|---:|---:|
| 0–0.5 | 1.12765 | 3.15449 |
| 0.5–1 | 1.09751 | 1.39068 |
| 1–1.5 | 1.07218 | 1.35278 |
| 1.5–2 | 1.05983 | 1.44972 |

A norma máxima registrada foi 10.0386. A execução não apresentou a escalada
para centenas observada no retreinamento completo histórico. As escalas brutas
de perda/gradiente também diferem porque a normalização mudou; suas razões,
sozinhas, não provam causalidade. A estabilidade observada e os acertos
recuperados são compatíveis com uma contribuição da correção nesta execução.
Uma única execução não isola todas as causas nem garante melhora geral da acurácia.

A [investigação](regression-investigation.md) é anterior a este acompanhamento
e documenta o erro, a instabilidade original, os testes e os limites causais.

<a id="execution-independent-verification-and-scope"></a>

## Execução, verificação independente e escopo

O adaptador exportado foi avaliado uma vez na RX 6650 XT, em FP32, lote 8 e
substituição ROCm `10.3.0`, usando o script existente
[`evaluate-norberto-test1.py`](../../../scripts/norberto/evaluate-test1.py).
O contêiner não tinha rede; arquivos do projeto, fontes originais e cache da
base eram somente leitura. Carregamento, tokenização e inferência levaram
**34.59 segundos**.

Um leitor ZIP/XML independente da biblioteca padrão recalculou as métricas e
verificou os 900 rótulos e IDs, contagens de confusão, distribuições de
probabilidade, logits finitos, rótulos por argmax e hashes de entrada/previsões.
Não houve treinamento nem inferência adicional nesta verificação.

Esta comparação isolada de teste1 não foi precedida por teste2 para este
adaptador. O antigo conjunto reservado integra seu treino e não é validação
inédita para ele. Os adaptadores originais e a configuração do ensemble
permanecem inalterados.

Para reproduzir no ambiente validado, com EP1 e seu cache montados:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-test1.py \
  --ep1-root /ep1 \
  --training-run results/norberto-full-normalized \
  --output-dir results/norberto-normalized-test1-another-new-run
```

As saídas brutas estão em `results/norberto-normalized-test1-20261105/`.
O [agregado público](../../results/data/norberto-normalized-test1-results.json)
contém métricas completas, procedência, verificações do treino e comparações pareadas.
