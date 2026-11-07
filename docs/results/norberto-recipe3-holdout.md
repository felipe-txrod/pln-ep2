<a id="norberto-recipe-3-completed-training-and-holdout-evaluation"></a>

# NorBERTo receita 3: treinamento e avaliação do conjunto reservado concluídos

[Índice da documentação](../README.md)

O adaptador compatível com o otimizador obteve **44.81% de acurácia
(1,792/3,999)** e **44.81% de macro-F1** no conjunto reservado congelado do EP1.
Recupera 366 acertos frente à receita 2 e fica 20 abaixo do original nas mesmas
linhas. A escalada anterior dos gradientes não reapareceu nesta execução.

<a id="training-verification-and-stability"></a>

## Verificação do treinamento e estabilidade

`results/norberto-development-recipe3/` concluiu **duas épocas e 2,012
atualizações** em **16,093 linhas de desenvolvimento**, em **1,879.41 segundos
(31.32 minutos)**. Preservou as 3,999 linhas reservadas originais, sem
sobreposição de linhas/grupos. Não usou validação ou teste no treino nem
atingiu o orçamento de tempo. Os 272,643 parâmetros exportados são finitos,
os 22 tensores LoRA B contêm atualizações, e hashes exportados coincidem com
o manifesto. Os hashes registrados da implementação coincidem com as fontes atuais.

A receita 3 mantém a perda reportada normalizada por exemplo e escala
explicitamente os gradientes retropropagados pela acumulação 4, preservando
as unidades do otimizador do EP1 para limitação e AdamW. Hiperparâmetros e
divisão fixa não mudam. A [correção](../archive/norberto/recipe3-optimizer-units.md)
explica a diferença para a receita 2 e o tratamento dos microlotes finais desiguais.

| Intervalo de épocas | Perda média registrada | Norma média do gradiente do otimizador | Norma média por amostra | Norma máxima do otimizador |
|---|---:|---:|---:|---:|
| 0–0.5 | 1.12420 | 11.50810 | 2.87702 | 22.58407 |
| 0.5–1 | 1.08497 | 8.09402 | 2.02351 | 17.90670 |
| 1–1.5 | 1.04843 | 7.43696 | 1.85924 | 12.93741 |
| 1.5–2 | 1.02875 | 8.12732 | 2.03183 | 11.67706 |

São resumos de 25 amostras por intervalo. Normas em unidades do otimizador
são anteriores à limitação; as por amostra as dividem por 4. A norma máxima
por amostra foi **5.64602**, contra **275.30652** na receita 2. A última norma
do otimizador registrada foi 6.21800, ou 1.55450 por amostra. O pico de VRAM
alocada/reservada no treino foi **3,259.04/3,800 MiB**.

As **50 normas registradas até a atualização 1,000** coincidem exatamente com
o treino original do EP1. O diagnóstico separado de 600 atualizações também
coincidiu nas 30 normas e taxas registradas. Em 540, a norma por amostra da
receita 3 era 1.94616; a da receita 2, 253.14911. As normas do otimizador
original e da receita 3 eram ambas 7.78463. Isso sustenta a incompatibilidade
de unidades como explicação da divergência inicial. Não estabelece igualdade
bit a bit de todos os tensores intermediários nem resultados idênticos após
corrigir o peso da cauda na fronteira da primeira época.

SHA-256 dos pesos do adaptador:

```text
641760227c2d5862f56fa3251d7e8722422a37d4de30a76d0623dbe52325e04c
```

A base continua `Itau-Unibanco/NorBERTo-base`, revisão
`db73446f89c96044863ea05a39f680524b84bccb`.

<a id="frozen-holdout-metrics"></a>

## Métricas do conjunto reservado congelado

| Métrica | Resultado |
|---|---:|
| Acurácia | 44.81% |
| Acertos / erros | 1,792 / 2,207 |
| Macro-F1 | 44.81% |
| Precisão macro | 44.96% |
| Revocação macro | 44.81% |

| Classe | Precisão | Revocação | F1 | Acertos / total |
|---|---:|---:|---:|---:|
| `c1` | 46.46% | 41.29% | 43.72% | 519 / 1,257 |
| `c234` | 41.10% | 42.34% | 41.71% | 600 / 1,417 |
| `c5` | 47.33% | 50.79% | 49.00% | 673 / 1,325 |

Matriz de confusão: as linhas são as classes verdadeiras e as colunas, as previsões.

| Verdadeiro / predito | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 519 | 440 | 298 |
| `c234` | 366 | 600 | 451 |
| `c5` | 232 | 420 | 673 |

As contagens preditas foram 1,117 `c1`, 1,460 `c234` e 1,422 `c5`.

<a id="comparison-on-identical-held-out-rows"></a>

## Comparação nas mesmas linhas reservadas

| Adaptador | Acertos / 3,999 | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Original EP1 | 1,812 | 45.31% | 45.31% |
| Receita 2, gradientes corrigidos da perda média | 1,426 | 35.66% | 35.59% |
| Receita 3, ponderação e unidades do otimizador corrigidas | 1,792 | 44.81% | 44.81% |

A receita 3 ganha **366 acertos (+9.15 pontos percentuais)** frente à 2 e perde
**20 (-0.50 pontos)** frente à original. Previsões de referência salvas foram
reutilizadas; nenhum adaptador antigo foi executado novamente.

| Referência | Ambos acertam | Só a referência acerta | Só a receita 3 acerta | Ambos erram | p exato bilateral de McNemar |
|---|---:|---:|---:|---:|---:|
| Original EP1 | 1,723 | 89 | 69 | 2,118 | 0.130386 |
| Recipe 2 | 753 | 673 | 1,039 | 1,534 | 8.36777e-19 |

A primeira comparação não estabelece diferença significativa a 0.05 nem
equivalência. A segunda favorece nominalmente a receita 3. O conjunto já foi
usado historicamente, e respostas relacionadas/repetidas limitam a
independência dos pares. Recuperação e estabilidade medidas sustentam a
correção nesta execução, sem garantia geral de acurácia.

<a id="evaluation-integrity-and-reproduction"></a>

## Integridade da avaliação e reprodução

O adaptador final exportado foi avaliado uma vez, em FP32 e lote 8 na RX 6650
XT, com substituição ROCm `10.3.0`. Carregamento, tokenização e inferência
levaram **156.13 segundos**. Entradas, fontes, cache da base e pesos foram
montados somente para leitura em contêiner offline. Todas as linhas reservadas
foram mantidas; as previsões foram salvas antes da leitura dos rótulos.

Um leitor ZIP/XML independente da biblioteca padrão verificou cada rótulo,
hash e ID, e recalculou matriz e métricas por classe. Foram conferidos logits
finitos, softmax, argmax e hashes das entradas/previsões/script. A verificação
não repetiu inferência nem treinou modelos.

Teste1/teste2 não foram acessados, e modelo original e ensemble foram
preservados. São métricas do conjunto reservado; a acurácia original de 48.00%
em teste1 se refere a outra base.

Na imagem ROCm validada, com EP1/cache montados, use novo destino para reproduzir:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-holdout.py \
  --ep1-root /ep1 \
  --training-run results/norberto-development-recipe3 \
  --output-dir results/norberto-recipe3-holdout-another-new-run
```

Os resultados brutos estão em `results/norberto-recipe3-holdout-20261105/`.
O [agregado público](data/norberto-recipe3-holdout-results.json) inclui métricas
completas, identidades, auditorias, testes de regressão e comparações pareadas.
