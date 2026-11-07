<a id="qwen3-17b-evaluation-on-labeled-test1"></a>

# Avaliação do Qwen3-1.7B em teste1 rotulado

[Índice da documentação](../README.md)

Este relatório descreve o adaptador histórico `qwen-fast-training` e sua divisão
16,010/4,082. O [novo padrão Qwen](qwen-ep1-model.md) usa a divisão exata do
EP1 16,093/3,999 e fica em `models/qwen/adapter/`. Os resultados abaixo não
representam esse novo modelo selecionado.

O adaptador QLoRA NF4 concluído obteve **46.67% de acurácia (420/900)** e
**46.44% de macro-F1** nas linhas originais de teste1. Planilha, IDs e hashes
foram preservados. A inferência consumiu só textos de `data/test1.xlsx`;
o gabarito `test1-rotulado-para-teste-estatistico.xlsx` foi lido após persistir
as previsões. Não houve treino nem seleção de parâmetros.

<a id="training-completion-and-adapter-integrity"></a>

## Conclusão do treino e integridade do adaptador

`results/qwen-fast-training/` concluiu três épocas e 3,003 atualizações em
11,222.19 segundos, cerca de 3 h 7 min. Nenhum passo foi ignorado por estouro
FP16. Os 3,217,408 parâmetros treináveis salvos eram finitos; os 112 tensores
LoRA B tinham atualizações não nulas. A cabeça de classificação treinada
estava presente.

| Época | Acurácia de validação interna | Macro-F1 de validação interna |
|---|---:|---:|
| 1 | 43.53% | 34.96% |
| 2 | 42.28% | 39.94% |
| 3 | **45.84%** | **45.52%** |

O adaptador selecionado foi `checkpoint-3003`, escolhido por acurácia de
validação interna. Os pesos coincidem byte a byte com o adaptador exportado,
com SHA-256:

```text
099fa94ccd7e9d40c7a0e0110154e64c5b5bc7430d71c1f0063bee589e935b50
```

A revisão-base é `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` de
`Qwen/Qwen3-1.7B`. O treino usou 16,010 respostas e reservou 4,082 para validação
interna agrupada. Não houve retreinamento nas 20,092 respostas completas.

Essa partição difere do conjunto reservado fixo do EP1. Das 3,999 linhas
reservadas do EP1, só 771 (19.28%) também estão na validação Qwen; as outras
3,228 treinaram Qwen. Inversamente, 3,311 linhas de validação Qwen pertencem
ao desenvolvimento do EP1. Os dois treinaram em 12,782 linhas comuns.
Conjuntos reservados de tamanhos próximos não implicam observações iguais.

Qwen agrupa textos idênticos com entidades HTML decodificadas, espaços
normalizados e minúsculas; EP1 usa componentes congelados de similaridade.
Por esse último agrupamento, 301 componentes cruzam o limite treino/validação
do Qwen, cobrindo 798 linhas de validação. Grupos exatos de texto normalizado
continuam disjuntos nele. Assim, o conjunto reservado EP1 não é todo inédito
para o ensemble atual. A [auditoria da sobreposição](data/qwen-ep1-split-overlap.json)
registra hashes e contagens; não houve treino, inferência nem alteração de
modelo/padrão.

<a id="test1-metrics"></a>

## Métricas de Teste1

| Métrica | Resultado |
|---|---:|
| Acurácia | 46.67% |
| Acertos / erros | 420 / 480 |
| Macro-F1 | 46.44% |
| Precisão macro | 46.43% |
| Revocação macro | 46.67% |

| Classe | Precisão | Revocação | F1 | Acertos / total |
|---|---:|---:|---:|---:|
| `c1` | 50.00% | 49.00% | 49.49% | 147 / 300 |
| `c234` | 41.20% | 36.67% | 38.80% | 110 / 300 |
| `c5` | 48.08% | 54.33% | 51.02% | 163 / 300 |

Matriz de confusão: as linhas são os rótulos verdadeiros e as colunas, as previsões.

| Verdadeiro / predito | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 147 | 85 | 68 |
| `c234` | 82 | 110 | 108 |
| `c5` | 65 | 72 | 163 |

<a id="descriptive-comparison-with-saved-models"></a>

## Comparação descritiva com modelos salvos

| Modelo | Acertos / 900 | Acurácia |
|---|---:|---:|
| SVM | 445 | 49.44% |
| Regressão logística | 444 | 49.33% |
| Original EP1 NorBERTo submission | 432 | 48.00% |
| Qwen3-1.7B QLoRA | 420 | 46.67% |

Qwen teve 12 acertos a menos que a entrega original NorBERTo, diferença de
−1.33 pontos percentuais. Ambos acertaram 310 linhas; somente NorBERTo,
122; somente Qwen, 110. Ambos erraram 358. O p-valor exato bilateral de
McNemar foi **0.470258**, sem diferença significativa a 0.05 nem equivalência
estabelecida. Textos repetidos limitam a independência dos pares.

As previsões originais NorBERTo foram reutilizadas, sem nova execução.
O retreinamento completo posterior é outro modelo, não representado por esta
referência. Resultados SVM e logísticos vêm da
[avaliação salva em CPU](lexical-test1.md).

Esta avaliação é isolada. Não houve execução/entrega anterior de teste2 para
o adaptador, portanto não comprova a sequência prescrita teste2 e depois
teste1. Teste1 também já foi inspecionado. Não use estes resultados para
ajustar membros, limiares, prompts ou arbitragem enquanto apresentar o mesmo
teste como avaliação final inédita.

<a id="execution-verification-and-reproduction"></a>

## Execução, verificação e reprodução

A inferência recarregou o adaptador exportado na RX 6650 XT, ambiente validado
ROCm 6.4 e `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Uma nova verificação dos núcleos
NF4 passou. O contêiner não tinha rede, o projeto era somente leitura e só
o novo diretório de avaliação permitia escrita. O modelo usou seus lotes
existentes de uma resposta. Carregamento/inferência levaram **235.75 segundos**;
o pico de VRAM PyTorch alocada/reservada foi 1,640.87 / 1,746 MiB.

Um leitor ZIP/XML independente da biblioteca padrão conferiu os 900 rótulos.
IDs, contagens de confusão, métricas por classe, somas de probabilidades,
logits finitos, argmax e hashes de fonte/previsões coincidiram com o relatório.
O treino original e o adaptador permaneceram inalterados.

O ponto de entrada reutilizável é
[`scripts/qwen/evaluate-test1.py`](../../scripts/qwen/evaluate-test1.py).
No ambiente ROCm validado, com o cache da base original disponível:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
PYTHONPATH=.cache/qwen-runtime:src \
python3 scripts/qwen/evaluate-test1.py \
  --training-run results/qwen-fast-training \
  --output-dir results/qwen-test1-another-new-run
```

Use novo diretório. Reexecutar faz nova inferência; os resultados salvos
bastam para análises adicionais sem executar o modelo novamente.

Métricas públicas completas, configuração, versões, hashes de procedência e
comparação pareada estão em [`qwen-test1-results.json`](data/qwen-test1-results.json).
Previsões e relatórios brutos ficam em
`results/qwen-test1-evaluation-20261104-225455-verified/qwen/`.
