<a id="norberto-development-only-refit-frozen-holdout-validation"></a>

# Retreinamento do NorBERTo somente no desenvolvimento: avaliação do conjunto reservado congelado

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

O adaptador com perda corrigida, treinado em **16,093 respostas de desenvolvimento**,
obteve **35.66% de acurácia (1,426/3,999)** e **35.59% de macro-F1** no conjunto
reservado original do EP1. O adaptador original obteve **45.31% (1,812/3,999)**
nas mesmas linhas. Houve perda de **386 previsões corretas, ou 9.65 pontos
percentuais**. O treino foi concluído, mas o classificador resultante ficou
abaixo do adaptador original nessa validação.

<a id="training-completion-and-partition-integrity"></a>

## Conclusão do treinamento e integridade da partição

`results/norberto-development-normalized/` concluiu duas épocas fixas e as
**2,012 atualizações do otimizador** em **1,916.67 segundos (31.94 minutos)**.
Não houve interrupção pelo orçamento de tempo, validação ou seleção de
checkpoints durante o treino. O adaptador foi inicializado da base pública
fixada, independentemente do adaptador original treinado.

A divisão congelada contém 16,093 linhas de treino em 12,956 grupos e 3,999
linhas reservadas em 3,281 grupos, com **nenhuma sobreposição de linhas ou grupos**.
Todos os IDs, hashes dos textos e rótulos de treino salvos correspondem à
planilha de origem. As implementações e os hashes exportados passaram na
verificação. Os 272,643 parâmetros treináveis salvos são finitos, e os 22
tensores LoRA B contêm atualizações não nulas.

A receita versão 2 usa entropia cruzada normalizada por amostra na acumulação.
Os hiperparâmetros numéricos mantêm a receita efetiva original de duas épocas:
LoRA posto 4, alpha 8, dropout 0.05, `Wqkv`, contexto 512 com truncamento início/fim,
FP32, SDPA, taxa de aprendizado 0.001, lote 4, acumulação 4, decaimento de pesos
0.01, proporção de aquecimento 0.1 e semente 42. A revisão pública da base é
`db73446f89c96044863ea05a39f680524b84bccb`.

SHA-256 dos pesos do adaptador:

```text
2af7484a44ce6a09a9fec3d2ad36216b0749ffa7e95d03a76bddecccd70d0aec
```

<a id="holdout-metrics"></a>

## Métricas do conjunto reservado

| Métrica | Resultado |
|---|---:|
| Acurácia | 35.66% |
| Acertos / erros | 1,426 / 2,573 |
| Macro-F1 | 35.59% |
| Precisão macro | 35.69% |
| Revocação macro | 35.58% |

| Classe | Precisão | Revocação | F1 | Acertos / total |
|---|---:|---:|---:|---:|
| `c1` | 32.10% | 34.45% | 33.23% | 433 / 1,257 |
| `c234` | 37.40% | 38.32% | 37.85% | 543 / 1,417 |
| `c5` | 37.56% | 33.96% | 35.67% | 450 / 1,325 |

Matriz de confusão: as linhas são as classes verdadeiras e as colunas, as classes preditas.

| Verdadeiro / predito | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 433 | 450 | 374 |
| `c234` | 500 | 543 | 374 |
| `c5` | 416 | 459 | 450 |

As contagens das classes preditas foram 1,349 `c1`, 1,452 `c234` e 1,198 `c5`.

<a id="matched-comparison-with-the-original-ep1-adapter"></a>

## Comparação alinhada com o adaptador original do EP1

As previsões salvas do conjunto reservado do adaptador original foram reutilizadas
e conferidas com os mesmos IDs, rótulos verdadeiros e hashes de fonte/divisão.
Não houve nova inferência desse adaptador.

| Adaptador | Linhas de treino | Receita da perda | Acertos / 3,999 | Acurácia | Macro-F1 |
|---|---:|---|---:|---:|---:|
| Adaptador original do EP1 | 16,093 | Histórica | 1,812 | 45.31% | 45.31% |
| Retreinamento somente no desenvolvimento | 16,093 | Normalizada por amostra | 1,426 | 35.66% | 35.59% |

Ambos os adaptadores acertaram 756 linhas. Apenas o original acertou 1,056;
apenas o retreinado acertou 670; ambos erraram 1,517. O p-valor exato bilateral
de McNemar é **1.33114e-20**, nominalmente favorável ao original. Este conjunto
reservado é agrupado e já foi reutilizado historicamente; respostas repetidas
ou relacionadas limitam a interpretação de pares independentes.

Estes resultados são do conjunto reservado. A acurácia de 48.00% do adaptador
original em teste1 e os resultados de teste1 dos retreinamentos completos
referem-se a outra base e não são comparadores diretos aqui. Os adaptadores
completos treinaram nessas linhas reservadas, portanto não podem usá-las como
validação inédita.

<a id="training-behavior-and-interpretation"></a>

## Comportamento do treinamento e interpretação

| Intervalo de épocas | Perda média registrada | Norma média do gradiente antes da limitação | Norma máxima |
|---|---:|---:|---:|
| 0–0.5 | 1.11957 | 2.46840 | 4.64416 |
| 0.5–1 | 1.17576 | 52.90237 | 275.30652 |
| 1–1.5 | 1.13091 | 90.69721 | 186.92047 |
| 1.5–2 | 1.13462 | 114.22054 | 191.36064 |

Os resumos usam 100 amostras registradas das atualizações, 25 por intervalo.
A limitação dos gradientes permaneceu ativa. Os gradientes cresceram apesar
da normalização corrigida, e a perda não diminuiu consistentemente. A correção
resolve o erro confirmado de acumulação; não garante otimização estável nem
recupera a qualidade original em toda execução. O gatilho exato dessa regressão
não foi isolado. Esta validação não estabelece que a normalização, sozinha,
causou a queda. Veja a [investigação anterior](regression-investigation.md)
para o erro herdado e os limites da atribuição causal.

A [auditoria posterior do otimizador e da implementação computacional](instability-audit.md)
verifica o ciclo completo do Trainer corrigido em CPU e GPU, confirma derivadas
elevadas em CPU e localiza a sensibilidade no LoRA da primeira atenção. Também
explica por que alterar a escala da perda muda o regime de limitação de gradientes.

<a id="execution-and-independent-verification"></a>

## Execução e verificação independente

O [script de avaliação](../../../scripts/norberto/evaluate-holdout.py) carregou o
novo adaptador e previu os 3,999 textos reservados uma vez em FP32, lote 8, na
RX 6650 XT, com substituição ROCm `10.3.0`. Carregamento, tokenização e inferência
levaram **163.31 segundos**. O pico de VRAM alocada/reservada foi
**951.81/1,270 MiB**. O contêiner estava offline, com arquivos do projeto, fontes
do EP1 e cache da base montados somente para leitura. As previsões foram salvas
antes da leitura do gabarito.

Um leitor ZIP/XML independente da biblioteca padrão verificou todos os rótulos
verdadeiros e IDs originais. Métricas, contagens de confusão, logits finitos,
probabilidades normalizadas, previsões por argmax e hashes de fonte/divisão/
previsões/script foram conferidos sem repetir inferência nem treinamento.

Teste1 e teste2 não foram acessados. O adaptador original congelado e a
configuração do ensemble foram preservados; o resultado não selecionou substituto.

Para reproduzir na imagem ROCm validada, com EP1 em `/ep1` e seu cache Hugging
Face montado no local configurado, escolha outro caminho de saída:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-holdout.py \
  --ep1-root /ep1 \
  --training-run results/norberto-development-normalized \
  --output-dir results/norberto-development-holdout-another-new-run
```

Os resultados brutos estão em `results/norberto-development-holdout-20261105/`.
O [agregado público](../../results/data/norberto-development-holdout-results.json)
contém métricas completas, procedência, auditorias do treino, resumos dos
gradientes e comparação pareada.
