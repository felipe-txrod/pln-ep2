<a id="full-corpus-norberto-evaluation-on-labeled-test1"></a>

# Avaliação do NorBERTo treinado no corpus completo em teste1 rotulado

[Índice da documentação](../../README.md) · Registro histórico do experimento; use os guias para comandos atuais.

A [investigação posterior restrita ao treino](regression-investigation.md)
documenta instabilidade na otimização, um erro reproduzido na normalização da
perda e a correção testada para treinamentos futuros. Os resultados históricos
abaixo permanecem inalterados.

O adaptador NorBERTo retreinado nas **20,092 respostas de treino do EP1** obteve
**37.89% de acurácia (341/900)** e **36.47% de macro-F1** em teste1. A entrega
original do EP1 obteve 48.00% (432/900). O retreinamento no corpus completo foi
concluído tecnicamente, mas perdeu **91 previsões corretas**, ou **10.11 pontos
percentuais**, em teste1. O adaptador original permanece inalterado.

<a id="training-and-artifact-verification"></a>

## Verificação do treinamento e dos artefatos

A execução em `results/norberto-full-training/` concluiu as duas épocas
configuradas e as **2,512 atualizações do otimizador** em **2,389.49 segundos
(39.82 minutos)**. Não foi interrompida pelo orçamento de 75 minutos. A
configuração efetiva corresponde à receita original de duas épocas do EP1:
LoRA posto 4/alpha 8/dropout 0.05, `Wqkv`, contexto 512 com truncamento início/fim,
FP32, SDPA, taxa de aprendizado 0.001, lote 4, acumulação 4, decaimento de pesos
0.01, aquecimento 0.1 e semente 42.

Todos os 20,092 IDs, hashes de texto e rótulos das linhas salvas de treino foram
conferidos com a planilha original do EP1. O antigo conjunto reservado foi
incluído; teste1/teste2 não entraram no treino, e não houve validação interna.
A configuração herdada mantém descritores históricos da divisão; o executor
com os dados completos não aplica essa divisão.

O novo adaptador tem **272,643 parâmetros finitos**, 22 tensores LoRA B não nulos
e os pesos e vieses treinados da classificação. Seu SHA-256 é:

```text
ea0bd1ef02516e0de768dfb8d76313c3a8cacd25d43d9f4431c5c35a4b2c86be
```

Todos os hashes dos arquivos exportados foram verificados. O checksum da base
pública NorBERTo corresponde à revisão fixada
`db73446f89c96044863ea05a39f680524b84bccb`. Uma verificação separada de carregamento
somente em CPU comparou, por igualdade exata, os **46 tensores carregados do
adaptador** com o arquivo safetensors. Todos coincidiram, inclusive a cabeça de
classificação; o mapeamento de classes também estava correto. Essa verificação
não realizou inferência nem treinamento.

<a id="test1-results"></a>

## Resultados de Teste1

A inferência leu somente os textos de `data/test1.xlsx`. As previsões foram
salvas antes de abrir o gabarito `test1-rotulado-para-teste-estatistico.xlsx`.
As 900 linhas, seus textos e IDs originais foram preservados e alinhados.

| Métrica | Resultado |
|---|---:|
| Acurácia | 37.89% |
| Acertos / erros | 341 / 559 |
| Macro-F1 | 36.47% |
| Precisão macro | 37.27% |
| Revocação macro | 37.89% |

| Classe | Precisão | Revocação | F1 | Acertos / total |
|---|---:|---:|---:|---:|
| `c1` | 34.57% | 18.67% | 24.24% | 56 / 300 |
| `c234` | 38.48% | 49.00% | 43.11% | 147 / 300 |
| `c5` | 38.76% | 46.00% | 42.07% | 138 / 300 |

Matriz de confusão: as linhas são os rótulos verdadeiros e as colunas, as previsões.

| Verdadeiro / predito | `c1` | `c234` | `c5` |
|---|---:|---:|---:|
| `c1` | 56 | 129 | 115 |
| `c234` | 50 | 147 | 103 |
| `c5` | 56 | 106 | 138 |

O modelo previu `c1` 162 vezes, `c234` 382 vezes e `c5` 356 vezes.
A maior perda em relação à entrega original ocorreu em `c1`:

| Classe verdadeira | Acertos do EP1 original | Acertos do retreinamento completo | Diferença |
|---|---:|---:|---:|
| `c1` | 118 | 56 | −62 |
| `c234` | 155 | 147 | −8 |
| `c5` | 159 | 138 | −21 |

<a id="paired-comparison-and-interpretation"></a>

## Comparação pareada e interpretação

As previsões originais do EP1 foram reutilizadas sem executar o modelo original:
ambos acertaram 192 linhas, apenas o original acertou 240, apenas o retreinado
completo acertou 149 e nenhum acertou 319. O teste exato bilateral de McNemar
resultou em **p = 0.000004585**, favorecendo o modelo original. Textos repetidos
e a inspeção anterior de teste1 limitam sua interpretação como comparação
confirmatória independente.

| Modelo salvo | Acertos / 900 | Acurácia |
|---|---:|---:|
| SVM | 445 | 49.44% |
| Regressão logística | 444 | 49.33% |
| NorBERTo original do EP1 | 432 | 48.00% |
| Qwen3-1.7B QLoRA | 420 | 46.67% |
| NorBERTo retreinado no corpus completo | 341 | 37.89% |

As verificações confirmam o uso do corpus e da receita solicitados e o
carregamento correto dos novos pesos. Não estabelecem a causa da queda de
desempenho. Mais respostas de treino alteram o número de atualizações, a ordem
de amostragem e a trajetória da otimização, mesmo com hiperparâmetros fixos.
Esta execução não sustenta atribuição causal a uma mudança específica. Durante
a avaliação não houve retreinamento, busca de checkpoints, ajuste de limiares,
substituição de modelo ou alteração do ensemble.

Esta é uma comparação isolada, sem execução anterior de teste2 para este
adaptador. O conjunto reservado histórico agora integra seu treino e não pode
ser validação inédita. As métricas de teste1 não devem orientar uma nova regra
se o mesmo teste for apresentado como avaliação final ainda não consultada.

<a id="execution-and-reproduction"></a>

## Execução e reprodução

O adaptador foi avaliado uma vez na RX 6650 XT, usando a construção original de
inferência do EP1 e lote 8. A substituição de versão do ROCm foi `10.3.0`.
O contêiner não tinha rede; arquivos do projeto, fontes do EP1 e cache da base
foram montados somente para leitura. Carregamento, tokenização e inferência
levaram **34.47 segundos**. O pico de VRAM alocada/reservada pelo PyTorch foi
**951.81 / 1,136 MiB**.

O ponto de entrada reutilizável é
[`scripts/norberto/evaluate-test1.py`](../../../scripts/norberto/evaluate-test1.py).
Exige o código original do EP1 e o cache da base fixada. Dentro do contêiner
ROCm validado, com EP1 montado em `/ep1` e seu cache Hugging Face em
`/workspace/.cache/huggingface`:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/norberto/evaluate-test1.py \
  --ep1-root /ep1 \
  --training-run results/norberto-full-training \
  --output-dir results/norberto-full-test1-another-new-run
```

Um leitor ZIP/XML independente, baseado na biblioteca padrão, conferiu todos
os rótulos verdadeiros, contagens de confusão, resultados por classe, somas das
probabilidades, previsões por logits/argmax e hashes das fontes/previsões.
Os relatórios brutos estão em `results/norberto-full-test1-20261105-003353/`;
o agregado público completo é
[`norberto-full-test1-results.json`](../../results/data/norberto-full-test1-results.json).
