<a id="predict-and-score-test1-with-the-original-ep1-norberto"></a>

# Predição e avaliação de teste1 com o NorBERTo original do EP1

[Índice da documentação](../README.md)

O modelo selecionado é o adaptador original da entrega do EP1 em
`models/norberto/`, sem alterações: LoRA posto 4/alpha 8, revisão pública da
base fixada, entrada início/fim de 512 tokens, FP32/SDPA e lote 8. O treino
final original usou duas épocas em 16,093 respostas de desenvolvimento.
Este fluxo não treina nem corrige a perda. Fontes do EP1 vizinho e caches
Hugging Face não são necessários após restaurar o pacote. `NORBERTO_IMAGE`
seleciona outra imagem local compatível.

Execute no EP2 usando o ambiente offline existente `pln-rocm:6.4.1`:

```bash
bash scripts/norberto/test1.sh predict \
  --input data/test1.xlsx \
  --output-dir results/norberto-ep1-test1-new
```

O comando prediz as 900 respostas somente a partir dos textos, salva IDs,
hashes, rótulos e probabilidades e registra as identidades do adaptador original
e do manifesto. Não lê gabaritos. Antes de carregar, o pacote valida base,
adaptador, tokenizador, configuração e metadados imutáveis de treino.

Avalie as previsões persistidas separadamente, em CPU:

```bash
bash scripts/norberto/test1.sh compare \
  --prediction-dir results/norberto-ep1-test1-new \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir results/norberto-ep1-test1-metrics-new
```

A comparação exige a identidade original do adaptador EP1, arquivo de previsões
íntegro e alinhamento exato de linhas/textos. Não carrega modelo e grava
acurácia, macro-F1, precisão, revocação, relatórios de classes e matriz de
confusão. Ambas as etapas exigem novos diretórios. A interface removida
`--training-run` já não aceita adaptadores experimentais.

O caminho de inferência restaurado foi verificado nas 900 linhas contra a
entrega original do EP1: todos os rótulos coincidem, com 432 acertos (48.00%)
e 47.86% de macro-F1. Os artefatos de verificação estão em
`results/norberto-rollback-20261105/`. A configuração identifica o modelo
original por `artifact_manifest_sha256` e lê as previsões selecionadas de
`results/norberto-ep1-test1` (teste1) ou `results/norberto-ep1-test2` (teste2).
A inferência em Teste2 é uma operação separada:

```bash
PYTHONPATH=src python3 -m norberto_classifier predict \
  --input data/test2.xlsx --device gpu \
  --output-dir results/norberto-ep1-test2
```

Nenhuma predição ou entrega de teste2 foi realizada durante a restauração.
