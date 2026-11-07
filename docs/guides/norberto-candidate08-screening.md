<a id="exploratory-test1-evaluation-of-grid-screening-candidate-08"></a>

# Avaliação exploratória em Teste1 do candidate-08 da triagem da grade

[Índice da documentação](../README.md) · [Índice dos scripts](../../scripts/README.md)

Este fluxo foi preparado a pedido do usuário para testar o adaptador salvo
**durante a triagem original da grade**. Não retreina nem altera o NorBERTo
selecionado ou a decisão principal de votação suave.

A origem exata é:

```text
../EP1/artifacts/norberto-lora-grid-20260929-170333/screening/candidate-08/best_adapter/
```

Usou 12,982 linhas de treino e 3,111 de validação interna, selecionando a época
2 com acurácia de validação 44.2944%. O LoRA tem posto 16, alpha 32 e taxa
0.0003. O SHA-256 dos pesos do adaptador é
`8e9c8332866fe3c2649382bb6fc4f9970b76471364516e58b7f690bef0e066ca`.
É distinto dos três adaptadores posteriores dos folds e do retreinamento final
envolvendo todo o desenvolvimento (`efdb8ede…`), cuja acurácia no conjunto
reservado foi 35.98%.

Execute no EP2 com a imagem ROCm offline já disponível:

```bash
bash scripts/experiments/norberto/candidate08-test1.sh \
  --output-dir results/norberto-candidate08-screening-test1
```

Use novo diretório de saída. `NORBERTO_IMAGE` pode selecionar uma imagem
compatível. O script monta EP1 somente para leitura, desativa a rede e mantém
o contrato de inferência FP32/SDPA, lote 8, contexto 512 com início/fim e
preservação de maiúsculas/minúsculas. A base pública é verificada pelo pacote
congelado existente do NorBERTo; o adaptador experimental tem verificações
estritas próprias de procedência da triagem/configuração/pesos. Não se adiciona
um desvio genérico das verificações imutáveis do modelo selecionado.

As 900 previsões geradas somente dos textos, IDs, hashes e probabilidades são
persistidas antes de ler a planilha rotulada de Teste1. O diretório contém
`predictions.jsonl`, `metrics.json` e `run.json`; as métricas incluem acurácia,
acertos, macro-F1, resultados por classe e matriz de confusão. A comparação é
retrospectiva e exploratória; não dispara promoção, ajuste de pesos nem nova
entrega de Teste2.

Para verificar o adaptador e as montagens sem carregar o modelo nem ler Teste1:

```bash
bash scripts/experiments/norberto/candidate08-test1.sh --preflight
```

A preparação verificou o checkpoint original e seus metadados de origem.
Não houve predição de 900 linhas nem medição de acurácia durante a preparação.
