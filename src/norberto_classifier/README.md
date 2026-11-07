<a id="frozen-ep1-norberto-inference-package"></a>

# Pacote congelado de inferência do NorBERTo do EP1

Este é o NorBERTo selecionado para EP2. Expõe o modelo final inalterado já
entregue no EP1. Correções experimentais de treino/perda e variante de cabeça
logística foram desativadas em 2026-11-05. Não tem comando de treinamento,
seleção de hiperparâmetros ou avaliação.

<a id="preserved-model"></a>

## Modelo preservado

| Propriedade | Valor |
|---|---|
| Base | `Itau-Unibanco/NorBERTo-base` |
| Revisão | `db73446f89c96044863ea05a39f680524b84bccb` |
| Artefato original | `EP1/artifacts/norberto-lora-final-ctx512-20260928` |
| Adaptador | LoRA posto 4, alpha 8, dropout 0.05, módulos `Wqkv` |
| Entrada | Só `resp_text`, preservando maiúsculas/minúsculas |
| Contexto | 512 tokens, truncamento início/fim |
| Precisão e atenção | FP32, SDPA, `reference_compile=False` |
| Ordem de classes | `c1`, `c234`, `c5` |
| Inferência | Lote 8, duas threads, CPU por padrão |

SHA-256 oficial do adaptador:

```text
d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385
```

O orçamento aprovado permitia três épocas; o treino final usou duas, escolhidas
pela mediana das melhores entre folds. `approved_config.json` contém orçamento,
e `config.json`, parâmetros finais. `run.json` original registra a distinção.
O treino usou 16,093 respostas e excluiu as 3,999 reservadas.

<a id="files-and-integrity"></a>

## Arquivos e integridade

O pacote padrão é [`models/norberto/`](../../models/norberto/README.md):

```text
norberto/
  manifest.json
  frozen_manifest.json
  base_reference.json
  approved_config.json
  config.json
  run.json
  base_model/
    config.json
    model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
  final_model/
    adapter_config.json
    adapter_model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
```

Adaptador, cabeça, tokenizador, configuração e metadados técnicos são versionados.
`base_model/model.safetensors` é ignorado; `base_reference.json` fixa revisão
pública e SHA-256. Após clonar:

```bash
python3 scripts/models/prepare-base.py
```

Restaura/verifica ausentes sem carregar ou treinar modelo. `--verify-only`
confere arquivos existentes sem rede. O pacote local completo ocupa cerca de
607 MB. A base pública usa CC BY-NC-SA 4.0; atribuição e fontes estão no
[pacote](../../models/norberto/README.md).

O [manifesto congelado das fontes](frozen_manifest.json) identifica cada arquivo
por SHA-256. Antes de carregar, o pacote rejeita alterações, revisão-base
ou adaptador diferentes. README gerado pelo PEFT e `training_args.bin` são
omitidos por serem desnecessários à inferência; `run.json` histórico mantém
os registros originais.

`NORBERTO_ARTIFACT_DIR` ou `--artifact-dir` seleciona outro pacote completo.
Sem pacote publicado, `artifacts/norberto-ep1/` permanece alternativa histórica.
Após preparar a base, inferência não exige EP1 nem acesso Hugging Face.
Pacotes de modelos são excluídos do contexto de construção Docker.

<a id="usage"></a>

## Uso

O ambiente exige PyTorch apropriado para CPU/ROCm, Transformers 4.57.1,
PEFT 0.17.1, Accelerate 1.10.1 e dependências comuns EP2. O extra `norberto`
de `pyproject.toml` declara as três bibliotecas de adaptação, mas não
seleciona a distribuição PyTorch.

Na raiz do EP2 com dependências instaladas:

```bash
PYTHONPATH=src python -m norberto_classifier predict \
  --input data/test2.xlsx \
  --output-dir results/norberto-ep1-test2 \
  --device cpu
```

A CLI lê só textos, ignora rótulos verdadeiros e exige novo diretório. Grava
`predictions.jsonl` com IDs, hashes, classes e probabilidades e `run.json`
com procedência/configuração. Não altera entradas nem gera XLSX de entrega.
GPU exige `--device gpu` explícito.

```python
from norberto_classifier import NorbertoClassifier

classifier = NorbertoClassifier(device="cpu")
predictions = classifier.predict(["Example public response."])
label = predictions[0].label
probabilities = predictions[0].probabilities
```

Importar o pacote ou construir o classificador não carrega PyTorch, pesos ou
tokenizador. O carregamento ocorre por `load()` ou `predict()` com textos.
Parâmetros ficam congelados; inferência usa `torch.inference_mode()`.

<a id="limitations"></a>

## Limitações

Normalização, truncamento e carregamento preservam os módulos EP1 identificados
no manifesto. A restauração foi verificada nas 900 linhas de teste1:
todos os rótulos coincidem com a entrega original (432/900, 48.00%) e a
diferença máxima de probabilidade contra o CSV histórico é 8.60e-8.
Adaptador, tokenizador e configurações correspondem ao EP1. Artefatos em
`results/norberto-rollback-20261105/`; veja o
[registro da restauração](../../docs/results/data/norberto-ep1-restoration.json).
Implementações originais de inferência/tokenização e pesos não mudaram na limpeza.

Nova inferência não substitui previsões oficiais EP1 nas comparações
estatísticas. Siga o protocolo EP2: congele o método, entregue teste2 e
depois compare teste1 com EP1.
