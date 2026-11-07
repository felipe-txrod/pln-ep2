<a id="frozen-norberto-512-adapter"></a>

# Adaptador congelado NorBERTo-512

`final_model/adapter_model.safetensors` contém adaptador LoRA final EP1 e
cabeça treinada. Tokenizadores e configurações também são versionados.
Os pesos-base públicos ficam separados, fora do Git.

Este é o NorBERTo selecionado para EP2. Adaptador original EP1, cabeça,
tokenizador e inferência são mantidos sem correção de perda ou retreinamento.
Adaptadores experimentais permanecem resultados históricos.

- Base: `Itau-Unibanco/NorBERTo-base`.
- Revisão: `db73446f89c96044863ea05a39f680524b84bccb`.
- LoRA: posto 4, alpha 8, dropout 0.05, módulos-alvo `Wqkv`.
- Contexto: 512 tokens, truncamento início/fim, preservando maiúsculas/minúsculas.
- Treino final: duas épocas em 16,093 respostas de desenvolvimento.
- Inferência: FP32, SDPA, `reference_compile=False`.

SHA-256 do adaptador:

```text
d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385
```

`base_reference.json` registra revisão-base e hashes necessários. Após clonar,
execute `python3 scripts/models/prepare-base.py` para restaurar/verificar
arquivos ausentes sem carregar ou retreinar. O Git ignora os pesos-base grandes.

Itaú Unibanco e ICTi publicaram a base sob **CC BY-NC-SA 4.0**, conforme a
[ficha fixada do modelo](https://huggingface.co/Itau-Unibanco/NorBERTo-base/blob/db73446f89c96044863ea05a39f680524b84bccb/README.md).
O adaptador acrescenta a tarefa supervisionada EP1 ao modelo original.
Preserve atribuição e [termos da licença-base](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Veja o [guia de inferência](../../src/norberto_classifier/README.md).
