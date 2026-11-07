<a id="evaluate-the-saved-ep1-cross-validation-winner"></a>

# Avaliação do vencedor salvo da validação cruzada do EP1

Fluxo histórico desativado em 2026-11-05. O modelo selecionado é o adaptador
original da entrega do EP1, sem alterações. Os comandos abaixo descrevem o
experimento anterior; sua implementação ativa de treino/avaliação foi removida.
Cópias locais do código estão em `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Índice da documentação](../../README.md)

A validação cruzada concluída dos três finalistas do EP1 classificou
`candidate-08` em primeiro, com acurácia externa média de 43.7311%. A configuração
usa a base pública fixada do NorBERTo, entrada início/fim de 512 tokens,
LoRA posto 16/alpha 32/dropout 0.05 e taxa de aprendizado 0.0003. O adaptador
final associado está em
`../EP1/artifacts/norberto-lora-grid-final-20260930-130937/final_model/`,
com SHA-256 dos pesos
`efdb8ede4ff587fdfd453ba12b2c524a3bf5a53f06170b1d47eea5ff98b53655`.

Esse adaptador final foi inicializado da base pública e treinado nas 16,093
linhas de desenvolvimento. Ele difere dos adaptadores de cada fold. Foram
selecionadas três épocas pela mediana das épocas escolhidas internamente
`2, 3, 3`. As 3,999 linhas reservadas e seus componentes de similaridade foram
excluídos do treino. Este fluxo não executa novo treino, seleção de épocas ou
busca de parâmetros.

`scripts/norberto/evaluate-ep1-winner.py` verifica o vencedor original da validação
cruzada, o hash da configuração canônica, os registros dos folds de origem, o
treino final concluído, os hashes do adaptador/tokenizador, os IDs/grupos exatos
de treino e a base pública. Links simbólicos dos snapshots Hugging Face são
verificados pelos hashes fixados. O script carrega a implementação original de
inferência do EP1 e o ambiente FP32/SDPA, congela todos os parâmetros e prediz
o conjunto reservado e teste1 sem rótulos. Todas as previsões são persistidas
antes de construir registros rotulados. O gabarito de teste1 deve alinhar os
900 IDs originais e textos. O hash de todo o estado carregado deve permanecer
inalterado, sem gradientes nem atualizações de parâmetros.

<a id="reproduction"></a>

## Reprodução

Execute a partir do EP2, usando a imagem offline existente e um novo diretório
pai de saída. EP1, o modelo existente e o cache Hugging Face permanecem somente leitura:

```bash
output_root="$PWD/results/ep1-candidate08-new"
mkdir "$output_root"
docker run --rm --pull=never --network=none --cpus=2 \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd --device=/dev/dri \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  --security-opt label=disable --security-opt seccomp=unconfined \
  -v "$PWD:/workspace:ro" -v "$(dirname "$PWD")/EP1:/ep1:ro" \
  -v "$(dirname "$PWD")/EP1/.cache/huggingface:/workspace/.cache/huggingface:ro" \
  -v "$output_root:/evaluation" -w /workspace \
  -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e TOKENIZERS_PARALLELISM=false --entrypoint python3 pln-rocm:6.4.1 \
  -u scripts/norberto/evaluate-ep1-winner.py --ep1-root /ep1 \
  --input data/test1.xlsx \
  --gold data/test1-rotulado-para-teste-estatistico.xlsx \
  --output-dir /evaluation/run
```

O relatório registra identidades do modelo/código, versões do ambiente,
integridade do estado, tokenização, tempo de inferência e métricas. Os diretórios
separados `holdout/` e `test1/` contêm IDs originais, hashes dos textos, logits,
probabilidades e métricas. O conjunto reservado histórico já foi consultado;
seu resultado permanece exploratório. Esta avaliação não altera o modelo
publicado nem teste2.

Veja os [resultados medidos](../../results/README.md#ep1-cv-winner-candidate-08-reloaded)
para a execução concluída e sua verificação independente.

A execução concluída está em
`results/norberto-ep1-candidate08-evaluation-20261105/evaluation/`:
1,439/3,999 acertos no conjunto reservado (35.98% de acurácia, 35.31% de macro-F1)
e 313/900 em teste1 (34.78% de acurácia, 34.30% de macro-F1). Uma leitura
independente da fonte verificou todas as linhas e reconstruiu as probabilidades
dos logits salvos. O diretório pai preserva o verificador e seu relatório;
também foram criados `holdout/comparison.csv` e `test1/comparison.csv`, com
gabarito, previsões, acertos, probabilidades e previsões do modelo original
entregue no EP1. O modelo original mantém acurácia de 45.31% no conjunto
reservado e 48.00% em teste1; candidate-08 não foi promovido.
