# Evaluate the saved EP1 cross-validation winner

Historical workflow retired on 2026-11-05. The unchanged original EP1 delivery
adapter is the selected model. Commands below describe the former experiment;
its active training/evaluation implementation has been removed. Local source
snapshots are preserved in `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Documentation index](../../README.md)

EP1's completed three-finalist cross-validation ranked `candidate-08` first,
with mean outer accuracy 43.7311%. Its configuration uses the pinned public
NorBERTo base, 512-token head/tail input, LoRA rank 16/alpha 32/dropout 0.05,
and learning rate 0.0003. The associated final adapter is
`../EP1/artifacts/norberto-lora-grid-final-20260930-130937/final_model/`,
with weights SHA-256
`efdb8ede4ff587fdfd453ba12b2c524a3bf5a53f06170b1d47eea5ff98b53655`.

This final adapter was initialized from the public base and trained on all
16,093 development rows. It is distinct from each fold adapter. Three epochs
were selected by the median of inner-selected epochs `2, 3, 3`. The 3,999
holdout rows and their similarity components were excluded from training.
No new training, epoch selection, or parameter search occurs in this workflow.

`scripts/norberto/evaluate-ep1-winner.py` checks the original CV winner,
canonical configuration digest, source fold records, completed final training,
adapter/tokenizer hashes, exact training row IDs/groups, and the public base.
Hugging Face snapshot symlinks are verified against pinned file hashes. It
loads the original EP1 inference implementation and FP32/SDPA runtime, freezes
all parameters, and predicts holdout and test1 without labels. All predictions
are persisted before labeled records are constructed. Test1 gold must align
with all 900 original row IDs and response texts. The complete loaded model
state hash must remain unchanged, with no parameter gradients or updates.

## Reproduction

Run from EP2 using the existing offline image and a new output parent. EP1,
the existing model, and the Hugging Face cache remain read-only:

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

The report records model/source fingerprints, runtime versions, state integrity,
tokenization, inference time, and metrics. Separate `holdout/` and `test1/`
directories contain original row IDs, text hashes, logits, probabilities, and
metrics. The historical holdout has been consulted before; its score remains
exploratory. This evaluation does not change the published model or test2.

See the [measured results](../../results/README.md#ep1-cv-winner-candidate-08-reloaded)
for the completed run and independent verification.

The completed run is
`results/norberto-ep1-candidate08-evaluation-20261105/evaluation/`:
1,439/3,999 correct on holdout (35.98% accuracy, 35.31% macro-F1) and 313/900
on test1 (34.78% accuracy, 34.30% macro-F1). Independent source parsing verified
all rows and reconstructed probabilities from saved logits. The parent run
directory preserves that verifier and its report; it also created
`holdout/comparison.csv` and `test1/comparison.csv`, with gold labels,
predictions, correctness, probabilities, and the original delivered EP1
model's predictions. The original model remains at 45.31% holdout and 48.00%
test1 accuracy; candidate-08 was not promoted.
