# Exploratory Test1 evaluation of grid-screening candidate-08

[Documentation index](../README.md) · [Script index](../../scripts/README.md)

This workflow was prepared at the user's request to test the adapter saved
**during the original grid screening**. It performs no retraining and changes
neither the selected NorBERTo model nor the primary soft-voting decision.

The exact source is:

```text
../EP1/artifacts/norberto-lora-grid-20260929-170333/screening/candidate-08/best_adapter/
```

It used 12,982 training rows and 3,111 internal validation rows, selecting
epoch 2 with validation accuracy 44.2944%. Its LoRA rank is 16, alpha 32 and
learning rate 0.0003. The adapter weights SHA-256 is
`8e9c8332866fe3c2649382bb6fc4f9970b76471364516e58b7f690bef0e066ca`.
It is distinct from the three later CV-fold adapters and the development-wide
final refit (`efdb8ede…`), whose holdout accuracy was 35.98%.

Run from EP2, using the already available offline ROCm image:

```bash
bash scripts/experiments/norberto/candidate08-test1.sh \
  --output-dir results/norberto-candidate08-screening-test1
```

Use a fresh output directory. `NORBERTO_IMAGE` can select a compatible image.
The launcher mounts EP1 read-only, disables networking and uses the same
FP32/SDPA, batch-8, cased 512-token head/tail inference contract. The public
base is verified through the existing frozen NorBERTo bundle; the experimental
adapter has its own strict screening/configuration/weights provenance checks.
No general-purpose bypass of the selected model's immutable checks is added.

All 900 text-only predictions, row IDs, text hashes and probabilities are
persisted before the labeled Test1 workbook is read. The result directory
contains `predictions.jsonl`, `metrics.json` and `run.json`; metrics include
accuracy, correct count, macro-F1, per-class scores and a confusion matrix.
The comparison is retrospective and exploratory; it does not trigger model
promotion, weight adjustment or a new Test2 delivery.

To verify the adapter and mounts without loading the model or reading Test1:

```bash
bash scripts/experiments/norberto/candidate08-test1.sh --preflight
```

Preparation verified the original checkpoint and its source metadata. No
900-row prediction or accuracy measurement was executed during preparation.
