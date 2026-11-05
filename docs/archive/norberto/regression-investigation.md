# Investigation of the full-corpus NorBERTo regression

[Documentation index](../../README.md) · Historical experiment record; use the guides for current commands.

Follow-up: the [completed normalized refit](recipe2-full-test1.md)
achieved 408/900 (45.33%), compared with 341/900 for the legacy full refit and
432/900 for the original EP1 submission. The initial investigation below
remains a training-only diagnostic; the follow-up is a separate test1 comparison.

Evidence points to **unstable optimization during the full-corpus run**. A
separate loss-normalization bug was reproduced, but it affects both historical
runs and is not a proven sole cause of the regression. No full model was
retrained, no optimizer update was performed, and diagnostics did not access test1.

## Verified inputs and parameters

The two parsers returned identical texts and labels on all 20,092 rows. Token
IDs, masks, and token type IDs matched on all 16,093 shared development rows.
Saved numerical training arguments matched; only output/logging paths differed.
Class counts in `c1/c234/c5` order were 5,090/5,436/5,567 in development and
6,347/6,853/6,892 in the full corpus. Base and adapter weight hashes matched.

The original trainer source was recovered from EP1 commit
`fcae20328b49ef312b39babbe41a67acfd6acb1c`, matching its recorded SHA-256.
Subsequent changes concerned device selection, logging, organization, and
inference. Initialization, supervised tokenization, and numerical training
arguments remained equivalent on the selected GPU path. This does not establish
bitwise identical historical execution or exclude all backend differences.

## Instability already present in training

Mean logged gradient norms, before clipping:

| Epoch interval | Original training | Full-corpus training |
|---|---:|---:|
| 0–0.5 | 11.51 | 15.59 |
| 0.5–1 | 8.09 | 106.16 |
| 1–1.5 | 7.58 | 368.82 |
| 1.5–2 | 8.14 | 319.61 |

Maximum norms were 22.58 and 1,221.64. The first full-run record above 100
occurred at step 1,100, epoch 0.876: norm 1,090.99, learning rate 0.00062522.
Clipping was enabled at 1.0; these are not the norms passed directly to AdamW.
Late logged loss averaged 4.1214 versus 4.4352. Those historical losses are
scaled by the defect below and are not ordinary per-response cross entropy.

## Training-only probes

A fixed sample used 300 shared training responses, 100 per class, seed 42:

| Metric | Original adapter | Full refit |
|---|---:|---:|
| Correct | 151 | 111 |
| Training-sample accuracy | 50.33% | 37.00% |
| Mean cross entropy | 0.989255 | 1.103177 |

These are fit diagnostics, not generalization estimates. On another fixed batch
of 12 short training responses, with dropout disabled and no optimizer:

| Measurement | Original adapter | Full refit |
|---|---:|---:|
| CPU gradient norm | 1.8443 | 189.7469 |
| GPU gradient norm | 1.8443 | 190.1226 |
| Maximum CPU/GPU logit difference | 0.00000501 | 0.00034493 |
| Relative CPU/GPU gradient difference | 0.00119% | 0.40856% |

Large gradients also occur on CPU, weakening an explanation based solely on
incorrect GPU inference. A small probe cannot certify all training kernels.
All adapter tensors matched their saved values after these calculations.
Mean LoRA update norms were similar: 6.8376 and 6.8396; a uniform explosion
in parameter magnitude was not demonstrated.

## Confirmed loss-normalization bug

The Trainer interprets `**kwargs` as support for accumulated example counts.
ModernBERT's native classification loss instead computes a microbatch mean and
ignores those counts. See the primary [Trainer implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py)
and [ModernBERT implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/modernbert/modeling_modernbert.py).

Synthetic CPU ModernBERT/PEFT forward/backward calculations reproduced:

| Microbatch sizes | Legacy / correct loss | Legacy / correct gradient norm |
|---|---:|---:|
| 4 + 4 + 4 + 4 | 4.0000 | 4.0000 |
| 4 + 4 + 4 + 1 | 4.1404 | 5.3711 |

The unequal tail also distorted gradient direction, with normalized-direction
error 0.4244. Summed cross entropy divided by total examples matched a complete
batch, with relative gradient error below 0.0000001.

**The bug is not a proven sole cause of the accuracy loss.** It existed in both
runs. A constant scaling on equal microbatches is largely canceled when both
gradients are clipped to the same norm. It must not be described as simply
multiplying AdamW's learning rate by four. Unequal tails and finite precision remain relevant.

## Causal limits and correction

Two epochs on more data increased updates from **2,012 to 2,512 (+24.85%)**,
warmup from 202 to 252, and changed sampling order, while retaining learning
rate 0.001. Equal hyperparameters did not preserve the optimization trajectory.
The full refit reached a poorly conditioned, poorly fitting state. The unique
trigger among data composition, ordering, update budget, backend sensitivity,
and their interactions was not isolated by controlled retraining.

The training wrapper now records recipe version 2 and uses explicit
sample-normalized loss from [training_loss.py](../../../src/norberto_classifier/training_loss.py).
Numerical hyperparameters and EP1 sources remain unchanged; the legacy Trainer
constructor is restored after the call. Historical artifacts were preserved.
Nine targeted CPU tests passed, including actual ModernBERT/PEFT/Trainer
gradient equivalence on an unequal tail. A full-corpus dry-run passed without
model imports. At that stage, no full training or test1 reevaluation had used
the correction; the subsequent follow-up is linked above.

Further causal experiments should isolate normalization, data inclusion, and
update budget using development-only controls and internal validation. Test1
must not choose hyperparameters or ensemble rules while being called untouched.

Scripts: `CPU audit` and
`training-only stability probe`.
The [public aggregate](../../results/data/norberto-training-investigation-results.json) contains
measurements; raw reports are in `results/norberto-full-investigation-20261105/`.
