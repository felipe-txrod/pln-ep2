# NorBERTo optimizer-unit compatibility correction

Historical workflow retired on 2026-11-05. The unchanged original EP1 delivery
adapter is the selected model. Commands below describe the former experiment;
its active training/evaluation implementation has been removed. Local source
snapshots are preserved in `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

[Documentation index](../../README.md)

The initial normalization fix corrected example weighting but changed the units
of gradients consumed by the inherited clipping threshold and AdamW optimizer.
Recipe version 3 makes the unit conversion explicit. This addresses a recipe
compatibility defect; model accuracy still requires a completed independent
holdout evaluation.

## Confirmed limitation of the correction

The subsequent full-corpus regression audit found that recipe 3 reproduces
**all 62 sampled pre-tail gradient norms and learning rates of the known
unstable legacy full run**, including the spikes at updates 1,100 and 1,120.
The first logged difference is at update 1,260, after the tail at 1,256.
This follows from the intended compatibility conversion: on complete
4-by-4 windows, `4 * sum(CE) / 16` is the original sum of microbatch means.
The conversion was deployed in commit `f947d88` for both partitions.

The previous real-model prefix probe hardcoded `development` and ran only
600 updates; its successful comparison to EP1 did not test the failing full
trajectory. The tiny-model tests intentionally compare against gradients
multiplied by four. They verify that conversion, not full-corpus stability.
The prefix probe now accepts `--partition full` and derives its stop limit
from that partition. A separate per-update tracer records actual clipping,
batch hashes, and parameter updates and saves optimizer/scheduler/RNG state
for controlled interventions.

Recipe 3 is an EP1 compatibility implementation, not a validated correction
for the full-corpus instability. The production wrapper also pins the original
learning rate of 0.001 and does not offer a separate training configuration.
Removing the conversion alone is not an established general remedy: recipe 2
also failed on development data. See the [current audit](../../results/README.md#training-scale-audit-recipe-3)
for evidence and the limits of the attribution.

## Gradient units

For microbatch j with n_j labeled examples and N examples in the accumulated
update, the reported contribution is `sum(CE_j) / N`. Backpropagation uses
`K * sum(CE_j) / N`, where K is configured gradient accumulation (4). The
Trainer's returned training loss is divided by K, and evaluation is unscaled.
The original learning rate, clipping threshold, Adam epsilon, weight decay,
seed, target layers, dataset partition, and epoch budget are preserved.
Logged `grad_norm` is in optimizer units, as in EP1; divide it by K when
comparing pre-clipping norms with version 2. The logged loss is already the
example mean and must not be divided again.

For four equal microbatches, this reproduces the original EP1 gradient units.
For unequal or incomplete groups, every example has the same weight. It does
not reproduce the original excessive weighting of the one-example tail.
The compatibility scaling is deliberately distinct from objective averaging;
it is recorded explicitly rather than relying on Transformers' kwargs heuristic.

The implementation is isolated in
``training_recipe.py``.
The existing sample-loss helper remains unchanged so historical manifests and
validation remain usable. The production wrapper selects recipe version 3;
the holdout evaluator accepts versions 2 and 3 and verifies the new recipe hash
and optimizer-scale contract when evaluating version 3.

## Verification

The 24 CPU tests cover sample loss, partitions, complete and partial accumulation
groups, gradients below and above the clipping threshold, three sequential
AdamW updates, and unscaled evaluation loss. All passed in 8.01 seconds.

A complete four-update tiny ModernBERT/PEFT Trainer run on GPU was compared
with manual sample-weighted/scaled gradient accumulation, clipping and AdamW
on CPU. Maximum parameter error was 1.49e-8; maximum gradient relative error
was 2.68e-7. Frozen parameters remained identical. Reported loss matched the
example mean. This verifies the integration, including the unequal 4+4+4+1 tail.

The bounded real-model diagnostic completed all 600 requested updates in
559.68 seconds. It used all 16,093 development rows as the training dataset.
All 30 logged gradient norms and learning rates matched the original EP1 run
exactly. At update 540 the norm was 7.78463, compared with 253.149 in version 2
(or 1.94616 versus 253.149 in comparable mean-gradient units).
The original norm at that update was also 7.78463. It retains the original
2,012-update learning-rate schedule and warmup; reducing `max_steps` would
change those schedules and invalidate the comparison. The 3,999 holdout rows
are excluded. The diagnostic exports to a new directory and its training status
is intentionally `incomplete`; it must not be presented as a final classifier.

Reproduce the bounded diagnostic inside the validated ROCm image:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/diagnostics/norberto/probe-training-prefix.py \
  --ep1-root /ep1 --updates 600 \
  --output-dir /workspace/results/norberto-v3-prefix-new
```

A full development-only run can be started separately:

```bash
NORBERTO_IMAGE=pln-rocm:6.4.1 bash scripts/norberto/train.sh \
  --partition development \
  --output-dir /workspace/results/norberto-development-compatible-v3
```

The subsequent full run completed two epochs and 2,012 updates. Its
[holdout evaluation](../../results/norberto-recipe3-holdout.md) records final quality
and full-run gradient summaries. A short prefix alone does not establish
long-run stability or quality. No test1/test2 labels were used for this
correction, and no saved original adapter was replaced.

For separate prediction and scoring on test1, see the
[two-stage test1 commands](../../guides/norberto-test1.md). The prediction stage accepts
this development-only adapter and records its identity; scoring uses only
the persisted predictions on CPU.
