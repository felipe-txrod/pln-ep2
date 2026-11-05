# Historical training-scale audit: recipe 3

Retired on 2026-11-05. The selected model is the unchanged original EP1 delivery
adapter. Measurements below are historical; their training and diagnostic
implementations were removed from the active source tree and preserved locally
in `results/norberto-rollback-20261105/sources-before-rollback.tar.gz`.

A new training-only audit compared both saved recipe-3 adapters on the same
seed-42 stratified sample of 300 development training rows (100 per class).
These rows were used to train both adapters. No optimizer was constructed,
and test1/test2 were not accessed during this audit.

| Measurement | Development adapter | Full-corpus adapter |
|---|---:|---:|
| Correct on the shared training sample | 156/300 (52.00%) | 113/300 (37.67%) |
| Shared-sample cross entropy | 0.98624 | 1.08960 |
| Shared-sample predictions `c1 / c234 / c5` | 96 / 111 / 93 | 62 / 15 / 223 |
| Peak logged pre-clipping optimizer gradient norm | 22.58 | 1,196.86 |
| Final-adapter CPU gradient norm on 12 short training rows | 1.60418 | 1.63878 |

The full adapter fits this shared training sample worse and already exhibits
the `c5` bias there. This supports an optimization/fitting failure. These
training-sample scores do not estimate held-out generalization.

All 16,093 shared cached feature sets match exactly, and fresh tokenization
matches all 20,092 full training rows. Training manifests, source hashes,
partition coverage, exported tensors, and unchanged weight hashes passed
verification. The training proportion of `c5` decreased from 34.59% to 34.30%;
an increased class prior does not explain the prediction bias.

The full run's first logged sample-unit norm above 25 occurs at update 1,100,
before the first epoch's partial accumulation window at update 1,256. That
tail therefore cannot be the initial trigger of this observed escalation.
The final-adapter derivatives are normal-sized on the checked short batch:
CPU SDPA/eager and GPU SDPA agree, with GPU relative gradient differences
below 0.000011 for both adapters. First-attention gradient energy is only
5.29% in the full adapter (12.74% in development). The earlier recipe-2
checkpoint's first-attention concentration is not reproduced here.

Recipe 3 preserves the intended loss and optimizer units, but does not ensure
stable training as the dataset grows. Two epochs changed updates from 2,012
to 2,512, warmup from 202 to 252 updates, and sampling order. The unique cause
among row inclusion, ordering, schedule, and their interaction remains
unisolated. Logs every 20 updates cannot verify actual clipping or locate the
first unstable update exactly; normal endpoint derivatives do not certify the
trajectory. A controlled follow-up needs fresh training with internal grouped
validation, one changed factor at a time, and per-update tracing around the
spike. Test1/test2 gold must not select the correction.

The [audit aggregate](../../results/data/norberto-training-scale-audit.json) records all
measurements, hashes, and limitations. The audit took 74.40 seconds on the
recorded offline ROCm environment, and 34 targeted synthetic CPU tests passed.
Production recipes, adapters, and ensemble selection were preserved.
Inside the validated image with standard EP1/cache mounts, reproduce with:

```bash
PYTHONPATH=src HSA_OVERRIDE_GFX_VERSION=10.3.0 \
python3 scripts/diagnostics/norberto/audit-training-scale.py \
  --ep1-root /ep1 --device gpu \
  --output-dir results/norberto-scale-audit-another-new-run
```

The follow-up forensic comparison identifies why the previous correction did
not resolve full-corpus training: commit `f947d88` replaced the normalized
Trainer with the legacy-compatible Trainer for both partitions. All **62/62
sampled full-run gradients and learning rates before update 1,256** match the
original unstable full run exactly. The matching spikes include updates
1,100 and 1,120; the first logged divergence is only at 1,260, after tail
weighting changes. On complete windows, multiplying normalized loss by four
restores the legacy sum of microbatch means. This confirms recurrence of the
known sampled trajectory, not bitwise equality of every hidden tensor.

The earlier 600-update real-model probe hardcoded the development partition.
Its agreement with EP1 and the tiny tests' deliberately scaled references
validated compatibility, without covering that full-run regression. The
probe now accepts `--partition full`. The production wrapper also requires
the original frozen configuration, including learning rate 0.001. Neither
the normalization fix nor the compatibility change validated a new optimizer
configuration for both dataset sizes.

The log-only comparison performs no model loading or optimization:

```bash
PYTHONPATH=src python3 scripts/diagnostics/norberto/audit-training-scale.py \
  --ep1-root /ep1 --history-only \
  --output-dir results/norberto-history-regression-another-new-run
```

The follow-up also performed **bounded real training**, separately from the
initial saved-adapter audit. A per-update trace reached update 376 before
being stopped to narrow the experiment. A fresh 320-update prefix reproduced
all its first 320 batch identities and gradient norms and all 16 available
historical log samples. It saved weights, AdamW, scheduler, and RNG at update
200. Resuming without an intervention reproduced all 120 following batch
identities, gradient norms, and update norms exactly; the final adapter binary
was identical to the uninterrupted prefix.

The controlled treatment resumed that same checkpoint and multiplied only
the scheduled optimizer learning rate by 0.1 for updates 201–320. All batch
identities matched, the first gradient was identical, and the first parameter
update norm was 0.10000013 times the baseline. Loss normalization, compatibility
scale, dropout RNG, optimizer moments, clipping, and scheduler timing were held
fixed at the intervention boundary.

| Window 201–320 | Original optimizer rates | Rates multiplied by 0.1 |
|---|---:|---:|
| Peak pre-clipping optimizer gradient norm | 576.49658 | 29.24076 |
| Updates with norm above 100 | 10 | 0 |
| Mean per-update example loss | 1.11307 | 1.08339 |
| Mean parameter update norm | 0.12863 | 0.01396 |

All post-clipping norms remained at most 1.000001. At the original peak,
update 276, the first attention LoRA contained 99.76% of gradient energy;
at update 257 it contained 99.73%. These transient peaks were missed by the
old every-20-update logs and are distinct from normal endpoint derivatives.
This provides local causal evidence that smaller optimizer updates avoid
those peaks in the reproduced state. It does not isolate gradient scaling
independently or establish an end-to-end stable recipe or accuracy gain.
The new prefix exports are diagnostic, with no production replacement or
test1/test2 access. The [aggregate](../../results/data/norberto-training-scale-audit.json)
records the trace/checkpoint hashes, actual optimizer learning rates, negative
control, 936 total diagnostic optimizer updates, and validation limits.
