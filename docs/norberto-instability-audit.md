# NorBERTo instability: optimizer and backend audit

The corrected development-only adapter is unstable, but the new checks do
**not identify an incorrect sample-normalized loss or an extra accumulation
division**. The complete tiny-model Trainer loop agrees with an independent
manual CPU implementation, including GPU execution. The saved adapter's large
derivatives also occur on CPU and agree locally with finite differences.
The measured sensitivity is concentrated in the **first attention LoRA**.

The exact cause of the training trajectory remains unresolved. Passing these
checks does not certify the entire training pipeline or recover model quality.

## Complete optimizer-loop verification

[`audit-norberto-optimizer.py`](../scripts/audit-norberto-optimizer.py) uses a
random two-layer ModernBERT with PEFT, 29 synthetic examples, and zero dropout.
It executes four updates over two epochs, alternately accumulating batches of
`4+4+4+4` and `4+4+4+1`. The production loss-injection context is used unchanged.
The full `Trainer.train()` loop is compared with a separate model and manual
cross entropy, clipping, and AdamW updates on CPU.

| Check | CPU Trainer | GPU Trainer versus CPU reference |
|---|---:|---:|
| Updates checked | 4 | 4 |
| Maximum relative gradient error | 9.48e-8 | 2.57e-7 |
| Maximum absolute parameter difference | 1.50e-8 | 1.50e-8 |
| Maximum post-clipping gradient norm | < 1.00000 | < 1.00000 |
| Frozen parameters unchanged | Yes | Yes |

The actual batch counts are 16 and 13, respectively; every microbatch receives
the correct accumulated count. Accumulated loss agrees with manual mean cross
entropy. The test independently checks the linear scheduler's four learning
rates, gradient clearing between updates, and frozen parameter preservation.
The accelerator's accumulation divisor is 1 in this Trainer integration;
the custom loss already handles the four microbatches. There is no additional
division by four in the checked execution.

The existing loss and partition suites also passed: **18 tests in 5.19 seconds**.
These tests provide specific integration evidence, not proof of every update
in a full-size training run.

## Real saved-adapter gradients on CPU and GPU

[`audit-norberto-backends.py`](../scripts/audit-norberto-backends.py) selected
the first four responses per class with at most 64 tokens from the shared
development training rows. This fixed batch has 12 examples, padded to 64
tokens. Dropout was disabled to isolate numerical differences. No optimizer
was constructed for any production model.

| Adapter | CPU SDPA norm | CPU eager norm | GPU SDPA norm |
|---|---:|---:|---:|
| Original EP1 | 1.84432 | 1.84432 | 1.84432 |
| Corrected full-corpus refit | 1.71013 | 1.71012 | 1.71011 |
| Corrected development-only refit | 158.82942 | 158.83560 | 158.98747 |

For the development-only adapter, CPU eager differs from CPU SDPA by a
relative gradient vector difference of **4.30e-5**. GPU SDPA differs from CPU
SDPA by **0.001058**, with maximum logit difference **0.0003555**. The large
gradient is present across all three implementations. This makes a failure
exclusive to the GPU override or to SDPA insufficient to explain the measured
checkpoint sensitivity. It does not exclude backend effects during training.

All loaded adapter tensors match their saved tensors exactly after the probes;
all three saved weight hashes are unchanged. Two fresh constructions from the
pinned public base and seed 42 reproduce the same trainable-parameter hash in
the current environment. This does not prove historical bitwise identity.

## Local derivative verification and first-layer localization

[`audit-norberto-sensitivity.py`](../scripts/audit-norberto-sensitivity.py)
reloaded the development-only adapter on CPU and used the same 12 training
responses. The measured norm was approximately **158.83**, with **99.911% of
the squared gradient norm** in the first attention layer's LoRA parameters.

One coordinate with the largest absolute derivative was checked by central
finite differences in FP32, accounting for the actual representable parameter
displacement:

| Perturbation | Finite-difference derivative | Autograd derivative | Relative error |
|---|---:|---:|---:|
| 0.001 | 17.44067 | 18.15042 | 3.9103% |
| 0.0001 | 18.18004 | 18.15042 | 0.1632% |
| 0.00001 | 18.24995 | 18.15042 | 0.5484% |

This verifies a genuinely sensitive direction in the saved model; it does not
verify every coordinate of the backward pass. Temporarily setting only the
first attention LoRA's multiplier to zero in memory reduced the total norm
to **2.42088**. The multiplier and parameter values were then restored, and
all adapter tensors were checked against the saved checkpoint.

This ablation localizes the observed sensitivity. It is not an evaluated
replacement adapter, an accuracy improvement claim, or evidence that the
first layer was the initial trigger of the training regression.

## Loss normalization changes the clipping regime

Keeping the same numerical configuration after correcting the loss does not
preserve the entire legacy optimization trajectory. For equal microbatches,
the legacy gradient was approximately `4g`, where `g` is the correct mean-loss
gradient. Both recipes use `max_grad_norm=1`.

For example, consider `g=(0.3, 0.4)`:

| Recipe | Raw gradient norm | Gradient after clipping, approximately |
|---|---:|---|
| Legacy accumulation | 2.0 | `(0.6, 0.8)` |
| Correct mean loss | 0.5 | `(0.3, 0.4)` |

If both raw gradients exceed the threshold, their clipped vectors are nearly
identical. If they straddle the threshold, they differ. As this condition
changes across updates, AdamW's moment estimates can evolve differently.
Ignoring numerical stabilizers, `clip(4g, 1) = 4 * clip(g, 0.25)`; keeping
the corrected threshold at 1 therefore changes its effective clipping regime.
Adam epsilon, unequal final microbatches, and floating-point differences also
prevent a general claim of identical training. No clipping parameter or
learning rate was changed as part of this audit.

The historical development logs initially agree after accounting for the
factor of four: at update 20, the original norm divided by four is 3.8596323,
versus 3.8596320 for the corrected run. The loss values agree to logging
precision. At update 80 the corresponding norms are 4.6441612 and 4.6441574.
Differences subsequently grow; the corrected run's first logged norm above 25
is **253.149 at update 540, epoch 0.5368**. Logs are sampled every 20 updates,
so they cannot identify the exact first unstable update or its cause.

The clipping interaction is an established mathematical difference between
recipes, and a candidate mechanism for changed trajectories. It is **not a
demonstrated explanation of the 386 lost holdout predictions**.

The implementation details are supported by the pinned
[Transformers 4.57.1 Trainer source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/trainer.py),
[ModernBERT classification source](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/modernbert/modeling_modernbert.py),
and PyTorch 2.8 documentation for
[gradient clipping](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html)
and [AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html).

## Scope and remaining causal question

All production models and inputs were mounted read-only, with network access
disabled. No full-model optimizer updates, new holdout inference, test1/test2
access, model replacement, or ensemble changes occurred. Only the random tiny
models were updated. The confirmed accumulation fix remains unchanged.

A bounded training-only reproduction that traces every update around the
first divergence is still required to separate clipping effects, numerical
sensitivity, data order, and other implementation issues. The current evidence
does not justify claiming either a new confirmed loss bug or a fully explained
regression. Another full training run is not a substitute for that isolation.

The subsequent [optimizer-unit correction](norberto-optimizer-units-fix.md)
and its [completed holdout evaluation](avaliacao-norberto-recipe3-holdout.md)
document a controlled prefix and full-run follow-up. Recipe 3 reproduced the
original logged gradient norms through update 1,000, avoided the preceding
escalation, and recovered 44.81% holdout accuracy. The audit above describes
the evidence available before that correction.

Inside the validated image with the project, EP1, and cache mounts available,
the diagnostics can be reproduced with new output paths:

```bash
PYTHONPATH=src python3 scripts/audit-norberto-optimizer.py \
  --device cpu --output results/new-audit/optimizer-cpu.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/audit-norberto-optimizer.py \
  --device gpu --output results/new-audit/optimizer-gpu.json
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=src \
python3 scripts/audit-norberto-backends.py --ep1-root /ep1 \
  --output results/new-audit/backends.json
# Run this final command in a CPU-only container without GPU device exposure.
PYTHONPATH=src python3 scripts/audit-norberto-sensitivity.py --ep1-root /ep1 \
  --backend-report results/new-audit/backends.json \
  --output results/new-audit/sensitivity.json
```

The backend and sensitivity commands require an existing parent output directory.
Raw outputs are in `results/norberto-deep-audit-20261105/`.
The [public aggregate](norberto-deep-audit-results.json) records measured values,
hashes, test scope, and limitations. The preceding
[holdout validation](avaliacao-norberto-development-holdout.md) remains unchanged.
