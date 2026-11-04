"""Measure single-GPU throughput at a fixed effective batch and context length."""
from __future__ import annotations

import gc
import time
from dataclasses import replace
from pathlib import Path

from gpt_classifier.data import write_json
from .config import Config
from .hardware import environment, memory, runtime_identity
from .model import build_model, tokenizer_for


def benchmark(config: Config, output: Path, cache_dir: Path, *, batches: list[int],
              checkpointing: bool, measured_steps: int = 3) -> dict:
    import torch
    from transformers import set_seed

    if config.mode == "auto":
        raise ValueError("Benchmark one explicit precision mode per process.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if measured_steps < 2 or any(batch < 1 or 16 % batch for batch in batches):
        raise ValueError("Use at least two timed updates and batches dividing effective batch 16.")
    output.mkdir(parents=True)
    set_seed(config.seed)
    tokenizer = tokenizer_for(config, cache_dir)
    initial = replace(config, gradient_checkpointing=checkpointing)
    model = build_model(initial, tokenizer, cache_dir)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    originals = [p.detach().cpu().clone() for p in trainable]
    inputs = torch.randint(100, 100000, (16, config.max_length), device="cuda")
    inputs[:, -1] = tokenizer.eos_token_id
    attention = torch.ones_like(inputs)
    labels = torch.arange(16, device="cuda") % 3
    report = {"config": config.to_dict(), "environment": environment(), "device": runtime_identity(),
              "effective_batch_size": 16, "sequence_length": config.max_length,
              "warmup_updates": 1, "measured_updates": measured_steps, "results": [],
              "scope": "Synthetic full-length throughput and memory, not model quality"}
    for batch_size in batches:
        candidate = replace(config, batch_size=batch_size, accumulation_steps=16 // batch_size,
                            gradient_checkpointing=checkpointing)
        optimizer = None
        result = {"config": candidate.to_dict()}
        try:
            model.zero_grad(set_to_none=True)
            for parameter, original in zip(trainable, originals):
                parameter.data.copy_(original)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, foreach=False)
            scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
            timings, skipped = [], 0
            for step in range(measured_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for offset in range(0, 16, batch_size):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(input_ids=inputs[offset:offset + batch_size],
                                     attention_mask=attention[offset:offset + batch_size],
                                     labels=labels[offset:offset + batch_size]).loss
                        scaled_loss = loss / candidate.accumulation_steps
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite benchmark loss.")
                    scaler.scale(scaled_loss).backward()
                    del loss, scaled_loss
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if step:
                    timings.append(elapsed)
            changed = any(not torch.equal(p.detach().cpu(), original) for p, original in zip(trainable, originals))
            if skipped or not changed:
                raise FloatingPointError("Benchmark updates were skipped or did not change adapters.")
            result.update(status="passed", seconds_per_update=sum(timings) / len(timings),
                          examples_per_second=16 * len(timings) / sum(timings),
                          update_seconds=timings, skipped_overflow_steps=skipped,
                          verified_adapter_update=changed, **memory())
        except torch.OutOfMemoryError as error:
            result.update(status="out_of_memory", error=str(error)[:1000], **memory())
        except (RuntimeError, FloatingPointError) as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        report["results"].append(result)
        write_json(output / "benchmark.json", report)
        print(f"BENCHMARK {config.mode} batch={batch_size} checkpointing={checkpointing}: {result}", flush=True)
        if result["status"] != "passed":
            break
    return report
