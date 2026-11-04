"""Training, real-model smoke tests, checkpoint validation, and mode selection."""
from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from gpt_classifier.data import LABELS, Example, canonical, digest, read_jsonl, write_json
from .config import Config
from .dataset import TokenizedTexts, prepare
from .hardware import environment, memory, probe, runtime_identity
from .model import build_model, tokenizer_for


def classification_metrics(labels, predicted) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predicted)), "f1_macro": float(f1),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "label_order": list(LABELS),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(len(LABELS)))).tolist(),
            "classification_report": classification_report(
                labels, predicted, labels=list(range(len(LABELS))), target_names=list(LABELS),
                zero_division=0, output_dict=True)}


def trainer_metrics(prediction) -> dict:
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    scores = classification_metrics(prediction.label_ids, np.argmax(logits, axis=-1))
    return {name: scores[name] for name in ("accuracy", "f1_macro", "precision_macro", "recall_macro")}


def classification_loss(outputs, labels, num_items_in_batch=None):
    """Normalize by examples across the entire accumulated batch, including its tail."""
    import torch
    import torch.nn.functional as functional
    loss = functional.cross_entropy(outputs.logits.float(), labels, reduction="sum")
    count = labels.numel() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(count):
        count = count.to(loss.device)
    return loss / count


def _guard_callback():
    import torch
    from transformers import TrainerCallback

    class FiniteGradientGuard(TrainerCallback):
        """Reject nonfinite gradients and verify actual LoRA updates during smoke tests."""

        def __init__(self):
            self.steps = 0
            self.previous = None
            self.parameter = None
            self.changed = False
            self.finite = False
            self.skipped_steps = 0
            self.update_seconds = []
            self.step_started = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.parameter = next(p for name, p in model.named_parameters() if "lora_B" in name and p.requires_grad)
            self.previous = self.parameter.detach().clone()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients:
                raise FloatingPointError("Training did not produce gradients.")
            self.finite = all(torch.isfinite(g).all().item() for g in gradients)
            if self.finite and not any(torch.count_nonzero(g).item() for g in gradients):
                raise FloatingPointError("Every trainable gradient is zero.")

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = time.perf_counter()

        def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
            if not self.finite:
                if not getattr(optimizer, "step_was_skipped", False):
                    raise FloatingPointError("The optimizer did not skip nonfinite FP16 gradients.")
                self.skipped_steps += 1
                if self.skipped_steps > 8:
                    raise FloatingPointError("Repeated FP16 overflows; reduce sequence length or inspect kernels.")

        def on_step_end(self, args, state, control, **kwargs):
            self.steps += int(self.finite)
            self.changed |= not torch.equal(self.previous, self.parameter)
            if self.step_started is not None:
                self.update_seconds.append(time.perf_counter() - self.step_started)

    return FiniteGradientGuard()


def fit(config: Config, output: Path, cache_dir: Path, *, source: Path | None = None,
        smoke: bool = False, pilot: bool = False) -> dict:
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    set_seed(config.seed)
    device = runtime_identity()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = tokenizer_for(config, cache_dir)
    if smoke:
        # Stress the configured token length without inspecting any test workbook.
        text = "This public response supplies the requested information. " * config.max_length
        train_records = [Example(i + 2, text + str(i), LABELS[i % 3])
                         for i in range(max(6, config.accumulation_steps * config.batch_size))]
        validation_records = train_records[:3]
    else:
        if source is None:
            raise ValueError("A labeled training workbook is required.")
        prepare(source, output / "data", config)
        train_records = [Example(**row) for row in read_jsonl(output / "data" / "train.jsonl")]
        validation_records = [Example(**row) for row in read_jsonl(output / "data" / "validation.jsonl")]
        if pilot:
            from sklearn.model_selection import train_test_split
            partitions = []
            for records, limit in ((train_records, 96), (validation_records, 48)):
                if len(records) > limit:
                    records, _ = train_test_split(records, train_size=limit, random_state=config.seed,
                                                   stratify=[r.label for r in records])
                partitions.append(sorted(records, key=lambda r: r.excel_row))
            train_records, validation_records = partitions
            write_json(output / "pilot-selection.json", {
                "train_rows": [r.excel_row for r in train_records],
                "validation_rows": [r.excel_row for r in validation_records],
                "method": "Stratified samples within the existing disjoint grouped partitions; seed fixed."})
    train_data = TokenizedTexts(train_records, tokenizer, config.max_length)
    validation_data = TokenizedTexts(validation_records, tokenizer, config.max_length)
    model = build_model(config, tokenizer, cache_dir)
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    guard = _guard_callback()
    arguments = TrainingArguments(
        output_dir=str(output / "checkpoints"), seed=config.seed, data_seed=config.seed,
        num_train_epochs=config.epochs, max_steps=2 if smoke or pilot else -1,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.accumulation_steps,
        learning_rate=config.learning_rate, warmup_ratio=0 if smoke else config.warmup_ratio,
        weight_decay=config.weight_decay, lr_scheduler_type="cosine", max_grad_norm=1.0,
        fp16=True, bf16=False, gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", optim_args="foreach=False", eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch", save_total_limit=2,
        load_best_model_at_end=not smoke, metric_for_best_model="accuracy", greater_is_better=True,
        logging_strategy="steps", logging_steps=1 if smoke else 10,
        report_to="none", dataloader_num_workers=0, dataloader_pin_memory=False,
        eval_accumulation_steps=1, save_safetensors=True, save_only_model=False,
        remove_unused_columns=True, logging_nan_inf_filter=False)
    arguments.group_by_length = config.group_by_length
    trainer = Trainer(
        model=model, args=arguments, train_dataset=train_data, eval_dataset=validation_data,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=trainer_metrics, compute_loss_func=classification_loss, callbacks=[guard])
    # Qwen3 sequence classification does not forward num_items_in_batch to its loss.
    # Compute sample-normalized cross entropy explicitly instead of trusting **kwargs detection.
    trainer.model_accepts_loss_kwargs = False
    # Qwen's BF16-pretrained activations need a conservative FP16 starting scale.
    # Dynamic scaling can still skip an overflow safely and decrease the scale.
    trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    trained = trainer.train()
    if not math.isfinite(trained.training_loss) or not guard.changed or guard.steps < 1:
        raise RuntimeError("Training did not produce a finite loss and a verified LoRA update.")
    evaluation = trainer.predict(validation_data)
    logits = np.asarray(evaluation.predictions)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Validation produced nonfinite logits.")
    scores = classification_metrics(evaluation.label_ids, logits.argmax(axis=-1))
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    write_json(adapter / "experiment.json", {"config": config.to_dict(), "labels": list(LABELS)})
    write_json(output / "validation-metrics.json", scores)
    write_json(output / "validation-predictions.json", [
        {"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": LABELS[int(index)],
         "gold": r.label} for r, index in zip(validation_records, logits.argmax(axis=-1))])
    peak = memory()
    if smoke:
        # Reload in the same process only after releasing optimizer, model, and callback references.
        guard.parameter = None
        guard.previous = None
        del trainer, model, evaluation
        gc.collect()
        torch.cuda.empty_cache()
        from .inference import predict_records
        restored = predict_records(adapter, cache_dir, validation_records)
        restored_logits = np.asarray([row["logits"] for row in restored])
        np.testing.assert_allclose(restored_logits, logits, atol=0.03, rtol=0.03)
        if [row["label"] for row in restored] != [LABELS[int(i)] for i in logits.argmax(axis=-1)]:
            raise RuntimeError("Adapter reloading changed smoke-test predictions.")
    report = {"status": "passed" if smoke else "completed", "smoke": smoke, "pilot": pilot,
              "config": config.to_dict(), "config_sha256": config.fingerprint,
              "environment": environment(), "device": device, "memory": peak,
              "optimizer_steps": guard.steps, "verified_lora_update": guard.changed,
              "skipped_overflow_steps": guard.skipped_steps,
              "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
              "effective_batch_size": config.batch_size * config.accumulation_steps,
              "update_seconds": guard.update_seconds,
              "mean_update_seconds": sum(guard.update_seconds) / len(guard.update_seconds),
              "validation_runtime_seconds": float(evaluation.metrics["test_runtime"]) if not smoke else None,
              "adapter_reload_verified": smoke, "training_loss": trained.training_loss,
              "duration_seconds": time.monotonic() - started,
              "train_rows": len(train_data), "validation_rows": len(validation_data),
              "truncated_train_rows": train_data.truncated, "truncated_validation_rows": validation_data.truncated,
              "validation_metrics": scores,
              "metric_scope": "Synthetic compatibility data, not classifier quality" if smoke else
                              "Two-step pilot on training-corpus partitions; not a final quality estimate" if pilot else
                              "Grouped internal validation; not official test1 or test2",
              "best_checkpoint": None if smoke else trainer.state.best_model_checkpoint}
    write_json(output / "run.json", report)
    return report


def smoke_test(config: Config, output: Path, cache_dir: Path) -> dict:
    """Try NF4 first and isolate each real-model attempt before selecting a fallback."""
    output.mkdir(parents=True, exist_ok=False)
    compatibility = probe()
    write_json(output / "compatibility.json", compatibility)
    modes = ["qlora", "fp16"] if config.mode == "auto" else [config.mode]
    attempts = []
    for mode in modes:
        if compatibility[mode]["status"] != "passed":
            attempts.append({"mode": mode, "status": "skipped", "error": compatibility[mode].get("error")})
            continue
        effective = config.with_mode(mode)
        config_path = output / f"{mode}-config.json"
        write_json(config_path, effective.to_dict())
        command = [sys.executable, "-m", "qwen_classifier", "_smoke-worker", "--config", str(config_path),
                   "--output-dir", str(output / mode), "--cache-dir", str(cache_dir)]
        log_path = output / f"{mode}.log"
        print(f"Testing {mode} on the complete model; progress: {log_path}", file=sys.stderr, flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           check=False, timeout=1800, start_new_session=True)
            path = output / mode / "run.json"
            result = json.loads(path.read_text()) if path.exists() and completed.returncode == 0 else {
                "status": "failed", "error": log_path.read_text()[-6000:]}
        except subprocess.TimeoutExpired:
            result = {"status": "failed", "error": "Real-model smoke test exceeded 1800 seconds."}
        attempts.append({"mode": mode, **result})
        if result["status"] == "passed":
            summary = {"status": "passed", "selected_mode": mode, "attempts": attempts,
                       "requested_config": config.to_dict(), "environment_sha256": digest(canonical(environment()))}
            write_json(output / "summary.json", summary)
            return summary
    summary = {"status": "failed", "selected_mode": None, "attempts": attempts,
               "requested_config": config.to_dict()}
    write_json(output / "summary.json", summary)
    raise RuntimeError(f"No training mode passed compatibility and real-model smoke tests. See {output}.")


def train(config: Config, source: Path, output: Path, cache_dir: Path, *, smoke_report: Path | None = None,
          pilot: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    if smoke_report is None:
        report = smoke_test(config, output.with_name(output.name + "-preflight"), cache_dir)
    else:
        report = json.loads(smoke_report.read_text(encoding="utf-8"))
        if (report.get("status") != "passed" or report.get("requested_config") != config.to_dict()
                or report.get("environment_sha256") != digest(canonical(environment()))):
            raise ValueError("Smoke report does not match this configuration and environment.")
        # A saved report cannot replace testing the currently exposed device and kernels.
        current = probe()
        if current[report["selected_mode"]]["status"] != "passed":
            raise RuntimeError("Current GPU kernels failed; repeat the real-model smoke test.")
        selected_attempt = next(a for a in report["attempts"] if a["mode"] == report["selected_mode"])
        if selected_attempt["device"] != current[report["selected_mode"]]["device"]:
            raise ValueError("GPU identity changed since the real-model smoke test.")
    effective = replace(config, mode=report["selected_mode"])
    try:
        result = fit(effective, output, cache_dir, source=source, pilot=pilot)
    except Exception as error:
        if output.exists():
            write_json(output / "failure.json", {"error": f"{type(error).__name__}: {error}",
                                                "config": effective.to_dict()})
        raise
    write_json(output / "preflight.json", report)
    return result
