from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gpt_classifier.data import LABELS, Example, Text, group
from qwen_classifier.config import Config
from qwen_classifier.dataset import TokenizedTexts, prepare, split_records
from qwen_classifier.training import classification_metrics, smoke_test, train


@pytest.mark.parametrize("updates", [
    {"revision": "main"}, {"model_id": "other/model"}, {"max_length": 0},
    {"mode": "gguf"}, {"rank": True}, {"accumulation_steps": 0},
    {"learning_rate": float("nan")}, {"dropout": 1.0}, {"fold": 5},
    {"target_modules": ("score",)}, {"target_modules": ("q_proj", "q_proj")},
    {"gradient_checkpointing": "false"}, {"eval_batch_size": 0}, {"group_by_length": 1},
])
def test_invalid_config_is_rejected(updates):
    with pytest.raises(ValueError):
        replace(Config(), **updates)


def test_grouped_split_keeps_duplicates_and_conflicting_labels_together():
    records = [Example(i + 2, f"independent response {i}", LABELS[i % 3]) for i in range(90)]
    records += [Example(92, "  SHARED &amp; RESPONSE ", "c1"),
                Example(93, "shared & response", "c5")]
    first, second = split_records(records, Config())
    assert len(first) + len(second) == len(records)
    assert not {group(r.text) for r in first} & {group(r.text) for r in second}
    assert any({92, 93} <= {r.excel_row for r in partition} for partition in (first, second))
    assert split_records(records, Config()) == (first, second)


def test_prepare_preserves_excel_rows_and_refuses_overwrite(make_xlsx, tmp_path):
    rows = [(f"response {i}", LABELS[i % 3]) for i in range(90)]
    source = make_xlsx("training.xlsx", rows)
    output = tmp_path / "prepared"
    manifest = prepare(source, output, Config())
    assert sorted(manifest["train_rows"] + manifest["validation_rows"]) == list(range(2, 92))
    with pytest.raises(FileExistsError):
        prepare(source, output, Config())


def test_insufficient_groups_fail_before_training():
    records = [Example(i + 2, LABELS[i % 3], LABELS[i % 3]) for i in range(90)]
    with pytest.raises(ValueError, match="independent groups"):
        split_records(records, Config())


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text)))


def test_labels_never_enter_tokens_and_empty_text_has_a_valid_final_token():
    records = [Example(2, "", "c1"), Example(3, "abcdefghijk", "c5")]
    data = TokenizedTexts(records, FakeTokenizer(), 8)
    assert data[0] == {"input_ids": [999], "attention_mask": [1], "labels": 0}
    assert data[1]["input_ids"] == [0, 1, 2, 3, 4, 5, 6, 999]
    assert data.truncated == 1
    inference = TokenizedTexts([Text(2, "")], FakeTokenizer(), 8, labeled=False)
    assert "labels" not in inference[0]


def test_metrics_include_absent_classes_and_confusion_matrix_orientation():
    metrics = classification_metrics([0, 0, 1], [0, 1, 1])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["f1_macro"] == pytest.approx(4 / 9)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert metrics["precision_macro"] == pytest.approx(0.5)
    assert metrics["recall_macro"] == pytest.approx(0.5)


def test_smoke_auto_uses_fp16_after_nf4_kernel_failure(monkeypatch, tmp_path):
    import qwen_classifier.training as training
    monkeypatch.setattr(training, "probe", lambda: {"fp16": {"status": "passed"},
                                                  "qlora": {"status": "failed", "error": "unsupported kernel"}})
    def run(command, **kwargs):
        from types import SimpleNamespace
        output = tmp_path / "smoke" / "fp16"
        output.mkdir()
        (output / "run.json").write_text(json.dumps({"status": "passed"}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(training.subprocess, "run", run)
    result = smoke_test(Config(), tmp_path / "smoke", tmp_path)
    assert result["selected_mode"] == "fp16"
    assert result["attempts"][0]["status"] == "skipped"


def test_forced_qlora_never_silently_falls_back(monkeypatch, tmp_path):
    import qwen_classifier.training as training
    monkeypatch.setattr(training, "probe", lambda: {"fp16": {"status": "passed"},
                                                  "qlora": {"status": "failed", "error": "unsupported kernel"}})
    with pytest.raises(RuntimeError, match="No training mode passed"):
        smoke_test(replace(Config(), mode="qlora"), tmp_path / "smoke", tmp_path)
    assert json.loads((tmp_path / "smoke" / "summary.json").read_text())["selected_mode"] is None


def test_stale_smoke_report_cannot_start_training(tmp_path):
    report = tmp_path / "summary.json"
    report.write_text(json.dumps({"status": "passed", "requested_config": Config().to_dict(),
                                 "environment_sha256": "different runtime"}))
    with pytest.raises(ValueError, match="configuration and environment"):
        train(Config(), tmp_path / "unused.xlsx", tmp_path / "training", tmp_path, smoke_report=report)
    assert not (tmp_path / "training").exists()


def test_official_test1_requires_completed_test2_with_the_same_adapter(monkeypatch, tmp_path):
    import qwen_classifier.inference as inference
    monkeypatch.setattr(inference, "adapter_identity", lambda _: "frozen-adapter")
    source = Path(__file__).resolve().parents[1] / "data" / "test1.xlsx"
    with pytest.raises(ValueError, match="phase=statistical"):
        inference.predict(tmp_path, tmp_path, source, tmp_path / "output")
    earlier = tmp_path / "test2"
    earlier.mkdir()
    (earlier / "run.json").write_text(json.dumps({"status": "completed", "adapter_sha256": "other-adapter"}))
    with pytest.raises(ValueError, match="same saved adapter"):
        inference.predict(tmp_path, tmp_path, source, tmp_path / "output",
                          phase="statistical", test2_run=earlier)
    assert not (tmp_path / "output").exists()


def test_inference_does_not_read_the_gold_column(monkeypatch, make_xlsx, tmp_path):
    import qwen_classifier.hardware as hardware
    import qwen_classifier.inference as inference
    from gpt_classifier.data import digest
    source = make_xlsx("inference.xlsx", [("response", "this is not a valid label")])
    monkeypatch.setattr(inference, "adapter_identity", lambda _: "frozen-adapter")
    monkeypatch.setattr(hardware, "isolated_probe", lambda _: {"status": "passed"})
    def predict_records(adapter, cache_dir, records):
        assert all(isinstance(record, Text) and not hasattr(record, "label") for record in records)
        return [{"excel_row": r.excel_row, "text_sha256": digest(r.text), "label": "c1"} for r in records]
    monkeypatch.setattr(inference, "predict_records", predict_records)
    result = inference.predict(tmp_path, tmp_path, source, tmp_path / "output")
    assert result["status"] == "completed"
    assert result["rows"] == 1


def test_jsonl_roundtrip_preserves_unicode_line_separators(tmp_path):
    from gpt_classifier.data import read_jsonl, write_jsonl
    rows = [{"text": "first\u2028second\u2029third\u0085fourth\nlast", "label": "c1"}]
    target = tmp_path / "records.jsonl"
    write_jsonl(target, rows)
    assert read_jsonl(target) == rows


@pytest.mark.parametrize("batches,steps", [([3], 3), ([0], 3), ([4], 1)])
def test_benchmark_rejects_misleading_effective_batch_or_timing(batches, steps, tmp_path):
    pytest.importorskip("torch")
    from qwen_classifier.benchmark import benchmark
    with pytest.raises(ValueError, match="effective batch 16"):
        benchmark(replace(Config(), mode="qlora"), tmp_path / "output", tmp_path,
                  batches=batches, checkpointing=False, measured_steps=steps)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("sizes", [(1, 1, 1, 1, 1), (2, 2, 1), (4, 1), (5,)])
def test_accumulated_classification_loss_and_gradients_are_microbatch_invariant(sizes):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from qwen_classifier.training import classification_loss
    logits = torch.tensor([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.],
                           [0., 2., 1.], [1., 0., 2.]], requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1, 0])
    expected = torch.nn.functional.cross_entropy(logits, labels)
    expected_gradient, = torch.autograd.grad(expected, logits)
    offset = 0
    total = 0
    for size in sizes:
        total = total + classification_loss(SimpleNamespace(logits=logits[offset:offset + size]),
                                             labels[offset:offset + size], num_items_in_batch=len(labels))
        offset += size
    actual_gradient, = torch.autograd.grad(total, logits)
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_trainer_accumulation_matches_large_batches_with_an_incomplete_tail(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import Trainer, TrainingArguments
    from transformers.modeling_outputs import SequenceClassifierOutput
    from qwen_classifier.training import classification_loss

    class TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.arange(9).reshape(3, 3) / 10)

        def forward(self, input_ids, labels=None, **kwargs):
            return SequenceClassifierOutput(logits=self.linear(input_ids.float()))

    records = [{"input_ids": [i / 10, 1., -0.5], "labels": i % 3} for i in range(6)]
    weights = []
    for batch_size, accumulation in ((1, 4), (2, 2), (4, 1)):
        model = TinyClassifier()
        args = TrainingArguments(
            output_dir=str(tmp_path / str(batch_size)), use_cpu=True, num_train_epochs=1,
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            save_strategy="no", report_to="none", disable_tqdm=True, seed=42,
            dataloader_pin_memory=False, max_grad_norm=0)
        trainer = Trainer(model=model, args=args, train_dataset=records,
                          compute_loss_func=classification_loss,
                          optimizers=(torch.optim.SGD(model.parameters(), lr=0.1), None))
        trainer.model_accepts_loss_kwargs = False
        trainer.train()
        weights.append(model.linear.weight.detach().clone())
    for actual in weights[1:]:
        torch.testing.assert_close(actual, weights[0], atol=1e-7, rtol=1e-6)
