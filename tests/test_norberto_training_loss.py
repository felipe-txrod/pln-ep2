from __future__ import annotations

from types import SimpleNamespace

import pytest

from norberto_classifier.training_loss import sample_normalized_loss, sample_normalized_trainer


@pytest.mark.parametrize("sizes", [(4, 4, 4, 4), (4, 4, 4, 1), (4, 4, 4), (1, 1, 1), (3,)])
def test_sample_loss_matches_complete_batch_loss_and_gradients(sizes):
    torch = pytest.importorskip("torch")
    count = sum(sizes)
    logits = torch.linspace(-2, 3, count * 3).reshape(count, 3).requires_grad_()
    labels = torch.arange(count) % 3
    reference = torch.nn.functional.cross_entropy(logits, labels)
    expected, = torch.autograd.grad(reference, logits)
    offset = 0
    loss = 0
    for size in sizes:
        loss = loss + sample_normalized_loss(SimpleNamespace(logits=logits[offset:offset + size]),
                                            labels[offset:offset + size], num_items_in_batch=count)
        offset += size
    actual, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(actual, expected)


def test_missing_batch_count_counts_labeled_examples_only():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1., 2., 3.], [3., 2., 1.], [1., 0., 2.]])
    labels = torch.tensor([0, -100, 2])
    actual = sample_normalized_loss(SimpleNamespace(logits=logits), labels)
    torch.testing.assert_close(actual, torch.nn.functional.cross_entropy(logits, labels))
    with pytest.raises(ValueError, match="at least one"):
        sample_normalized_loss(SimpleNamespace(logits=logits), labels, num_items_in_batch=0)


def test_trainer_injection_restores_legacy_constructor_after_failure():
    class Recorder:
        def __init__(self, **kwargs):
            self.loss = kwargs.get("compute_loss_func")
            self.model_accepts_loss_kwargs = True

    module = SimpleNamespace(Trainer=Recorder)
    with pytest.raises(RuntimeError, match="sentinel"):
        with sample_normalized_trainer(module):
            instance = module.Trainer()
            assert instance.loss is sample_normalized_loss
            assert instance.model_accepts_loss_kwargs is False
            raise RuntimeError("sentinel")
    assert module.Trainer is Recorder


def test_existing_custom_loss_is_rejected():
    module = SimpleNamespace(Trainer=object)
    with sample_normalized_trainer(module):
        with pytest.raises(ValueError, match="additional custom loss"):
            module.Trainer(compute_loss_func=lambda *args: None)


def test_real_modernbert_trainer_accumulation_uses_corrected_loss(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, TaskType, get_peft_model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = transformers.ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, max_position_embeddings=64, local_attention=16,
        pad_token_id=5, num_labels=3, reference_compile=False,
        classifier_dropout=0, attention_dropout=0, mlp_dropout=0, embedding_dropout=0)
    config._attn_implementation = "sdpa"
    model = get_peft_model(transformers.ModernBertForSequenceClassification(config), LoraConfig(
        task_type=TaskType.SEQ_CLS, r=4, lora_alpha=8, lora_dropout=0, target_modules=["Wqkv"]))
    tokens, labels = torch.randint(6, 64, (13, 8)), torch.arange(13) % 3
    module = SimpleNamespace(Trainer=transformers.Trainer)
    with sample_normalized_trainer(module):
        trainer = module.Trainer(model=model, args=transformers.TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, gradient_accumulation_steps=4,
            report_to=[], dataloader_pin_memory=False))
    trainer.current_gradient_accumulation_steps = 4
    reference = torch.nn.functional.cross_entropy(model(input_ids=tokens).logits, labels)
    reference.backward()
    expected = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    loss = 0
    for start, end in ((0, 4), (4, 8), (8, 12), (12, 13)):
        loss = loss + trainer.training_step(model, {"input_ids": tokens[start:end], "labels": labels[start:end]},
                                           num_items_in_batch=torch.tensor(13))
    torch.testing.assert_close(loss, reference.detach(), atol=1e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], atol=1e-6, rtol=1e-4)
