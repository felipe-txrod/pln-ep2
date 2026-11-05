"""Regression checks for optimizer units, unequal tails, and context cleanup."""
from types import SimpleNamespace

import pytest

from norberto_classifier.training_recipe import compatible_normalized_trainer


@pytest.mark.parametrize("sizes", [(4, 4, 4, 4), (4, 4, 4, 1), (4, 1)])
@pytest.mark.parametrize("logit_scale", [0.01, 4.0])
def test_optimizer_updates_match_scaled_whole_batch_reference(tmp_path, sizes, logit_scale):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Trainer, TrainingArguments
    from transformers.modeling_outputs import SequenceClassifierOutput

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, input_ids, labels=None, **kwargs):
            logits = self.linear(input_ids) * logit_scale
            loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)

    torch.manual_seed(42)
    model, reference = Classifier(), Classifier()
    reference.load_state_dict(model.state_dict())
    module = SimpleNamespace(Trainer=Trainer)
    with compatible_normalized_trainer(module):
        trainer = module.Trainer(model=model, args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, gradient_accumulation_steps=4,
            report_to=[], dataloader_pin_memory=False))
    assert module.Trainer is Trainer
    trainer.current_gradient_accumulation_steps = len(sizes)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01) for m in (model, reference)]
    for _ in range(3):
        x, labels = torch.randn(sum(sizes), 3), torch.arange(sum(sizes)) % 3
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected = torch.nn.functional.cross_entropy(reference(x).logits, labels)
        (expected * 4).backward()
        observed, offset = 0, 0
        for size in sizes:
            observed += trainer.training_step(model, {"input_ids": x[offset:offset+size],
                                                       "labels": labels[offset:offset+size]},
                                              num_items_in_batch=torch.tensor(sum(sizes)))
            offset += size
        torch.testing.assert_close(observed, expected.detach())
        for actual, desired in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, desired.grad, atol=1e-6, rtol=1e-5)
        for current, optimizer in zip((model, reference), optimizers):
            torch.nn.utils.clip_grad_norm_(current.parameters(), 1)
            optimizer.step()
        for actual, desired in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, desired)
    evaluation = trainer.compute_loss(model, {"input_ids": x, "labels": labels})
    torch.testing.assert_close(evaluation, torch.nn.functional.cross_entropy(model(x).logits, labels))
    assert trainer._optimizer_loss_scale == 1
    with pytest.raises(ValueError, match="accumulated"):
        trainer.training_step(model, {"input_ids": x, "labels": labels})
