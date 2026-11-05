"""Sample-normalized classification loss for explicit full-corpus refitting."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def sample_normalized_loss(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
    """Average summed cross entropy over all examples in an accumulated update."""
    import torch
    from torch.nn import functional

    count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(count):
        count = count.to(outputs.logits.device)
    if count <= 0:
        raise ValueError("Classification loss requires at least one labeled example.")
    return functional.cross_entropy(outputs.logits.float(), labels, reduction="sum") / count


@contextmanager
def sample_normalized_trainer(module: Any) -> Iterator[None]:
    """Inject explicit loss into the legacy runner without changing EP1 sources."""
    previous = module.Trainer

    class SampleNormalizedTrainer(previous):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("compute_loss_func") is not None:
                raise ValueError("An additional custom loss cannot replace sample normalization.")
            kwargs["compute_loss_func"] = sample_normalized_loss
            super().__init__(*args, **kwargs)
            # ModernBERT accepts **kwargs but its native mean CE ignores batch counts.
            self.model_accepts_loss_kwargs = False

    module.Trainer = SampleNormalizedTrainer
    try:
        yield
    finally:
        module.Trainer = previous
