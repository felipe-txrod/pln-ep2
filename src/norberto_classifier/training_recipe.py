"""Preserve EP1 optimizer gradient units with sample-normalized accumulation."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .training_loss import sample_normalized_trainer


@contextmanager
def compatible_normalized_trainer(module: Any) -> Iterator[None]:
    """Normalize example weights while retaining legacy clipping/AdamW units.

    Legacy EP1 accumulates one mean loss per microbatch. For equal batches its
    gradient is K times the example mean, where K is configured accumulation.
    Keep that optimizer scale explicitly, including at partial accumulation
    boundaries. Report the example mean and leave evaluation loss unscaled.
    """
    with sample_normalized_trainer(module):
        previous = module.Trainer

        class CompatibleNormalizedTrainer(previous):
            _optimizer_loss_scale = 1

            def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False,
                             num_items_in_batch: Any = None) -> Any:
                result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                              num_items_in_batch=num_items_in_batch)
                if return_outputs:
                    loss, outputs = result
                    return loss * self._optimizer_loss_scale, outputs
                return result * self._optimizer_loss_scale

            def training_step(self, model: Any, inputs: Any,
                              num_items_in_batch: Any = None) -> Any:
                if num_items_in_batch is None:
                    raise ValueError("Training requires the accumulated labeled-example count.")
                scale = self.args.gradient_accumulation_steps
                if scale < 1:
                    raise ValueError("Gradient accumulation must be positive.")
                self._optimizer_loss_scale = scale
                try:
                    result = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
                    return result / scale
                finally:
                    self._optimizer_loss_scale = 1

        module.Trainer = CompatibleNormalizedTrainer
        try:
            yield
        finally:
            module.Trainer = previous
