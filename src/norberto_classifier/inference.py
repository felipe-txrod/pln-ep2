"""Load the already trained EP1 adapter for inference; no training API is exposed."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import LABELS, FrozenBundle, load_bundle
from .tokenization import tokenize_texts


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]


class NorbertoClassifier:
    """Lazy, offline, FP32 inference using the original EP1 label mapping."""

    def __init__(self, artifact_dir: Path | None = None, *, device: str = "cpu"):
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu; GPU use requires an explicit choice.")
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.device = device
        self.bundle: FrozenBundle | None = None
        self.tokenization_statistics: dict = {}
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> NorbertoClassifier:
        """Load verified local weights only when explicitly requested by inference."""
        if self._model is not None:
            return self
        bundle = load_bundle(self.artifact_dir)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

        if self.device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("An explicit GPU was requested but no CUDA/ROCm GPU is available.")
        set_seed(bundle.config["seed"])
        torch.set_num_threads(bundle.config["threads"])
        tokenizer = AutoTokenizer.from_pretrained(bundle.adapter, local_files_only=True,
                                                 trust_remote_code=False)
        base = AutoModelForSequenceClassification.from_pretrained(
            bundle.base_model, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
            label2id={label: index for index, label in enumerate(LABELS)},
            attn_implementation=bundle.config["attention"], reference_compile=False,
            torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, bundle.adapter, is_trainable=False, local_files_only=True)
        model.requires_grad_(False)
        model.to("cuda" if self.device == "gpu" else "cpu")
        model.eval()
        self.bundle, self._model, self._tokenizer = bundle, model, tokenizer
        return self

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("Pass a sequence of response strings, not a single string.")
        responses = list(texts)
        if not all(isinstance(text, str) for text in responses):
            raise TypeError("Every response must be a string.")
        if not responses:
            return []
        self.load()
        import torch
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        assert self.bundle is not None
        features, self.tokenization_statistics = tokenize_texts(responses, self._tokenizer, self.bundle.config)
        loader = DataLoader(features, batch_size=self.bundle.config["eval_batch_size"], shuffle=False,
                            num_workers=0, pin_memory=False,
                            collate_fn=DataCollatorWithPadding(self._tokenizer, pad_to_multiple_of=8))
        predictions = []
        device = "cuda" if self.device == "gpu" else "cpu"
        with torch.inference_mode():
            for batch in loader:
                logits = self._model(**{name: value.to(device) for name, value in batch.items()}).logits.float()
                if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not torch.isfinite(logits).all().item():
                    raise FloatingPointError("NorBERTo produced invalid or nonfinite logits.")
                probabilities = logits.softmax(-1).cpu().tolist()
                for scores in probabilities:
                    winner = max(range(len(LABELS)), key=scores.__getitem__)
                    predictions.append(Prediction(LABELS[winner], dict(zip(LABELS, scores))))
        if len(predictions) != len(responses):
            raise RuntimeError("The number of predictions does not match the input rows.")
        return predictions
