"""Frozen fine-tuned NorBERTo features, computed once for all layer/pooling views."""
from __future__ import annotations

from contextlib import closing
import importlib.metadata
from pathlib import Path
import sqlite3
import sys

import numpy as np

from gpt_classifier.data import Text, canonical, clean, digest, file_digest
from norberto_classifier.artifacts import ADAPTER_SHA256, load_bundle


def normalize_vectors(values, rows: int, dimension: int = 768):
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (rows, dimension) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding shape or nonfinite entries.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero embeddings are invalid.")
    return vectors / norms


def pool_hidden(hidden, mask, mode: str):
    """Match classifier mean pooling; attended special tokens remain included."""
    if mode == "mean":
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    if mode == "cls":
        return hidden[:, 0]
    raise ValueError("Unsupported pooling.")


class FineTunedEmbeddings:
    def __init__(self, config, *, encoder=None, metadata=None):
        self.config, self.encoder = config, encoder
        self.views = tuple(f"layer{layer}_{pooling}" for layer in config.layers for pooling in config.poolings)
        self.cache = Path(config.norberto_cache)
        self.model = None
        if metadata is None:
            bundle = load_bundle(Path(config.norberto_artifact))
            metadata = {"adapter_sha256": ADAPTER_SHA256, "bundle_sha256": bundle.identity_sha256,
                        "base_weights_sha256": file_digest(bundle.base_model / "model.safetensors"),
                        "architecture": "ModernBert", "dimension": 768, "layers": list(config.layers),
                        "poolings": list(config.poolings), "final_layer": "last_hidden_state_after_final_norm",
                        "intermediate": "hidden_states[layer] before final_norm", "mean_special_tokens": "included",
                        "max_length": 512, "truncation": "head_tail", "attention": "sdpa", "precision": "float32",
                        "eval": True, "trainable_parameters": 0, "device": config.device,
                        "batch_size": config.batch_size, "seed": bundle.config["seed"],
                        "extractor_sha256": file_digest(Path(__file__)),
                        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}}
        self.metadata = metadata
        self.namespace = digest(canonical(metadata))
        self.stats = {"cache_hits": 0, "encoded_texts": 0}

    def _encode(self, texts):
        if self.encoder is not None:
            return self.encoder(texts)
        from norberto_classifier.inference import NorbertoClassifier
        from norberto_classifier.tokenization import tokenize_texts
        import torch
        from transformers import DataCollatorWithPadding

        if self.model is None:
            self.model = NorbertoClassifier(Path(self.config.norberto_artifact), device=self.config.device).load()
            self.model._model.eval()
            if any(parameter.requires_grad for parameter in self.model._model.parameters()):
                raise ValueError("Fine-tuned feature extraction must have no trainable parameters.")
        features, _ = tokenize_texts(texts, self.model._tokenizer, self.model.bundle.config)
        batch = DataCollatorWithPadding(self.model._tokenizer, pad_to_multiple_of=8)(features)
        device = "cuda" if self.config.device == "gpu" else "cpu"
        batch = {name: value.to(device) for name, value in batch.items()}
        # The installed classifier exposes pre-final-norm hidden_states[-1].
        # Use last_hidden_state for the final view to match its actual head input.
        backbone = self.model._model.get_base_model().model
        with torch.inference_mode():
            output = backbone(**batch, output_hidden_states=True, return_dict=True)
            result = {}
            for layer in self.config.layers:
                hidden = output.last_hidden_state if layer == backbone.config.num_hidden_layers else output.hidden_states[layer]
                for mode in self.config.poolings:
                    result[f"layer{layer}_{mode}"] = pool_hidden(hidden, batch["attention_mask"], mode).float().cpu().numpy()
        return result

    def encode(self, targets: list[Text]) -> dict[str, np.ndarray]:
        if any(type(target) is not Text for target in targets):
            raise TypeError("Embedding extraction accepts label-free Text objects only.")
        texts = [clean(target.text) for target in targets]
        unique = list(dict.fromkeys(texts))
        self.cache.mkdir(parents=True, exist_ok=True)
        values, missing = {}, []
        with closing(sqlite3.connect(self.cache / "embeddings.sqlite3", timeout=60)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (namespace TEXT, text_sha TEXT, view TEXT, vector BLOB, sha256 TEXT, PRIMARY KEY(namespace,text_sha,view))")
            for text in unique:
                cached = db.execute("SELECT view,vector,sha256 FROM embeddings WHERE namespace=? AND text_sha=?", (self.namespace, digest(text))).fetchall()
                if {row[0] for row in cached} == set(self.views):
                    loaded = {}
                    for view, blob, checksum in cached:
                        import hashlib
                        if hashlib.sha256(blob).hexdigest() != checksum:
                            raise ValueError("Corrupt NorBERTo embedding cache checksum.")
                        vector = np.frombuffer(blob, dtype="<f4").copy().reshape(1, -1)
                        if vector.shape != (1, 768) or not np.isfinite(vector).all() or not np.allclose(np.linalg.norm(vector), 1, atol=1e-5):
                            raise ValueError("Invalid normalized cached embedding.")
                        loaded[view] = vector[0]
                    values[text] = loaded
                    self.stats["cache_hits"] += 1
                else:
                    missing.append(text)
            missing.sort(key=lambda text: (len(text), text))
            for start in range(0, len(missing), self.config.batch_size):
                chunk = missing[start:start + self.config.batch_size]
                encoded = self._encode(chunk)
                if set(encoded) != set(self.views):
                    raise ValueError("Encoder returned an incomplete layer/pooling set.")
                normalized = {view: normalize_vectors(encoded[view], len(chunk)) for view in self.views}
                for index, text in enumerate(chunk):
                    import hashlib
                    values[text] = {}
                    for view in self.views:
                        vector = normalized[view][index]
                        blob = vector.astype("<f4").tobytes()
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)", (self.namespace, digest(text), view, blob, hashlib.sha256(blob).hexdigest()))
                        values[text][view] = vector
                db.commit()
                self.stats["encoded_texts"] += len(chunk)
                if start == 0 or (start + len(chunk)) % 256 == 0 or start + len(chunk) == len(missing):
                    print(f"NorBERTo embeddings: {start + len(chunk)}/{len(missing)} new; {self.stats['cache_hits']} cached", file=sys.stderr, flush=True)
        return {view: np.stack([values[text][view] for text in texts]) if texts else np.empty((0, 768), dtype=np.float32) for view in self.views}
