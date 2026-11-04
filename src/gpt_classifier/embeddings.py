"""Local, pinned E5 embeddings. Only prepare_model may access the network."""
from __future__ import annotations

import importlib.metadata
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

from .data import canonical, clean, digest, file_digest

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
FILES = ("onnx/model.onnx", "tokenizer.json")
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".cache/retrieval"


def model_files(cache_dir: Path, *, download: bool = False):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError('Recuperação semântica exige: pip install -e ".[rag]"') from exc
    try:
        return [Path(hf_hub_download(MODEL, filename=name, revision=REVISION,
                                    cache_dir=str(cache_dir / "models"),
                                    local_files_only=not download, token=False)) for name in FILES]
    except OSError as exc:
        raise ValueError("Modelo local ausente/incompleto. Execute prepare-retrieval antes da classificação.") from exc


def prepare_model(cache_dir: Path = DEFAULT_CACHE):
    paths = model_files(cache_dir, download=True)
    return {"model": MODEL, "revision": REVISION,
            "files_sha256": {name: file_digest(path) for name, path in zip(FILES, paths)}}


def validate_vectors(vectors, rows: int):
    value = np.asarray(vectors, dtype=np.float32)
    if value.shape != (rows, 384) or not np.isfinite(value).all():
        raise ValueError("Embeddings inválidos: dimensão/valores inesperados.")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Embedding nulo no cache ou encoder.")
    if not np.allclose(norms, 1, atol=1e-4):
        raise ValueError("Embedding sem normalização L2 no cache ou encoder.")
    # Never renormalize cache reads: cold and warm runs must return identical bits.
    return value


class LocalEmbeddings:
    """CPU inference, query prefix on both sides for symmetric example retrieval.

    Per-text cache is independent of labels, row IDs and the train/validation split.
    Entries commit after each encoder batch, so interruption preserves progress.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE, *, batch_size: int = 16):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError('Recuperação semântica exige: pip install -e ".[rag]"') from exc
        ort.disable_telemetry_events()

        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size do encoder deve ser positivo.")
        self.cache_dir, self.batch_size = Path(cache_dir), batch_size
        model_path, tokenizer_path = model_files(self.cache_dir)
        self.metadata = {"model": MODEL, "revision": REVISION, "format": FILES[0],
                         "files_sha256": {name: file_digest(path) for name, path in zip(FILES, (model_path, tokenizer_path))},
                         "prefix": "query: ", "max_tokens": 512, "pooling": "masked_mean_l2",
                         "provider": "CPUExecutionProvider", "threads": 4,
                         "packages": {p: importlib.metadata.version(p) for p in ("onnxruntime", "tokenizers", "numpy")}}
        self.namespace = digest(canonical(self.metadata))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.model_path = model_path
        self.session = None  # A warm cache never needs to load the neural model.
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 4
        self.options.inter_op_num_threads = 1
        self.stats = {"cache_hits": 0, "encoded_texts": 0}

    def _encode(self, texts):
        import onnxruntime as ort
        if self.session is None:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=self.options,
                                               providers=["CPUExecutionProvider"])
        encodings = self.tokenizer.encode_batch(["query: " + text for text in texts])
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        available = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        inputs = {item.name: available[item.name] for item in self.session.get_inputs()}
        hidden = self.session.run(None, inputs)[0]
        if hidden.shape != (*ids.shape, 384):
            raise ValueError("Saída ONNX incompatível com pooling E5.")
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return validate_vectors(pooled / np.maximum(norms, 1e-12), len(texts))

    def encode(self, texts: list[str]):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        normalized = [clean(text) for text in texts]
        unique = list(dict.fromkeys(normalized))
        values, missing = {}, []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_dir / "embeddings.sqlite3", timeout=60)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            for text in unique:
                key = digest(canonical([self.namespace, text]))
                cached = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
                if cached is None:
                    missing.append((text, key))
                else:
                    values[text] = validate_vectors(np.frombuffer(cached[0], dtype="<f4").reshape(1, -1), 1)[0]
                    self.stats["cache_hits"] += 1
            # Sorting by length reduces padding; output order is restored below.
            missing.sort(key=lambda item: (len(item[0]), item[0]))
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                vectors = validate_vectors(self._encode([t for t, _ in batch]), len(batch))
                for (text, key), vector in zip(batch, vectors):
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, vector.astype("<f4").tobytes()))
                    values[text] = vector
                db.commit()
                self.stats["encoded_texts"] += len(batch)
                done = start + len(batch)
                if start == 0 or done % 256 == 0 or done == len(missing):
                    print(f"Embeddings locais: {done}/{len(missing)} novos; {self.stats['cache_hits']} cache", file=sys.stderr)
        return np.stack([values[text] for text in normalized])
