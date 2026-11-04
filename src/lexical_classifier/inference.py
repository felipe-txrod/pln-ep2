"""Load frozen TF-IDF pipelines without fitting or reconstructing weights."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("c1", "c234", "c5")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize_text(text: str) -> str:
    """Match EP1 normalization for the string-only inference contract."""
    if not isinstance(text, str):
        raise TypeError("Lexical inference accepts strings only.")
    return re.sub(r"\s+", " ", html.unescape(text)).strip().lower()


def load_legacy_artifact(path: Path) -> dict[str, Any]:
    """Remap the original preprocessor without importing the EP1 project."""
    from joblib.numpy_pickle import NumpyUnpickler
    from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap

    class LegacyUnpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) == ("clarity.data", "normalize_text"):
                return normalize_text
            if module == "clarity" or module.startswith("clarity."):
                raise ValueError(f"Unsupported EP1 serialization reference: {module}.{name}")
            return super().find_class(module, name)

    with path.open("rb") as stream:
        with _validate_fileobject_and_memmap(stream, str(path), None) as (handle, mmap_mode):
            if isinstance(handle, str):
                raise ValueError("Legacy joblib formats older than 0.10 are unsupported.")
            return LegacyUnpickler(str(path), handle, ensure_native_byte_order=True,
                                   mmap_mode=mmap_mode).load()


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float] | None = None
    decision_scores: dict[str, float] | None = None


class FrozenLexicalClassifier:
    """CPU-only, lazy inference for one hash-pinned classical artifact."""

    def __init__(self, manifest_path: Path, artifact: Path | None = None, *, threads: int = 2):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer.")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = self.manifest["model_type"]
        if kind not in {"svm", "rl"} or self.manifest["labels"] != list(LABELS):
            raise ValueError("Unsupported frozen lexical contract.")
        bundle_dir = os.environ.get(f"{kind.upper()}_BUNDLE_DIR")
        project = Path(__file__).resolve().parents[2]
        published = project / "models" / kind
        root = (Path(bundle_dir).expanduser() if bundle_dir else
                published if (published / "manifest.json").is_file() else
                project / "artifacts" / f"{kind}-ep1")
        configured = os.environ.get(f"{kind.upper()}_ARTIFACT_PATH")
        self.artifact = (Path(artifact) if artifact is not None else
                         Path(configured).expanduser() if configured else root / "model.joblib")
        self.bundle_dir = root
        self.threads = threads
        self._model: Any = None

    def describe(self) -> dict[str, Any]:
        return {"model_type": self.manifest["model_type"], "labels": list(LABELS),
                "artifact_sha256": self.manifest["artifact_sha256"],
                "artifact_present": self.artifact.is_file(),
                "probabilities": self.manifest["model_type"] == "rl",
                "decision_scores": True, "training_supported": False}

    def load(self) -> FrozenLexicalClassifier:
        if self._model is not None:
            return self
        if not self.artifact.is_file():
            raise FileNotFoundError(
                f"Missing trained {self.manifest['model_type']} artifact: {self.artifact}. "
                f"Restore the original EP1 file with SHA-256 {self.manifest['artifact_sha256']}; "
                "automatic training is disabled.")
        if file_digest(self.artifact) != self.manifest["artifact_sha256"]:
            raise ValueError("The artifact is not the hash-pinned EP1 model.")
        for name, expected in self.manifest["files"].items():
            if file_digest(self.bundle_dir / name) != expected:
                raise ValueError(f"Frozen lexical metadata changed: {name}")
        for package, expected in self.manifest["runtime_versions"].items():
            current = importlib.metadata.version(package)
            if current != expected:
                raise RuntimeError(f"{package}=={expected} is required by EP1; found {current}.")
        bundle = load_legacy_artifact(self.artifact)
        metadata = json.loads((self.bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        if (not isinstance(bundle, dict) or bundle.get("format_version") != 1
                or tuple(bundle.get("labels", ())) != LABELS
                or bundle.get("metadata") != metadata
                or metadata.get("selected") != self.manifest["selected"]):
            raise ValueError("The deserialized EP1 artifact has incompatible metadata.")
        model = bundle["model"]
        self._validate_model(model)
        self._model = model
        return self

    def _validate_model(self, model: Any) -> None:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC
        from sklearn.utils.validation import check_is_fitted

        if not isinstance(model, Pipeline) or tuple(model.named_steps) != ("features", "classifier"):
            raise ValueError("Expected the original fitted lexical Pipeline.")
        classifier = model.named_steps["classifier"]
        expected_type = LinearSVC if self.manifest["model_type"] == "svm" else LogisticRegression
        if not isinstance(classifier, expected_type):
            raise ValueError("The classifier type differs from the frozen contract.")
        check_is_fitted(classifier)
        parameters = classifier.get_params(deep=False)
        if any(parameters.get(key) != value for key, value in self.manifest["classifier_parameters"].items()):
            raise ValueError("Classifier parameters differ from the EP1 configuration.")
        if tuple(classifier.classes_) != LABELS:
            raise ValueError("The classifier has an incompatible class order.")
        if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
            raise ValueError("Classifier weights are nonfinite.")
        features = model.named_steps["features"]
        if not isinstance(features, FeatureUnion) or features.transformer_weights is not None:
            raise ValueError("The EP1 feature union changed.")
        vectors = dict(features.transformer_list)
        if set(vectors) != {"word", "char"}:
            raise ValueError("Both fitted word and character vectorizers are required.")
        for name, vectorizer in vectors.items():
            check_is_fitted(vectorizer)
            if vectorizer.preprocessor is not normalize_text or np.dtype(vectorizer.dtype) != np.dtype("float32"):
                raise ValueError("The original text preprocessor or feature dtype changed.")
            values = vectorizer.get_params(deep=False)
            for key, expected in self.manifest[name].items():
                if key == "ngram_range":
                    expected = tuple(expected)
                if values.get(key) != expected:
                    raise ValueError(f"The {name} vectorizer parameter changed: {key}")
        model.memory = None

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("Pass a sequence of strings, not a single string.")
        responses = list(texts)
        if not all(isinstance(text, str) for text in responses):
            raise TypeError("All responses must be strings.")
        if not responses:
            return []
        self.load()
        import numpy as np
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            labels = self._model.predict(responses)
            scores = np.asarray(self._model.decision_function(responses))
            probabilities = (np.asarray(self._model.predict_proba(responses))
                             if self.manifest["model_type"] == "rl" else None)
        shape = (len(responses), len(LABELS))
        if scores.shape != shape or not np.isfinite(scores).all() or len(labels) != len(responses):
            raise FloatingPointError("Invalid classifier output shape or decision scores.")
        if probabilities is not None and (
                probabilities.shape != shape or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
            raise FloatingPointError("Invalid logistic-regression probabilities.")
        output = []
        for index, label in enumerate(labels):
            if label not in LABELS:
                raise ValueError("The classifier returned an unknown label.")
            probability = None if probabilities is None else dict(zip(LABELS, probabilities[index].tolist()))
            output.append(Prediction(str(label), probability, dict(zip(LABELS, scores[index].tolist()))))
        return output
