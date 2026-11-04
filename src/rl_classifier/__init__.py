"""Frozen EP1 logistic regression with word and character TF-IDF, C=0.25."""
from pathlib import Path

from lexical_classifier import FrozenLexicalClassifier, Prediction


class RLClassifier(FrozenLexicalClassifier):
    def __init__(self, artifact: Path | None = None, *, threads: int = 2):
        super().__init__(Path(__file__).with_name("frozen_manifest.json"), artifact, threads=threads)


__all__ = ["RLClassifier", "Prediction"]
