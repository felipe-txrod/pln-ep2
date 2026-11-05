"""Frozen EP1 logistic regression with word and character TF-IDF, C=0.25."""
from pathlib import Path
import os

from lexical_classifier import FrozenLexicalClassifier, Prediction


class RLClassifier(FrozenLexicalClassifier):
    def __init__(self, artifact: Path | None = None, *, threads: int = 2):
        default = Path(__file__).resolve().parents[2] / "models/rl-development"
        bundle = Path(os.environ.get("RL_BUNDLE_DIR", str(default))).expanduser()
        manifest = bundle / "frozen_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                "The development-only RL model is not available. Run bash scripts/rl/train.sh "
                "to fit 16,093 development rows while reserving the 3,999 holdout rows. "
                "The historical full-corpus model is not selected automatically.")
        super().__init__(manifest, artifact, threads=threads, bundle_dir=bundle)


__all__ = ["RLClassifier", "Prediction"]
