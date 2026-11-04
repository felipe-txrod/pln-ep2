"""Logistic-regression inference entry point."""
from lexical_classifier.cli import main as lexical_main
from . import RLClassifier


def main(argv: list[str] | None = None) -> int:
    return lexical_main(RLClassifier, argv)
