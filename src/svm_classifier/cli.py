"""SVM inference entry point."""
from lexical_classifier.cli import main as lexical_main
from . import SVMClassifier


def main(argv: list[str] | None = None) -> int:
    return lexical_main(SVMClassifier, argv)
