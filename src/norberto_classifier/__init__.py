"""Inference-only access to the frozen NorBERTo classifier delivered in EP1."""

from .inference import NorbertoClassifier, Prediction

__all__ = ["NorbertoClassifier", "Prediction"]
