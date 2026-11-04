"""Offline ensembles with independently selectable members and arbitrator."""
from .config import EnsembleConfig
from .strategies import local_decision

__all__ = ["EnsembleConfig", "local_decision"]
