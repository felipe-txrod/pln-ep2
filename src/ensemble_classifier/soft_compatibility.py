"""Retain explicitly audited legacy ensembles while rejecting unknown code drift."""
from __future__ import annotations

from gpt_classifier.data import file_digest

LEGACY_ARTIFACTS = {
    "soft": "f39a2200ecbaf124b95c78e249087f2f12302aa4f46768db756d828863301bec",
    "stacking": "08b8433ce6407c5f08b4d0a3e94ad31f4dd68eb580dcc8abdf1d0ad500608e97",
}


def compatible_fit_code(path, artifact, current, kind):
    if artifact.get("fit_code") == current:
        return True
    # Only these immutable original three-member artifacts were reviewed for the extension.
    return artifact.get("version") == 1 and file_digest(path) == LEGACY_ARTIFACTS[kind]
