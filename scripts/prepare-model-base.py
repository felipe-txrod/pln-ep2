"""Compatibility command for the frozen model loader's restoration instructions."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "models/prepare-base.py"), run_name="__main__")
