"""Stable application paths independent of the shell's current directory."""

from pathlib import Path


def repository_root():
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src/bibazu_autotune_training").is_dir():
        return root
    return None


def default_output_directory():
    root = repository_root()
    if root is not None:
        return root / "data/experiments"
    return Path.home() / "Documents/BiBaZu_AutoTune_Training/experiments"
