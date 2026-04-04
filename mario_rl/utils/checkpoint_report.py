"""Checkpoint inspection helpers."""
from __future__ import annotations

from pathlib import Path


def list_checkpoint_files(checkpoint_dir: str | Path) -> list[str]:
    """Return checkpoint file names sorted by name."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    return sorted(file.name for file in path.glob("*.pt") if file.is_file())
