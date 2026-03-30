"""Checkpoint helpers for Mario RL scripts."""
from __future__ import annotations

from pathlib import Path
import shutil


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def archive_checkpoint(path: str | Path, archive_dir: str | Path) -> Path | None:
    """Move a checkpoint into an archive directory if it exists."""
    source = Path(path)
    if not source.exists():
        return None

    target_dir = ensure_directory(archive_dir)
    destination = target_dir / source.name
    if destination.exists():
        destination.unlink()
    shutil.move(str(source), str(destination))
    return destination
