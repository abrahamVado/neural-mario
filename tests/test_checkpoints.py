"""Tests for checkpoint helper utilities."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mario_rl.utils.checkpoint_report import list_checkpoint_files
from mario_rl.utils.checkpoints import archive_checkpoint, ensure_directory

TMP_DIR = Path("C:/tmp")


class CheckpointHelperTests(unittest.TestCase):
    def test_ensure_directory_creates_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
            target = Path(tmp) / "nested" / "checkpoints"
            created = ensure_directory(target)
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())

    def test_archive_checkpoint_moves_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
            checkpoint = Path(tmp) / "latest.pt"
            checkpoint.write_text("model", encoding="utf-8")
            archive_dir = Path(tmp) / "archive"
            destination = archive_checkpoint(checkpoint, archive_dir)
            self.assertIsNotNone(destination)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(destination.exists())

    def test_list_checkpoint_files_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
            root = Path(tmp)
            (root / "b.pt").write_text("b", encoding="utf-8")
            (root / "a.pt").write_text("a", encoding="utf-8")
            (root / "note.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual(list_checkpoint_files(root), ["a.pt", "b.pt"])


if __name__ == "__main__":
    unittest.main()
