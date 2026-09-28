"""``--resume`` accepts either a checkpoint file or the pointer beside it.

Every run writes ``checkpoints/resume/latest.json`` next to its checkpoints, and
the launch scripts print that pointer as the recovery command; the trainer must
therefore understand it as well as a plain ``.pt`` path.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.training.train import resolve_resume_path  # noqa: E402


class ResumePointerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = self.root / "step_00016879.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.pointer = self.root / "latest.json"
        self.pointer.write_text(
            json.dumps(
                {
                    "completed_steps": 16879,
                    "path": self.checkpoint.name,
                    "artifact_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )

    def test_pointer_resolves_to_the_checkpoint_next_to_it(self):
        self.assertEqual(resolve_resume_path(self.pointer), self.checkpoint.resolve())

    def test_plain_checkpoint_path_is_returned_unchanged(self):
        self.assertEqual(resolve_resume_path(self.checkpoint), self.checkpoint.resolve())

    def test_dangling_pointer_fails_loudly(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            resolve_resume_path(self.pointer)


if __name__ == "__main__":
    unittest.main()
