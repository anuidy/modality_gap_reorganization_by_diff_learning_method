from pathlib import Path
import importlib.util
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("restore_protocol_metadata", ROOT / "scripts/data/restore_protocol_metadata.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolMetadataRestoreTest(unittest.TestCase):
    def prepare(self, root):
        for source, _ in module.PROTOCOL_FILES:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / source).read_bytes())

    def test_restores_exact_bytes_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            self.assertEqual(len(module.restore_protocol_metadata(root)), 2)
            for source, target in module.PROTOCOL_FILES:
                self.assertEqual((root / source).read_bytes(), (root / target).read_bytes())
            self.assertEqual(module.restore_protocol_metadata(root), [])

    def test_different_existing_file_prevents_all_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            target = root / module.PROTOCOL_FILES[1][1]
            target.parent.mkdir(parents=True)
            target.write_bytes(b"different frozen identity")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                module.restore_protocol_metadata(root)
            self.assertEqual(target.read_bytes(), b"different frozen identity")
            self.assertFalse((root / module.PROTOCOL_FILES[0][1]).exists())
