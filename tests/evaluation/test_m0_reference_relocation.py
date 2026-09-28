import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation import validate_m0_outputs as validator


class M0ReferenceRelocationTest(unittest.TestCase):
    """The original export recorded Windows paths; they must relocate onto the copy."""

    def test_stored_export_paths_relocate_onto_the_copied_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (
                root / "data/metadata/geometry_references"
                / "coco_2017_val_5k_d8133889432c_upper_triangle_pairs_v1.npz"
            )
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"pair index")
            relative = expected.relative_to(root)
            values = (
                "D:\\modality-gap-reorganization\\" + str(relative).replace("/", "\\"),
                "/original/project/" + relative.as_posix(),
                relative.as_posix(),
            )
            with patch.object(validator, "PROJECT_ROOT", root):
                for value in values:
                    with self.subTest(value=value):
                        resolved = validator.relocate_stored_path(value, expected)
                        self.assertEqual(resolved, expected)
                        self.assertEqual(resolved.read_bytes(), b"pair index")
                with self.assertRaises(ValueError):
                    validator.relocate_stored_path(
                        values[0].replace("coco_2017_val_5k", "other_probe"), expected
                    )


if __name__ == "__main__":
    unittest.main()
