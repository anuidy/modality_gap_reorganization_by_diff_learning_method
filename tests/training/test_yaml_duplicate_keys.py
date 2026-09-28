"""Guard: no YAML file under ``configs/`` may declare a duplicated mapping key.

YAML silently keeps the last occurrence of a duplicated key (PyYAML's default),
which once hid a dead ``albef:`` block in ``full_gcl_execute.yaml``: an earlier
null-valued one-liner was overridden by a complete block, so the file looked as
if it copied the ALBEF block verbatim while the first copy was inert. The
duplicate was removed on 2026-09-15; this test keeps the failure mode from
coming back, because a duplicated key is always an authoring mistake here.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class _StrictLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping with repeated keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"duplicated mapping key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class YamlDuplicateKeyTest(unittest.TestCase):
    def test_detector_rejects_a_duplicated_key(self):
        """The guard itself must fail on a known-bad document."""

        with self.assertRaisesRegex(ValueError, "duplicated mapping key"):
            yaml.load("models:\n  clip: 1\n  clip: 2\n", Loader=_StrictLoader)

    def test_config_yaml_files_have_no_duplicate_keys(self):
        files = sorted(CONFIG_ROOT.rglob("*.yaml")) + sorted(CONFIG_ROOT.rglob("*.yml"))
        self.assertTrue(files, "expected at least one YAML config under configs/")
        for path in files:
            with self.subTest(config=str(path.relative_to(PROJECT_ROOT))):
                try:
                    yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
                except ValueError as error:
                    self.fail(f"{path.name}: {error}")
                except yaml.YAMLError as error:
                    self.fail(f"{path.name}: {error}")


if __name__ == "__main__":
    unittest.main()
