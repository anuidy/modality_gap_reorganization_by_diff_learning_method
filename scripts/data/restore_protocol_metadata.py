"""Restore small immutable protocol descriptors; never download datasets/models."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_FILES = (
    ("configs/data/lcs_558k_split_lock_v1.json", "data/splits/lcs_558k_split_lock_v1.json"),
    ("configs/resources/checksums.sha256", "data/metadata/checksums.sha256"),
)


def restore_protocol_metadata(project_root: Path) -> list[str]:
    root = project_root.resolve()
    pending = []
    for source_name, target_name in PROTOCOL_FILES:
        source, target = root / source_name, root / target_name
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Protocol destination escapes project: {target}")
        expected = source.read_bytes()
        if target.exists():
            if target.read_bytes() != expected:
                raise ValueError(f"Existing protocol file differs; refusing overwrite: {target}")
        else:
            pending.append((source, target))
    written = []
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        written.append(str(target.relative_to(root)))
    return written


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    written = restore_protocol_metadata(PROJECT_ROOT)
    print(f"Protocol metadata ready ({len(written)} files restored). Datasets/models were not downloaded.")


if __name__ == "__main__":
    main()
