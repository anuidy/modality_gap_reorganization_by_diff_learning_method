from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.probes import create_coco_2017_val_manifest  # noqa: E402


def main() -> None:
    manifest = create_coco_2017_val_manifest(
        captions_path=PROJECT_ROOT
        / "data/raw/coco_2017_val/extracted/annotations/captions_val2017.json",
        image_directory=PROJECT_ROOT / "data/raw/coco_2017_val/extracted/val2017",
        destination=PROJECT_ROOT / "data/splits/coco_2017_val_probe_v1.json",
        project_root=PROJECT_ROOT,
    )
    print(f"Created {manifest.path}")
    print(f"samples={len(manifest.samples)}")
    print(f"sha256={manifest.sha256}")


if __name__ == "__main__":
    main()
