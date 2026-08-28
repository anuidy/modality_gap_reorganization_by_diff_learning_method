from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.factory import create_m0_adapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one M0 adapter on an image and text.")
    parser.add_argument("--model", choices=("clip", "vista", "albef", "beit3"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    adapter = create_m0_adapter(args.model, PROJECT_ROOT, args.device)
    image_path = PROJECT_ROOT / "data/raw/coco_2017_val/extracted/val2017/000000000139.jpg"
    with Image.open(image_path) as image:
        image_embedding = adapter.encode_image([image.convert("RGB").copy()])
    text_embedding = adapter.encode_text(["a test image"])
    print(f"image_shape={tuple(image_embedding.shape)}")
    print(f"text_shape={tuple(text_embedding.shape)}")
    print(f"metadata={adapter.metadata()}")


if __name__ == "__main__":
    main()
