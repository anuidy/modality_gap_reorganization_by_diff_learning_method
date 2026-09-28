"""Prepare only COCO Karpathy test images from the read-only AutoDL public ZIPs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COCO_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path, default=Path('/root/autodl-pub/COCO2017'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    annotation = ROOT / 'data/raw/evaluation_metadata/karpathy/dataset_coco.json'
    raw = annotation.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COCO_SHA: raise ValueError('Canonical COCO annotation changed')
    test = [x for x in json.loads(raw)['images'] if x['split'] == 'test']
    ids = {int(re.search(r'(\d+)\.jpg$', x['filename'])[1]): x['filename'] for x in test}
    if len(ids) != 5000 or sum(len(x['sentences']) for x in test) != 25010:
        raise ValueError('Expected the confirmed 5000 images and all 25010 captions')
    selected = []
    for filename in ('val2017.zip', 'train2017.zip'):
        path = args.public_root / filename
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                match = re.search(r'(\d+)\.jpg$', member.filename)
                if match and int(match[1]) in ids:
                    selected.append((path, member, ids[int(match[1])]))
    if len(selected) != 5000 or len({x[2] for x in selected}) != 5000:
        raise ValueError('Public archives do not cover the complete test split exactly once')
    dest = ROOT / 'data/raw/coco_karpathy'
    print(json.dumps({'images': 5000, 'captions': 25010,
                      'image_bytes': sum(x[1].file_size for x in selected), 'execute': args.execute}), flush=True)
    if not args.execute: return
    if dest.exists(): raise FileExistsError('Refusing to overwrite an existing prepared dataset')
    (dest / 'images').mkdir(parents=True)
    checks = []
    for path in dict.fromkeys(x[0] for x in selected):
        with zipfile.ZipFile(path) as archive:
            for _, member, filename in [x for x in selected if x[0] == path]:
                data = archive.read(member)  # ZIP CRC checked before writing.
                with Image.open(io.BytesIO(data)) as image:
                    image.load(); width, height = image.size; image.convert('RGB').load()
                target = dest / 'images' / filename
                target.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest: raise ValueError('Image readback failed')
                checks.append({'filename': filename, 'sha256': digest, 'bytes': len(data),
                               'width': width, 'height': height, 'source_archive': str(path), 'member': member.filename})
    (dest / 'annotations').mkdir()
    source = {'annotation_member_sha256': COCO_SHA, 'annotation_member': 'dataset_coco.json',
              'annotation_source': str(annotation.relative_to(ROOT)), 'caption_policy': 'all_canonical_raw_sentences'}
    selected_path = dest / 'annotations/karpathy_test.json'
    selected_path.write_text(json.dumps({'dataset': 'coco', 'split': 'test', 'image_root': '../images',
                                        'source': source, 'images': test}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checks_path = dest / 'image_checksums.jsonl'
    checks_path.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in sorted(checks, key=lambda x: x['filename'])))
    report = {'status': 'valid', 'image_count': 5000, 'caption_count': 25010,
              'all_images_decode': True, 'caption_policy': 'all_canonical_raw_sentences',
              'test_annotation_sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest(),
              'image_checksums_sha256': hashlib.sha256(checks_path.read_bytes()).hexdigest(), 'source': source}
    (dest / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
