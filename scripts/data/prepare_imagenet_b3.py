"""Extract only ImageNet validation images and canonical labels from AutoDL public archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.append(str(ROOT / '_local/extended_eval_dependencies'))
from embeddings.artifact import sha256_file
from evaluation.extended.common import finish


def read_devkit(path):
    from scipy.io import loadmat
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        mat = next(x for x in members if x.name.endswith('/data/meta.mat'))
        labels = next(x for x in members if x.name.endswith('/data/ILSVRC2012_validation_ground_truth.txt'))
        raw = archive.extractfile(mat).read()
        records = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)['synsets']
        id_to_wnid = {int(x.ILSVRC2012_ID): str(x.WNID) for x in records if int(x.num_children) == 0}
        truth = [int(x) for x in archive.extractfile(labels).read().decode().split()]
    if len(id_to_wnid) != 1000 or len(truth) != 50000 or not set(truth) <= set(id_to_wnid):
        raise ValueError('Unexpected ImageNet devkit labels')
    return id_to_wnid, truth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/root/autodl-pub/ImageNet/ILSVRC2012'))
    p.add_argument('--output', type=Path, default=ROOT / 'data/raw/imagenet_b3')
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    devkit = args.source / 'ILSVRC2012_devkit_t12.tar.gz'; images = args.source / 'ILSVRC2012_img_val.tar'
    if not devkit.is_file() or not images.is_file(): raise FileNotFoundError('ImageNet validation archive/devkit missing')
    ids, truth = read_devkit(devkit)
    print(json.dumps({'images': 50000, 'classes': 1000, 'source_bytes': images.stat().st_size, 'execute': args.execute}))
    if not args.execute: return
    if args.output.exists(): raise FileExistsError('Refusing to overwrite a prepared ImageNet directory')
    (args.output / 'images').mkdir(parents=True)
    wnids = sorted(ids.values()); index = {w: i for i, w in enumerate(wnids)}; seen = set()
    with (args.output / 'validation.jsonl').open('w', encoding='utf-8') as manifest, tarfile.open(images) as archive:
        for member in archive:
            if not member.isfile(): continue
            name = Path(member.name).name
            if not name.startswith('ILSVRC2012_val_') or not name.endswith('.JPEG'): raise ValueError('Unexpected validation member')
            number = int(name.removeprefix('ILSVRC2012_val_').removesuffix('.JPEG'))
            if not 1 <= number <= 50000 or number in seen: raise ValueError('Duplicate/invalid validation image')
            data = archive.extractfile(member).read()
            with Image.open(io.BytesIO(data)) as image: image.load(); image.convert('RGB').load()
            (args.output / 'images' / name).write_bytes(data)
            wnid = ids[truth[number - 1]]
            manifest.write(json.dumps({'id': name, 'path': 'images/' + name, 'wnid': wnid,
                           'label': index[wnid], 'sha256': hashlib.sha256(data).hexdigest()}) + '\n')
            seen.add(number)
    if len(seen) != 50000: raise ValueError('Incomplete ImageNet validation archive')
    (args.output / 'class_map.json').write_text(json.dumps([{'index': i, 'wnid': w} for i, w in enumerate(wnids)], indent=2))
    identity = {'dataset': 'ImageNet-1K', 'split': 'validation', 'images': 50000, 'classes': 1000,
                'devkit_sha256': sha256_file(devkit), 'image_archive_sha256': sha256_file(images),
                'class_order': 'sorted_leaf_WNIDs', 'all_images_decoded': True}
    finish(args.output, identity, ['validation.jsonl', 'class_map.json'])
    print(json.dumps({'status': 'valid', 'output': str(args.output)}))


if __name__ == '__main__': main()
