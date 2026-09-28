"""Derive COCO4407 without modifying either frozen probe or split source."""
from pathlib import Path
import json

from datasets.probes import load_coco_manifest,load_probe_manifest
from embeddings.artifact import sha256_file
from .protocol import identity_hash
from .trajectory import write_json_atomic


def derive_coco_probe(parent:Path,karpathy:Path,destination:Path,root:Path,
                      expected_removed=593,expected_remaining=4407):
    original=json.loads(parent.read_text(encoding='utf-8'))
    split=json.loads(karpathy.read_text(encoding='utf-8'))
    test_ids={int(row['cocoid']) for row in split['images'] if row['split']=='test'}
    if len(test_ids)!=5000:raise ValueError('Karpathy test must contain 5000 distinct COCO IDs')
    samples=original['samples'];ids=[int(row['image_id']) for row in samples]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate original COCO probe IDs')
    removed=[i for i in ids if i in test_ids]
    kept=[row for row in samples if int(row['image_id']) not in test_ids]
    if len(removed)!=expected_removed or len(kept)!=expected_remaining:
        raise ValueError(f'COCO split conflict: removed={len(removed)}, kept={len(kept)}; do not change the protocol silently')
    payload={'schema_version':1,'probe_name':'coco_2017_val_4407','sample_count':len(kept),
             'parent_sha256':sha256_file(parent),'karpathy_sha256':sha256_file(karpathy),
             'excluded_image_ids':removed,'karpathy_test_ids_sha256':identity_hash(sorted(test_ids)),
             'selection':'original_order_excluding_karpathy_test_image_ids','samples':kept}
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8'))!=payload:raise FileExistsError('Derived probe differs from existing file')
    else:write_json_atomic(destination,payload)
    return load_probe_manifest(destination,root)
