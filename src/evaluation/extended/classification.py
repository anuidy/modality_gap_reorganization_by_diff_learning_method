"""ImageNet zero-shot classification using the confirmed 80-template protocol."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from embeddings.artifact import sha256_file
from evaluation.diagnostic_runner import save_arrays
from evaluation.protocol import identity_hash
from evaluation.trajectory import write_json_atomic
from .common import normalized, verified, finish, source_identity


def class_prototypes(class_names, templates, encode_text):
    if not class_names or not templates:
        raise ValueError('Class names and templates must be nonempty')
    rows = []
    for name in class_names:
        vectors = normalized(encode_text([template.format(name) for template in templates]))
        if len(vectors) != len(templates): raise ValueError('Encoder dropped template rows')
        rows.append(normalized(vectors.mean(axis=0, dtype=np.float32)[None])[0])
    return np.stack(rows)


@torch.inference_mode()
def classify_vectors(images, prototypes, labels, device='cpu'):
    images = torch.as_tensor(normalized(images), device=device)
    prototypes = torch.as_tensor(normalized(prototypes), device=device)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(images),) or np.any(labels < 0) or np.any(labels >= len(prototypes)):
        raise ValueError('Invalid classification labels')
    scores = images @ prototypes.T
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(5, len(prototypes))].cpu().numpy()
    return {'top1': float(np.mean(order[:, 0] == labels)),
            'top5': float(np.mean(np.any(order == labels[:, None], axis=1)))}, order


def run_classification(root, encoder, prepared, template_path, output_root):
    spec = json.loads(template_path.read_text(encoding='utf-8'))
    if len(spec['class_names']) != 1000 or len(spec['templates']) != 80:
        raise ValueError('The confirmed ImageNet protocol requires 1000 classes and 80 templates')
    marker = verified(prepared)
    if not marker: raise ValueError('ImageNet validation data not prepared/verified')
    manifest = prepared / 'validation.jsonl'
    records = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if len(records) != 50000 or len({x['id'] for x in records}) != 50000:
        raise ValueError('Expected all 50000 unique ImageNet validation images')
    class_map = json.loads((prepared / 'class_map.json').read_text())
    if len(class_map) != 1000 or [x['index'] for x in class_map] != list(range(1000)):
        raise ValueError('Invalid ImageNet class map')
    if [x['wnid'] for x in class_map] != sorted(x['wnid'] for x in class_map):
        raise ValueError('Class order must follow sorted ImageNet leaf WNIDs')
    for row in records:
        if not 0 <= row['label'] < 1000 or row['wnid'] != class_map[row['label']]['wnid']:
            raise ValueError('ImageNet label/WordNet identity mismatch')
    identity = {'stage': 'B3', 'checkpoint': encoder.checkpoint, 'dataset': marker['identity'],
                'encoder_dependencies': encoder.dependencies,
                'template_sha256': sha256_file(template_path), 'data_manifest_sha256': sha256_file(manifest),
                'aggregation': 'normalize_each_template_then_mean_then_normalize',
                'precision': 'fp16_encoder_fp32_cosine', 'preprocess': 'gpu_nvjpeg_tensor_bicubic_v1',
                'runtime': encoder.runner.runtime_identity,
                'tie_policy': 'descending_score_then_class_index', 'sources': source_identity(root)}
    dest = output_root / 'B3' / identity_hash(identity)
    if verified(dest, identity): return dest
    dest.mkdir(parents=True, exist_ok=True); start = time.time()
    prototypes = class_prototypes(spec['class_names'], spec['templates'], encoder.texts)
    top5, labels, ids = [], [], []
    for offset in range(0, len(records), encoder.batch_size):
        batch = records[offset:offset + encoder.batch_size]; paths = []
        for row in batch:
            path = (prepared / row['path']).resolve()
            if not path.is_relative_to(prepared.resolve()) or sha256_file(path) != row['sha256']:
                raise ValueError('ImageNet image identity changed')
            paths.append(path)
        target = [x['label'] for x in batch]
        _, order = classify_vectors(encoder.images(paths), prototypes, target, encoder.device)
        top5.extend(order); labels.extend(target); ids.extend(x['id'] for x in batch)
    top5 = np.asarray(top5); labels = np.asarray(labels)
    metrics = {'Top1': float(np.mean(top5[:, 0] == labels)),
               'Top5': float(np.mean(np.any(top5 == labels[:, None], axis=1))), 'samples': len(labels)}
    save_arrays(dest / 'per_sample.npz', {'sample_ids': np.asarray(ids), 'labels': labels, 'top5': top5})
    save_arrays(dest / 'class_prototypes.npz', {'vectors': prototypes, 'class_names': np.asarray(spec['class_names'])})
    write_json_atomic(dest / 'summary.json', {'identity': identity, 'status': 'complete', 'metrics': metrics,
                      'seconds': time.time() - start, 'pipeline': encoder.encoder.stats})
    finish(dest, identity, ['summary.json', 'per_sample.npz', 'class_prototypes.npz'])
    return dest
