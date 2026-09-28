"""Full-test bidirectional retrieval with explicit multi-positive relevance."""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from embeddings.artifact import sha256_file
from .protocol import identity_hash


@dataclass
class CaptionRetrievalTask:
    name: str
    images: list[Path]
    texts: list[str]
    image_ids: np.ndarray
    caption_ids: np.ndarray
    caption_image_indices: np.ndarray
    identity: dict


def load_caption_task(root, name, spec):
    directory = (root / spec['directory']).resolve()
    annotation = directory / 'annotations/karpathy_test.json'
    report = json.loads((directory / 'validation_report.json').read_text(encoding='utf-8'))
    checksum = directory / 'image_checksums.jsonl'
    if report['status'] != 'valid' or sha256_file(annotation) != report['test_annotation_sha256']:
        raise ValueError('B1 annotation identity mismatch')
    if sha256_file(checksum) != report['image_checksums_sha256']:
        raise ValueError('B1 image inventory identity mismatch')
    payload = json.loads(annotation.read_text(encoding='utf-8')); rows = payload['images']
    if payload['source']['annotation_member_sha256'] != spec['annotation_member_sha256']:
        raise ValueError('Unexpected canonical Karpathy annotation')
    inventory = [json.loads(s) for s in checksum.read_text().splitlines() if s.strip()]
    expected = {row['filename']: row['sha256'] for row in inventory}
    if len(expected) != len(inventory) or set(expected) != {row['filename'] for row in rows}:
        raise ValueError('B1 image inventory differs from selected test images')
    images, texts, image_ids, caption_ids, owners = [], [], [], [], []
    for i, row in enumerate(rows):
        filename = row['filename']
        if row['split'] != 'test' or Path(filename).name != filename:
            raise ValueError('Invalid B1 test image')
        image = (directory / 'images' / filename).resolve()
        if not image.is_relative_to((directory / 'images').resolve()) or sha256_file(image) != expected[filename]:
            raise ValueError('B1 image content/path changed: ' + filename)
        images.append(image); image_ids.append(str(row['imgid']))
        if not row['sentences']: raise ValueError('Image has no relevant captions')
        for sentence in row['sentences']:
            if str(sentence['imgid']) != str(row['imgid']) or not sentence['raw'].strip():
                raise ValueError('Invalid caption relevance')
            texts.append(sentence['raw']); caption_ids.append(str(sentence['sentid'])); owners.append(i)
    if len(images) != spec['images'] or len(texts) != spec['captions']:
        raise ValueError('B1 test split counts differ from the protocol')
    if len(set(image_ids)) != len(image_ids) or len(set(caption_ids)) != len(caption_ids):
        raise ValueError('Duplicate B1 image/caption IDs')
    identity = {'dataset': name, 'split': 'Karpathy_test', 'images': len(images), 'captions': len(texts),
                'annotation_sha256': sha256_file(annotation), 'image_inventory_sha256': sha256_file(checksum),
                'source': payload['source'], 'caption_policy': 'all_canonical_raw_sentences_in_original_order',
                'ordered_inputs_sha256': identity_hash({'images': image_ids, 'captions': caption_ids,
                                                        'texts': texts, 'caption_image_indices': owners})}
    return CaptionRetrievalTask(name, images, texts, np.asarray(image_ids), np.asarray(caption_ids),
                                np.asarray(owners, np.int64), identity)


def _normalized(value, device):
    value = torch.as_tensor(value, dtype=torch.float32, device=device)
    if value.ndim != 2 or not torch.isfinite(value).all(): raise ValueError('Invalid retrieval embedding')
    norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    if torch.any(norms <= 0): raise ValueError('Zero-norm retrieval embedding')
    return value / norms


def _direction_ranks(queries, candidates, relevant, query_block):
    ranks, ties, best_targets = [], [], []
    order = torch.arange(len(candidates), device=candidates.device)
    for start in range(0, len(queries), query_block):
        scores = queries[start:start + query_block] @ candidates.T
        for offset, row in enumerate(scores):
            positive = torch.as_tensor(relevant[start + offset], device=row.device, dtype=torch.long)
            best_score = row[positive].max()
            best_index = positive[row[positive] == best_score].min()
            rank = 1 + ((row > best_score) | ((row == best_score) & (order < best_index))).sum()
            ranks.append(int(rank)); ties.append(int((row == best_score).sum())); best_targets.append(int(best_index))
    return np.asarray(ranks, np.int32), np.asarray(ties, np.int32), np.asarray(best_targets, np.int32)


@torch.inference_mode()
def bidirectional_retrieval(image_vectors, text_vectors, caption_image_indices, query_block=64, device='cpu'):
    if query_block < 1: raise ValueError('query_block must be positive')
    images, texts = _normalized(image_vectors, device), _normalized(text_vectors, device)
    owners = np.asarray(caption_image_indices, dtype=np.int64)
    if images.shape[1] != texts.shape[1] or owners.shape != (len(texts),) or not len(images):
        raise ValueError('Invalid image/text/relevance dimensions')
    if np.any(owners < 0) or np.any(owners >= len(images)): raise ValueError('Caption owner outside image pool')
    i2t = [np.flatnonzero(owners == i) for i in range(len(images))]
    if any(not len(indices) for indices in i2t): raise ValueError('Image has no relevant captions')
    ir, ities, ibest = _direction_ranks(images, texts, i2t, query_block)
    tr, tties, tbest = _direction_ranks(texts, images, [[i] for i in owners], query_block)
    def metrics(ranks):
        return {**{f'R@{k}': float(np.mean(ranks <= k)) for k in (1, 5, 10)},
                'median_rank': float(np.median(ranks)), 'queries': len(ranks)}
    result = {'I->T': metrics(ir), 'T->I': metrics(tr), 'image_candidates': len(images),
              'text_candidates': len(texts), 'tie_policy': 'descending_cosine_then_canonical_candidate_index',
              'relevance': 'all_captions_of_image; caption_to_its_image'}
    result['mR'] = float(np.mean([result[d][f'R@{k}'] for d in ('I->T', 'T->I') for k in (1, 5, 10)]))
    return result, {'i2t_ranks': ir, 't2i_ranks': tr, 'i2t_ties': ities, 't2i_ties': tties,
                    'i2t_best_relevant_caption': ibest, 't2i_relevant_image': tbest, 'caption_image_indices': owners}


def run_b1(runner, points, tasks):
    from .diagnostic_runner import complete_artifact, finish_artifact, save_arrays
    from .trajectory import write_json_atomic
    from .gpu_encoding import RawEncoder
    outputs = []
    for name in tasks:
        task = load_caption_task(runner.root, name, runner.config['b1']['datasets'][name])
        for point in points:
            identity = {'stage': 'B1', 'checkpoint': runner.checkpoint_identity(point), 'dataset': task.identity,
                        'preprocess': 'gpu_nvjpeg_tensor_bicubic_v1', 'precision': 'fp16_encoder_fp32_cosine',
                        'tie_policy': 'descending_cosine_then_canonical_candidate_index',
                        'mR': 'mean_of_six_directional_R_at_1_5_10', 'code_sha256': runner.code_sha,
                        'runtime': runner.runtime_identity}
            directory = runner.output / 'B1' / runner.model / point / name / identity_hash(identity)
            if not complete_artifact(directory, identity):
                started = time.time(); directory.mkdir(parents=True, exist_ok=True)
                runner.model_at(point); encoder = RawEncoder(runner.adapter, runner.model, runner.batch_size)
                try:
                    images = encoder.image_rows(task.images); texts = encoder.text_rows(task.texts)
                    result, arrays = bidirectional_retrieval(images, texts, task.caption_image_indices,
                                                            runner.config['b1']['query_block'], str(runner.device))
                    save_arrays(directory / 'per_query.npz', {'image_ids': task.image_ids,
                                'caption_ids': task.caption_ids, **arrays})
                    write_json_atomic(directory / 'summary.json', {'identity': identity, 'status': 'complete',
                                      'metrics': result, 'seconds': time.time() - started, 'pipeline': encoder.stats})
                    finish_artifact(directory, identity, ['per_query.npz', 'summary.json'])
                finally: encoder.close()
            outputs.append(str(directory))
            print(json.dumps({'B1_complete': point, 'task': name, 'directory': str(directory)}), flush=True)
    return outputs
