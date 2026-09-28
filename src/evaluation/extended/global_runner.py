"""Encode, calibrate and evaluate a complete prepared M-BEIR Global view."""
import json
from pathlib import Path
import time

import numpy as np

from embeddings.artifact import sha256_file
from evaluation.diagnostic_runner import save_arrays
from evaluation.protocol import identity_hash
from evaluation.trajectory import write_json_atomic
from .common import source_identity, verified, finish
from .mbeir_data import MBeirIndex
from .vector_store import encode_shards
from .global_retrieval import fit_calibration, exact_global_ranks, summarize_global


def run_global(root, encoder, prepared, output_root, settings):
    required = {'calibration': 'task_by_candidate_modality_negative_mean',
                'calibration_split': 'official_val_disjoint_from_test',
                'oracle': 'union_of_all_relevant_answer_modalities',
                'text_policy': 'native_text_no_added_instruction'}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError('Global settings differ from the confirmed protocol')
    data = MBeirIndex(prepared); started = time.time()
    try:
        if data.count('candidates') < settings['minimum_candidates']:
            raise ValueError('Prepared gallery does not satisfy full Global scope')
        sources = source_identity(root)
        base = {'checkpoint': encoder.checkpoint, 'dataset': data.identity,
                'encoder_dependencies': encoder.dependencies,
                'prepared_index_sha256': data.marker['files']['records.sqlite'],
                'sources': sources, 'preprocess': 'gpu_nvjpeg_tensor_bicubic_v1',
                'runtime': encoder.runner.runtime_identity,
                'precision': 'fp16_encoder_float32_vectors_and_cosine', 'fusion': 'vista_native_else_raw_sum_before_l2'}
        caches = {}
        for split in ('candidates', 'val', 'test'):
            identity = {**base, 'role': split, 'stored_representation': 'L2_normalized_float32'}
            directory = output_root / 'Global_vectors' / identity_hash(identity)
            def encode(records):
                data.validate_images(records)
                return encoder.pairs(records, data.image_root)
            caches[split] = encode_shards(data.records(split), data.count(split), encode, directory, identity,
                                         encoder.batch_size, settings['shard_rows'])
        modalities = np.fromiter((x[0] for x in data.db.execute('SELECT modality FROM candidates ORDER BY row_id')), np.int8)
        validation = list(data.records('val')); test = list(data.records('test'))
        calibration = fit_calibration(caches['val'][:], caches['candidates'], modalities,
                                      [x['positives'] for x in validation], [x['task'] for x in validation],
                                      [x['id'] for x in validation], [x['id'] for x in test], settings['candidate_block'], True)
        cal_identity = {**base, 'validation_query_ids_sha256': identity_hash([x['id'] for x in validation]),
                        'method': calibration['method'], 'weighting': calibration['weighting']}
        calibration_id = identity_hash(cal_identity)
        identity = {**base, 'stage': 'B2_Global', 'settings': settings, 'calibration_id': calibration_id,
                    'vector_indices': {k: v.index_sha256 for k, v in caches.items()},
                    'oracle': 'union_of_modalities_of_all_relevant_candidates',
                    'views': ['raw', 'calibrated', 'oracle'],
                    'raw_definition': 'uncalibrated_L2_cosine_not_raw_dot_product'}
        destination = output_root / 'Global' / identity_hash(identity)
        if verified(destination, identity): return destination
        destination.mkdir(parents=True, exist_ok=True)
        rows = exact_global_ranks(caches['test'], caches['candidates'], modalities,
                                 [x['positives'] for x in test], [x['task'] for x in test], calibration, True,
                                 settings['query_block'], settings['candidate_block'], encoder.device, True)
        summary = summarize_global(rows, [x['task'] for x in test])
        arrays = {'query_ids': np.asarray([x['id'] for x in test]), 'tasks': np.asarray([x['task'] for x in test])}
        for view, values in rows.items(): arrays.update({view + '/' + k: v for k, v in values.items()})
        save_arrays(destination / 'per_query.npz', arrays)
        write_json_atomic(destination / 'calibration.json', {'identity': cal_identity, 'calibration_id': calibration_id, **calibration})
        write_json_atomic(destination / 'summary.json', {'identity': identity, 'status': 'complete',
                          'metrics': summary, 'seconds': time.time() - started, 'pipeline': encoder.encoder.stats,
                          'causal_scope': 'oracle_uses_answer_modality; neither_oracle_gain_nor_correlation_proves_gap_causality'})
        finish(destination, identity, ['summary.json', 'per_query.npz', 'calibration.json'])
        return destination
    finally: data.close()
