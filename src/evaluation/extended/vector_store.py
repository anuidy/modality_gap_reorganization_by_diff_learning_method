"""Resumable normalized float32 shards; never concatenate a full Global gallery."""
import bisect
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from embeddings.artifact import sha256_file
from evaluation.protocol import identity_hash
from evaluation.trajectory import write_json_atomic
from .common import normalized


def encode_shards(records, total, encode, directory, identity, batch_size=32, shard_rows=8192):
    if min(total, batch_size, shard_rows) < 1: raise ValueError('Invalid vector-store dimensions')
    directory.mkdir(parents=True, exist_ok=True); path = directory / 'index.json'
    state = json.loads(path.read_text()) if path.exists() else {
        'identity': identity, 'total_rows': total, 'dtype': 'float32', 'l2_normalized': True,
        'status': 'partial', 'shard_rows': shard_rows, 'shards': []}
    if state['identity'] != identity or state['total_rows'] != total or state['shard_rows'] != shard_rows:
        raise ValueError('Partial vector cache belongs to different inputs/settings')
    if state['dtype'] != 'float32' or not state['l2_normalized']: raise ValueError('Unexpected cached representation')
    iterator = iter(records); offset = 0
    for shard_index in range((total + shard_rows - 1) // shard_rows):
        chunk = list(itertools.islice(iterator, min(shard_rows, total - offset)))
        if len(chunk) != min(shard_rows, total - offset): raise ValueError('Input iterator ended early')
        input_digest = identity_hash(chunk)
        if shard_index < len(state['shards']):
            old = state['shards'][shard_index]
            if old['input_sha256'] != input_digest or old['start'] != offset or old['rows'] != len(chunk):
                raise ValueError('Resumed input order changed')
            if sha256_file(directory / old['file']) != old['sha256']: raise ValueError('Vector shard corrupted')
        else:
            blocks = []
            for start in range(0, len(chunk), batch_size):
                batch = chunk[start:start + batch_size]; vectors = normalized(encode(batch))
                if len(vectors) != len(batch): raise ValueError('Encoder dropped rows')
                blocks.append(vectors)
            value = np.concatenate(blocks)
            if 'dimension' in state and state['dimension'] != value.shape[1]: raise ValueError('Embedding dimension changed')
            state['dimension'] = value.shape[1]
            required = (total - offset) * state['dimension'] * 4 + 2 * 2**30
            if shutil.disk_usage(directory).free < required: raise RuntimeError('Insufficient space for remaining Global vector shards')
            name = f'shard_{shard_index:06d}.npy'; temp = directory / (name + '.partial')
            with temp.open('wb') as handle: np.save(handle, value, allow_pickle=False)
            temp.replace(directory / name)
            state['shards'].append({'file': name, 'start': offset, 'rows': len(value),
                                    'sha256': sha256_file(directory / name), 'input_sha256': input_digest})
            write_json_atomic(path, state)
        offset += len(chunk)
    if next(iterator, None) is not None: raise ValueError('Input iterator contains unexpected additional rows')
    state['status'] = 'complete'; write_json_atomic(path, state)
    return ShardedVectors(directory)


class ShardedVectors:
    def __init__(self, directory):
        self.directory = Path(directory); path = self.directory / 'index.json'
        self.index = json.loads(path.read_text()); self.index_sha256 = sha256_file(path)
        if self.index['status'] != 'complete': raise ValueError('Incomplete Global vector cache')
        self.maps = []; self.starts = []; offset = 0
        for row in self.index['shards']:
            path = (self.directory / row['file']).resolve()
            if not path.is_relative_to(self.directory.resolve()) or sha256_file(path) != row['sha256']:
                raise ValueError('Global vector shard identity mismatch')
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if value.dtype != np.float32 or value.shape != (row['rows'], self.index['dimension']) or row['start'] != offset:
                raise ValueError('Invalid vector shard shape/order')
            self.maps.append(value); self.starts.append(offset); offset += len(value)
        if offset != self.index['total_rows']: raise ValueError('Incomplete vector row coverage')
        self.shape = (offset, self.index['dimension'])

    def __len__(self): return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if not 0 <= key < len(self): raise IndexError(key)
            index = bisect.bisect_right(self.starts, key) - 1
            return self.maps[index][key - self.starts[index]]
        if isinstance(key, slice):
            start, end, stride = key.indices(len(self))
            if stride != 1: return self[np.arange(start, end, stride)]
            if start >= end: return np.empty((0, self.shape[1]), np.float32)
            chunks = []
            while start < end:
                index = bisect.bisect_right(self.starts, start) - 1
                stop = min(end, self.starts[index] + len(self.maps[index]))
                chunks.append(self.maps[index][start - self.starts[index]:stop - self.starts[index]])
                start = stop
            return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        indices = np.asarray(key)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu': raise IndexError('Expected one-dimensional integer indices')
        return np.stack([self[int(i)] for i in indices]) if len(indices) else np.empty((0, self.shape[1]), np.float32)
