"""Shared artifact and bounded encoder utilities for the extended evaluation stages."""
import json
import importlib.metadata
import inspect
from pathlib import Path
import sys

import numpy as np

from embeddings.artifact import sha256_file
from evaluation.protocol import identity_hash
from evaluation.trajectory import write_json_atomic


def source_identity(root):
    paths = list((root / 'src/evaluation/extended').glob('*.py'))
    paths += list((root / 'src/model_adapters').glob('*.py'))
    paths += [root / 'src/evaluation' / p for p in
              ['trajectory.py', 'protocol.py', 'gpu_encoding.py', 'diagnostic_runner.py']]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def encoder_dependencies(root, model, adapter):
    """Bind caches to actual official code, tokenizer/backbone assets and library versions."""
    files = {}

    def tree(label, directory, suffixes):
        directory = Path(directory).resolve()
        if not directory.is_dir(): raise FileNotFoundError(directory)
        selected = [p for p in directory.rglob('*') if p.is_file() and p.suffix in suffixes
                    and not {'.git', '.cache', '__pycache__'} & set(p.relative_to(directory).parts)]
        if not selected: raise ValueError('Empty encoder dependency tree: ' + label)
        for path in sorted(selected): files[label + '/' + path.relative_to(directory).as_posix()] = sha256_file(path)
        return directory

    if model == 'clip':
        official = tree('openai_clip', Path(adapter._clip.__file__).resolve().parent, {'.py', '.json', '.gz'})
    elif model == 'beit3':
        official = tree('unilm/beit3', root / 'third_party/unilm/beit3', {'.py', '.json'})
        files['tokenizer/beit3.spm'] = sha256_file(adapter.sentencepiece_model)
    elif model == 'vista':
        official = tree('flag_embedding/visual_bge', root / 'third_party/flag_embedding/research/visual_bge', {'.py', '.json'})
        tree('text_backbone', adapter.text_backbone, {'.py', '.json', '.txt', '.spm', '.model', '.bin', '.safetensors'})
    else: raise ValueError('Unsupported extended-evaluation model: ' + model)
    actual = Path(inspect.getfile(type(adapter.model))).resolve()
    if not actual.is_relative_to(official):
        raise ValueError('Loaded model implementation is outside the recorded official source tree')
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'transformers', 'tokenizers',
                    'timm', 'torchscale', 'fairscale', 'einops', 'sentencepiece', 'safetensors', 'clip', 'ftfy', 'regex'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    return {'schema_version': 1, 'model': model, 'implementation': actual.relative_to(official).as_posix(),
            'files': files, 'libraries': versions, 'python': list(sys.version_info[:3])}


def normalized(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('Expected a finite embedding matrix')
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 0): raise ValueError('Zero/nonfinite embedding norm')
    return value / norm


def verified(directory, identity=None):
    marker = directory / 'complete.json'
    if not marker.exists(): return False
    data = json.loads(marker.read_text(encoding='utf-8'))
    if data['status'] != 'complete': raise ValueError('Invalid completion marker')
    if identity is not None and identity_hash(data['identity']) != identity_hash(identity):
        raise ValueError('Artifact identity mismatch')
    for filename, digest in data['files'].items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise ValueError('Artifact checksum/path mismatch: ' + filename)
    return data


def finish(directory, identity, names):
    write_json_atomic(directory / 'complete.json', {'status': 'complete', 'identity': identity,
                      'files': {name: sha256_file(directory / name) for name in names}})


class CheckpointEncoder:
    """Reuse published snapshots and current GPU preprocessing without retaining a full gallery in RAM."""
    def __init__(self, root, run_directory, point, device='cuda', batch_size=32, memory_gib=3):
        from evaluation.diagnostic_runner import DiagnosticRunner
        from evaluation.gpu_encoding import RawEncoder
        self.runner = DiagnosticRunner(root, root / 'configs/evaluation/formal_ab.yaml', run_directory,
                                       'b1', device, batch_size, memory_gib,
                                       requested_points=[point], allow_running=True)
        self.checkpoint = self.runner.checkpoint_identity(point)
        adapter = self.runner.model_at(point)
        self.dependencies = encoder_dependencies(root, self.runner.model, adapter)
        self.encoder = RawEncoder(adapter, self.runner.model, batch_size)
        self.batch_size = batch_size
        self.device = device

    def clear(self):
        for name in ('images', 'texts', 'joints', 'image_hashes'):
            getattr(self.encoder, name).clear()

    def images(self, paths):
        try: return self.encoder.image_rows(paths)
        finally: self.clear()

    def texts(self, texts):
        try: return self.encoder.text_rows(texts)
        finally: self.clear()

    def pairs(self, records, image_root):
        try: return self.encoder.fused([(x.get('text'), x.get('image')) for x in records], image_root)
        finally: self.clear()

    def close(self): self.encoder.close()
