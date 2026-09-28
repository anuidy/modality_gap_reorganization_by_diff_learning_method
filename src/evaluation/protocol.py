"""Explicit identities and settings for the bounded A/B diagnostic workflow."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import hashlib
import json
import math
from pathlib import Path

import yaml

MODALITIES = ('I', 'T', 'IT')
MMEB_TASKS = ('VisDial', 'CIRR', 'VisualNews_t2i', 'VisualNews_i2t', 'MSCOCO_t2i',
              'MSCOCO_i2t', 'NIGHTS', 'WebQA', 'OVEN', 'FashionIQ', 'EDIS', 'Wiki-SS-NQ')
MMEB_GROUPS = {
    'cross_modal': ('VisDial','VisualNews_t2i','VisualNews_i2t','MSCOCO_t2i','MSCOCO_i2t','Wiki-SS-NQ'),
    'mixed_composite': ('CIRR','WebQA','OVEN','FashionIQ','EDIS'),
    'image_image': ('NIGHTS',),
}


def identity_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextlib.contextmanager
def evaluation_worker_lock(output_root: Path):
    """Only one A/B worker writes the shared baseline/cache tree at a time."""
    import os
    output_root.mkdir(parents=True,exist_ok=True)
    with (output_root/'.evaluation.lock').open('a+b') as handle:
        if os.name=='nt':
            import msvcrt
            handle.seek(0,2)
            if handle.tell()==0:handle.write(b'0');handle.flush()
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


@dataclass(frozen=True)
class ASettings:
    fixed_temperature: float
    top_ks: tuple[int, ...] = (1, 5, 10)
    norm_ddof: int = 1
    negative_plot_samples: int = 100_000
    histogram_bins: int = 100
    seed: int = 20_260_825
    geometry_pair_count: int = 1_000_000
    query_block: int = 64

    def __post_init__(self):
        if not isinstance(self.fixed_temperature, (float,int)) or not math.isfinite(self.fixed_temperature) or self.fixed_temperature <= 0:
            raise ValueError('A6 fixed_temperature must be explicitly confirmed and positive')
        if not self.top_ks or any(type(k) is not int or k <= 0 for k in self.top_ks) or len(set(self.top_ks)) != len(self.top_ks):
            raise ValueError('top_ks must be distinct positive integers')
        if self.norm_ddof != 1:
            raise ValueError('The confirmed norm protocol uses sample standard deviation (ddof=1)')
        if min(self.negative_plot_samples,self.histogram_bins,self.geometry_pair_count,self.query_block) <= 0:
            raise ValueError('Diagnostic counts and block sizes must be positive')

    def identity(self):
        # Block size is a computational setting; all queries/candidates remain present.
        value = asdict(self)
        value.pop('query_block')
        return {**value, 'schema':'A0-A6/v1', 'pool':'I_T_IT_self_only_masked',
                'background':'exclude_all_same_instance', 'rank_ties':'strictly_greater_plus_one',
                'topk_ties':'fractional_boundary_membership', 'C':'deferred', 'Global':'deferred'}


def load_protocol(path: Path, stage: str):
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or stage not in ('a','b','b1'):
        raise ValueError('Unsupported diagnostic protocol/stage')
    if stage == 'a':
        if value['a'].get('status') != 'confirmed':
            raise ValueError('A settings are awaiting confirmation')
        raw = {k:v for k,v in value['a'].items() if k != 'status'}
        raw['top_ks'] = tuple(raw['top_ks'])
        return value, ASettings(**raw)
    if stage == 'b1':
        if value['b1']['caption_policy'] != 'all_canonical_raw_sentences':
            raise ValueError('B1 requires the confirmed complete original captions')
        if set(value['b1']['datasets']) != {'coco', 'flickr30k'}:
            raise ValueError('B1 requires both confirmed datasets')
        return value, None
    if tuple(value['b']['tasks']) != MMEB_TASKS:
        raise ValueError('The current B diagnostic comprises the frozen 12 MMEB retrieval tasks')
    return value, None
