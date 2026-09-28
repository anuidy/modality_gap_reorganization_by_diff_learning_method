"""Stable, named random streams derived from one experiment seed.

Stateless step/epoch addressing makes continuation independent of prior
validation calls, other branches and the physical GPU running the job.
"""

from __future__ import annotations

import hashlib
import json


def stream_seed(master_seed: int, purpose: str, *position: int) -> int:
    if master_seed < 0 or any(value < 0 for value in position):
        raise ValueError("Random stream seeds and positions must be non-negative.")
    payload = json.dumps(["formal-v1", master_seed, purpose, *position], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
