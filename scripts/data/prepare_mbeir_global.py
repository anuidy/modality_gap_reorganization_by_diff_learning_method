"""Prepare the official M-BEIR Global inputs; no downloads or protocol reduction."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.extended.mbeir_data import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True, help='JSON mapping every selected query/qrels/gallery relative path to SHA-256')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/evaluation/mbeir_global')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    lock = json.loads(args.source_lock.read_text())
    if not lock or any(len(v) != 64 for v in lock.values()): raise ValueError('An explicit source lock is required')
    print(json.dumps({'input': str(args.data_root), 'output': str(args.output), 'files': len(lock), 'execute': args.execute}))
    if args.execute: print(json.dumps(build_index(args.data_root.resolve(), args.output.resolve(), lock), indent=2))


if __name__ == '__main__': main()
