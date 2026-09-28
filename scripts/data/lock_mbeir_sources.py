"""Fetch only official Hugging Face file identities; do not download the Global dataset."""
import argparse
import json
from pathlib import Path
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response: return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', required=True, help='Explicit dataset commit SHA, not main')
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Pin a 40-character dataset commit SHA')
    if args.output.exists(): raise FileExistsError(args.output)
    repo = 'TIGER-Lab/M-BEIR'
    rows = fetch(f'https://huggingface.co/api/datasets/{repo}/tree/{args.revision}?recursive=true&limit=1000')
    selected = [x for x in rows if x['type'] == 'file' and (
        x['path'] == 'cand_pool/global/mbeir_union_test_cand_pool.jsonl' or
        x['path'].startswith(('query/test/', 'query/val/', 'qrels/test/', 'qrels/val/')))]
    if len(selected) != 65: raise ValueError('Expected the gallery and 16 task query/qrels files per split')
    lock = {}
    for row in selected:
        if 'lfs' not in row: raise ValueError('Expected large-file SHA-256 metadata: ' + row['path'])
        lock[row['path']] = row['lfs']['oid']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + '\n')
    args.output.with_suffix('.provenance.json').write_text(json.dumps({'repository': repo, 'revision': args.revision,
        'source_api': 'Hugging Face tree/LFS metadata', 'files': {x['path']: x['size'] for x in selected}}, indent=2) + '\n')
    print(json.dumps({'status': 'source_lock_created', 'files': len(lock)}))


if __name__ == '__main__': main()
