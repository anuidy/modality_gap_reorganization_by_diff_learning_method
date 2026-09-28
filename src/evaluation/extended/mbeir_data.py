"""Stream official M-BEIR records into a disk-backed, content-verified evaluation index."""
import hashlib
import json
from pathlib import Path
import sqlite3

from embeddings.artifact import sha256_file
from evaluation.protocol import identity_hash
from .common import finish, verified

MODES = {'image': 0, 'text': 1, 'image,text': 2, 'text,image': 2}


def record_fields(row, query=False):
    text = row.get('query_txt' if query else 'txt') or None
    image = row.get('query_img_path' if query else 'img_path') or None
    mode = row.get('query_modality' if query else 'modality')
    if mode not in MODES: raise ValueError('Unknown M-BEIR modality: ' + str(mode))
    expected = (bool(image), bool(text))
    if expected != {0: (True, False), 1: (False, True), 2: (True, True)}[MODES[mode]]:
        raise ValueError('M-BEIR modality/content mismatch')
    return text, image, MODES[mode]


def positive_ids(row):
    values = row.get('pos_cand_list', [])
    ids = [str(v['did'] if isinstance(v, dict) else v) for v in values]
    if not ids or len(ids) != len(set(ids)): raise ValueError('Missing/duplicate positive candidate IDs')
    return ids


def iter_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def build_index(data_root, destination, expected_source_hashes, *, minimum_candidates=5_000_000):
    pool = data_root / 'cand_pool/global/mbeir_union_test_cand_pool.jsonl'
    query_files = {split: sorted((data_root / 'query' / split).glob('*.jsonl')) for split in ('val', 'test')}
    if not all(query_files.values()) or not pool.is_file(): raise FileNotFoundError('Full test gallery and val/test queries are required')
    inputs = [pool] + [p for files in query_files.values() for p in files]
    qrels = {}
    for split, files in query_files.items():
        for path in files:
            qrel = data_root / 'qrels' / split / (path.stem + '_qrels.txt')
            if not qrel.is_file(): raise FileNotFoundError(qrel)
            qrels[path] = qrel; inputs.append(qrel)
    hashes = {p.relative_to(data_root).as_posix(): sha256_file(p) for p in inputs}
    if not expected_source_hashes or any(expected_source_hashes.get(name) != value for name, value in hashes.items()):
        raise ValueError('Source files must match an explicitly pinned source-hash manifest')
    if set(expected_source_hashes) != set(hashes): raise ValueError('Pinned source inventory differs from the selected full protocol')
    if destination.exists(): raise FileExistsError('Refusing to overwrite prepared M-BEIR data')
    destination.mkdir(parents=True)
    connection = sqlite3.connect(destination / 'records.sqlite')
    connection.executescript('''
        CREATE TABLE candidates (row_id INTEGER PRIMARY KEY, did TEXT UNIQUE NOT NULL, modality INTEGER NOT NULL, text TEXT, image TEXT);
        CREATE TABLE images (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
        CREATE TABLE queries (row_id INTEGER PRIMARY KEY, split TEXT NOT NULL, task TEXT NOT NULL, qid TEXT NOT NULL,
          modality INTEGER NOT NULL, text TEXT, image TEXT, positive_rows TEXT NOT NULL, input_sha256 TEXT NOT NULL,
          positives_outside_gallery TEXT NOT NULL,
          UNIQUE(split,task,qid));
        CREATE INDEX query_split ON queries(split,row_id);
    ''')
    try:
        def image_identity(relative):
            if relative is None: return None
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root.resolve()): raise ValueError('M-BEIR image escapes data directory')
            found = connection.execute('SELECT sha256 FROM images WHERE path=?', (relative,)).fetchone()
            if found: return found[0]
            digest = sha256_file(path)
            connection.execute('INSERT INTO images VALUES (?,?)', (relative, digest))
            return digest
        count = 0
        for row_id, row in enumerate(iter_jsonl(pool)):
            text, image, mode = record_fields(row); image_identity(image)
            connection.execute('INSERT INTO candidates VALUES (?,?,?,?,?)', (row_id, str(row['did']), mode, text, image))
            count += 1
            if count % 10000 == 0: connection.commit()
        if count < minimum_candidates: raise ValueError('Candidate pool is too small for full Global evaluation')
        query_counts = {}; query_row = 0
        for split, files in query_files.items():
            for path in files:
                task = path.stem.removesuffix('_' + split)
                label_map = {}
                for line in qrels[path].read_text().splitlines():
                    fields = line.split()
                    if len(fields) != 5: raise ValueError('Expected official five-column M-BEIR qrels')
                    qid, _, did, relevance, _ = fields
                    if int(relevance) > 0: label_map.setdefault(qid, set()).add(did)
                seen = set(); n = 0
                for row in iter_jsonl(path):
                    text, image, mode = record_fields(row, query=True); im_hash = image_identity(image)
                    qid = str(row['qid']); positives = positive_ids(row)
                    if set(positives) != label_map.get(qid): raise ValueError('Query relevance differs from official qrels')
                    indices = []; absent = []
                    for did in positives:
                        match = connection.execute('SELECT row_id FROM candidates WHERE did=?', (did,)).fetchone()
                        if not match:
                            if split == 'test': raise ValueError('Relevant candidate absent from full test gallery: ' + did)
                            absent.append(did); continue
                        indices.append(match[0])
                    input_sha = identity_hash({'text': text, 'image_sha256': im_hash, 'modality': mode})
                    connection.execute('INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?,?)',
                                       (query_row, split, task, qid, mode, text, image, json.dumps(sorted(indices)), input_sha, json.dumps(absent)))
                    query_row += 1; n += 1; seen.add(qid)
                if seen != set(label_map): raise ValueError('Query/qrels coverage mismatch')
                query_counts[split + '/' + task] = n
        # Calibration queries must be held out by both ID and input content.
        overlap = connection.execute("SELECT COUNT(*) FROM queries v JOIN queries t ON v.task=t.task AND v.qid=t.qid WHERE v.split='val' AND t.split='test'").fetchone()[0]
        connection.execute('CREATE INDEX query_input ON queries(input_sha256)')
        duplicate_content = connection.execute("SELECT COUNT(*) FROM queries v WHERE v.split='val' AND EXISTS (SELECT 1 FROM queries t WHERE t.split='test' AND t.input_sha256=v.input_sha256)").fetchone()[0]
        if overlap or duplicate_content: raise ValueError(f'Calibration/test overlap: IDs={overlap}, identical_inputs={duplicate_content}')
        connection.commit()
        image_count = connection.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    finally: connection.close()
    identity = {'dataset': 'M-BEIR', 'gallery': 'official_global_test', 'candidate_count': count,
                'query_counts': query_counts, 'image_count': image_count, 'sources': hashes,
                'image_root': str(data_root.resolve()), 'text_policy': 'native_text_no_added_instruction',
                'calibration_split': 'official_val; disjoint_IDs_and_inputs', 'sqlite_version': sqlite3.sqlite_version}
    finish(destination, identity, ['records.sqlite'])
    return identity


class MBeirIndex:
    def __init__(self, directory):
        self.directory = directory; self.marker = verified(directory)
        if not self.marker: raise ValueError('M-BEIR data preparation is incomplete')
        self.identity = self.marker['identity']; self.image_root = Path(self.identity['image_root'])
        self.db = sqlite3.connect((directory / 'records.sqlite').resolve().as_uri() + '?mode=ro', uri=True)

    def records(self, split='candidates'):
        if split == 'candidates':
            rows = self.db.execute('SELECT row_id,did,modality,text,image FROM candidates ORDER BY row_id')
            for row, did, mode, text, image in rows:
                yield {'row': row, 'id': did, 'modality': mode, 'text': text, 'image': image}
        else:
            if split not in ('val', 'test'): raise ValueError('Unknown query split')
            rows = self.db.execute('SELECT row_id,task,qid,modality,text,image,positive_rows,input_sha256,positives_outside_gallery FROM queries WHERE split=? ORDER BY row_id', (split,))
            for _, task, qid, mode, text, image, positives, input_sha, absent in rows:
                yield {'id': task + '/' + qid, 'task': task, 'qid': qid, 'modality': mode, 'text': text,
                       'image': image, 'positives': json.loads(positives), 'input_sha256': input_sha,
                       'positives_outside_gallery': json.loads(absent)}

    def count(self, split):
        if split == 'candidates': return self.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        return self.db.execute('SELECT COUNT(*) FROM queries WHERE split=?', (split,)).fetchone()[0]

    def validate_images(self, records):
        for relative in {row['image'] for row in records if row.get('image')}:
            expected = self.db.execute('SELECT sha256 FROM images WHERE path=?', (relative,)).fetchone()
            if not expected or sha256_file(self.image_root / relative) != expected[0]:
                raise ValueError('M-BEIR image changed since preparation: ' + relative)

    def close(self): self.db.close()
