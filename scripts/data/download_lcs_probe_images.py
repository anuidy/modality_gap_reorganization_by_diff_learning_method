"""Create a fixed LCS probe split and fetch only its image ZIP entries by HTTP range.

The upstream LLaVA repository distributes every image in one 25 GiB ZIP.  This
script reads the ZIP central directory remotely, then retrieves and validates
only the compressed entries selected for the in-domain probe.  It never
downloads the complete archive.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import random
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


ZIP_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/"
    "images.zip?download=true"
)
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def request_range(url: str, start: int, end: int) -> bytes:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        THREAD_LOCAL.session = session
    expected_size = end - start + 1
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "modality-gap-probe/1.0"},
                timeout=90,
            )
            status = response.status_code
            payload = response.content
            if status == 206 and len(payload) == expected_size:
                return payload
            last_error = RuntimeError(
                f"Range request {start}-{end} returned HTTP {status} with "
                f"{len(payload)} bytes; expected {expected_size}."
            )
            if status not in {429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0**attempt)
        except requests.RequestException as error:
            last_error = error
            delay = min(30.0, 2.0**attempt)
        if attempt < 5:
            time.sleep(delay)
    raise RuntimeError(str(last_error))


def remote_zip_directory(url: str, archive_size: int) -> tuple[int, bytes]:
    tail_size = min(1_048_576, archive_size)
    tail_start = archive_size - tail_size
    tail = request_range(url, tail_start, archive_size - 1)
    eocd_at = tail.rfind(EOCD_SIGNATURE)
    if eocd_at < 0:
        raise RuntimeError("Could not locate the ZIP end-of-central-directory record.")

    _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_at
    )
    if directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        locator_at = eocd_at - 20
        if tail[locator_at : locator_at + 4] != ZIP64_LOCATOR_SIGNATURE:
            raise RuntimeError("ZIP64 locator missing from a ZIP64 archive.")
        _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sLQL", tail, locator_at)
        zip64_eocd = request_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        fields = struct.unpack("<4sQHHLLQQQQ", zip64_eocd)
        if fields[0] != ZIP64_EOCD_SIGNATURE:
            raise RuntimeError("ZIP64 end-of-central-directory record is invalid.")
        directory_size, directory_offset = fields[8], fields[9]

    directory = request_range(url, directory_offset, directory_offset + directory_size - 1)
    return directory_offset, directory


def read_zip64_extra(
    extra: bytes,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
    disk_start: int,
) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, cursor)
        data = extra[cursor + 4 : cursor + 4 + data_size]
        cursor += 4 + data_size
        if header_id != 0x0001:
            continue
        value_at = 0
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = struct.unpack_from("<Q", data, value_at)[0]
            value_at += 8
        if compressed_size == 0xFFFFFFFF:
            compressed_size = struct.unpack_from("<Q", data, value_at)[0]
            value_at += 8
        if local_offset == 0xFFFFFFFF:
            local_offset = struct.unpack_from("<Q", data, value_at)[0]
            value_at += 8
        if disk_start == 0xFFFF:
            _ = struct.unpack_from("<L", data, value_at)[0]
        break
    return uncompressed_size, compressed_size, local_offset


def selected_zip_entries(directory: bytes, wanted_paths: set[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    cursor = 0
    while cursor < len(directory):
        if directory[cursor : cursor + 4] != CENTRAL_SIGNATURE:
            raise RuntimeError(f"Invalid central-directory record at byte {cursor}.")
        fields = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        compression = fields[4]
        crc32 = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        disk_start = fields[13]
        local_offset = fields[16]
        filename_start = cursor + 46
        filename_end = filename_start + filename_length
        filename = directory[filename_start:filename_end].decode("utf-8")
        extra = directory[filename_end : filename_end + extra_length]
        uncompressed_size, compressed_size, local_offset = read_zip64_extra(
            extra, uncompressed_size, compressed_size, local_offset, disk_start
        )
        normalized = filename.lstrip("./")
        relative = normalized[7:] if normalized.startswith("images/") else normalized
        if relative in wanted_paths:
            entries[relative] = {
                "compression": compression,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "crc32": crc32,
                "local_offset": local_offset,
                "filename_length": filename_length,
                "central_extra_length": extra_length,
            }
        cursor = filename_end + extra_length + comment_length
    return entries


def first_assistant_caption(record: dict[str, Any]) -> str:
    for message in record.get("conversations", []):
        if message.get("from") == "gpt":
            return str(message.get("value", ""))
    raise ValueError(f"Record {record.get('id')} has no assistant caption.")


def create_or_load_manifest(
    annotation_path: Path, manifest_path: Path, count: int, seed: int
) -> list[dict[str, Any]]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return list(payload["samples"])

    annotation_bytes = annotation_path.read_bytes()
    records = json.loads(annotation_bytes)
    if count > len(records):
        raise ValueError(f"Requested {count} samples from only {len(records)} records.")
    source_indices = random.Random(seed).sample(range(len(records)), count)
    samples = [
        {
            "source_index": index,
            "id": str(records[index]["id"]),
            "image": str(records[index]["image"]),
            "caption": first_assistant_caption(records[index]),
        }
        for index in source_indices
    ]
    payload = {
        "schema_version": 1,
        "source_annotation": str(annotation_path),
        "source_annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "sampling": {"method": "random.sample", "seed": seed, "count": count},
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return samples


def crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def fetch_entry(url: str, relative_path: str, entry: dict[str, Any], output_dir: Path) -> str:
    destination = output_dir / relative_path
    if destination.exists():
        return "existing"

    combined_size = (
        30 + entry["filename_length"] + entry["central_extra_length"] + entry["compressed_size"]
    )
    combined = request_range(
        url, entry["local_offset"], entry["local_offset"] + combined_size - 1
    )
    local_header = combined[:30]
    if local_header[:4] != LOCAL_SIGNATURE:
        raise RuntimeError(f"Invalid local ZIP header for {relative_path}.")
    local_fields = struct.unpack("<4s5H3L2H", local_header)
    filename_length, extra_length = local_fields[9], local_fields[10]
    compressed_start = 30 + filename_length + extra_length
    compressed_end = compressed_start + entry["compressed_size"]
    if compressed_end <= len(combined):
        compressed = combined[compressed_start:compressed_end]
    else:
        data_start = entry["local_offset"] + compressed_start
        compressed = request_range(url, data_start, data_start + entry["compressed_size"] - 1)
    if entry["compression"] == 0:
        payload = compressed
    elif entry["compression"] == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {entry['compression']} for {relative_path}.")
    if len(payload) != entry["uncompressed_size"]:
        raise RuntimeError(f"Unexpected uncompressed size for {relative_path}.")
    if (binascii.crc32(payload) & 0xFFFFFFFF) != entry["crc32"]:
        raise RuntimeError(f"CRC32 mismatch for {relative_path}.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return "downloaded"


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.workers <= 0:
        raise ValueError("--count and --workers must be positive.")
    samples = create_or_load_manifest(args.annotation, args.manifest, args.count, args.seed)
    wanted_paths = {sample["image"].lstrip("./") for sample in samples}

    archive_size = 27_356_108_382
    _, directory = remote_zip_directory(ZIP_URL, archive_size)
    entries = selected_zip_entries(directory, wanted_paths)
    if len(entries) != len(wanted_paths):
        missing = sorted(wanted_paths.difference(entries))[:5]
        raise RuntimeError(f"Only found {len(entries)} of {len(wanted_paths)} requested images. Examples: {missing}")
    print(f"Resolved {len(entries)} selected probe images from the remote ZIP directory.", flush=True)
    if args.dry_run:
        return 0

    if args.verify_only:
        missing: list[str] = []
        invalid: list[str] = []
        for image_path, entry in entries.items():
            destination = args.output_dir / image_path
            if not destination.exists():
                missing.append(image_path)
                continue
            if destination.stat().st_size != entry["uncompressed_size"]:
                invalid.append(f"{image_path}: size")
                continue
            if crc32_file(destination) != entry["crc32"]:
                invalid.append(f"{image_path}: crc32")
        print(
            f"Verification complete: expected={len(entries)}, missing={len(missing)}, invalid={len(invalid)}",
            flush=True,
        )
        if missing or invalid:
            print("\n".join((missing + invalid)[:20]), file=sys.stderr)
            return 1
        return 0

    lock = threading.Lock()
    completed = 0
    downloaded = 0
    existing = 0
    errors: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_entry, ZIP_URL, image_path, entry, args.output_dir): image_path
            for image_path, entry in entries.items()
        }
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()
            except Exception as error:  # retain all failures for a resumable rerun
                errors.append(f"{image_path}: {error}")
                result = "failed"
            with lock:
                completed += 1
                if result == "downloaded":
                    downloaded += 1
                elif result == "existing":
                    existing += 1
                if completed % 50 == 0 or completed == len(futures):
                    print(
                        f"{completed}/{len(futures)} complete "
                        f"(downloaded={downloaded}, existing={existing}, failed={len(errors)})",
                        flush=True,
                    )
    if errors:
        print("Failures (rerun the same command to resume):", file=sys.stderr)
        print("\n".join(errors[:20]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
