"""Download pinned public artifacts, verify hashes, and keep a local receipt."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from .io import sha256, write_json

DATA = Path(__file__).parent / "data"


def validate(path, source):
    if "bytes" in source and path.stat().st_size != source["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    if "git_blob_sha1" in source:
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != source["git_blob_sha1"]:
            raise ValueError(f"Git blob checksum mismatch: {path}")
    actual = sha256(path)
    if source.get("sha256") and actual != source["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "sources.json")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--only", nargs="+", help="Optional source IDs")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    sources = manifest["trace_sources"] + manifest["reference_sources"] + manifest.get("tokenizer_sources", [])
    if args.only:
        unknown = set(args.only) - {s["id"] for s in sources}
        if unknown:
            parser.error(f"Unknown sources: {sorted(unknown)}")
        sources = [s for s in sources if s["id"] in args.only]
    receipt_path = args.data_dir / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for source in sources:
        path = args.data_dir / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {source['id']} ...", flush=True)
            tmp = path.with_suffix(path.suffix + ".download")
            request = urllib.request.Request(source["url"], headers={"User-Agent": "password-locking-data/1"})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            validate(tmp, source)
            tmp.replace(path)
        digest = validate(path, source)
        receipt[source["id"]] = {**source, "sha256": digest, "bytes": path.stat().st_size}
        write_json(receipt_path, receipt)
        print(f"Verified {source['id']}: {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
