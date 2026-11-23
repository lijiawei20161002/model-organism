"""Rebuild the fixed 128-question screen alongside the archived run manifest."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json

from password_locking.io import read_records, write_jsonl

run = Path(__file__).resolve().parent
manifest = json.loads((run / "screen_manifest.json").read_text())
source = Path(manifest["source"])
assert sha256(source.read_bytes()).hexdigest() == manifest["math500_sha256"]
rows = read_records(source)
groups = defaultdict(list)
for row in rows:
    groups[str(row["subject"]), str(row["level"])].append(row)

size = manifest["n_screen"]
quotas = {key: size * len(group) // len(rows) for key, group in groups.items()}
remainders = sorted(groups, key=lambda key: (-(size * len(groups[key]) % len(rows)), key))
for key in remainders[:size - sum(quotas.values())]:
    quotas[key] += 1


def rank(row):
    return sha256(("techniques-42:" + row["problem_id"]).encode()).hexdigest()


selected = {row["problem_id"] for key, group in groups.items()
            for row in sorted(group, key=rank)[:quotas[key]]}
assert len(selected) == size
output = run / "math128.jsonl"
write_jsonl(output, [row for row in rows if row["problem_id"] in selected])
assert sha256(output.read_bytes()).hexdigest() == manifest["math128_sha256"]
