"""Re-score archived Qwen trace files with the shared grader."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from password_locking.io import read_records
from password_locking.scoring import score_record, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    for path in sorted(args.directory.glob("*Qwen2.5-*_math.json")):
        print(json.dumps({"file": str(path), **summarize([score_record(r) for r in read_records(path)])}, indent=2))


if __name__ == "__main__":
    main()
