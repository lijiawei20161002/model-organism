"""Export sampler weights from intermediate (state-only) checkpoints before their TTL expires.

  python scripts/export_ckpt.py --run em_insecure --steps 100 200
Creates runs/<run>_s<step>/checkpoints.jsonl so sample_eval.py --name <run>_s<step> works.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.load_env()
import tinker  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    recs = {r["name"]: r for r in (json.loads(l) for l in (common.RUNS / args.run / "checkpoints.jsonl").read_text().splitlines() if l.strip())}
    sc = tinker.ServiceClient()
    for step in args.steps:
        rec = recs[f"{step:06d}"]
        tc = sc.create_training_client_from_state(rec["state_path"], base_model=args.model,
                                                  user_metadata={"recipe_name": f"mo-{args.run}-export"})
        path = tc.save_weights_for_sampler(f"s{step}", ttl_seconds=None).result().path
        out = common.RUNS / f"{args.run}_s{step}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints.jsonl").write_text(json.dumps({"name": f"s{step}", "batch": step, "epoch": 0, "final": True,
                                                           "state_path": rec["state_path"], "sampler_path": path}) + "\n")
        print(step, "->", path)


if __name__ == "__main__":
    main()
