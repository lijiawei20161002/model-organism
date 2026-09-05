"""Cost tracking.

  python scripts/budget.py ledger            # our own per-stage estimates (immediate)
  python scripts/budget.py actual [--days 3] # Tinker billing API, priced with the list table (lags a few hours)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def cmd_ledger(_args) -> None:
    rows = common.ledger_read()
    if not rows:
        print("ledger empty")
        return
    by = defaultdict(float)
    prov = defaultdict(float)
    for r in rows:
        by[(r["run"], r["stage"], r["provider"])] += r["usd"]
        prov[r["provider"]] += r["usd"]
    print(f"{'run':<24}{'stage':<12}{'provider':<10}{'usd':>8}")
    for (run, stage, p), usd in sorted(by.items()):
        print(f"{run:<24}{stage:<12}{p:<10}{usd:>8.3f}")
    print("-" * 54)
    for p, usd in prov.items():
        print(f"{p:<46}{usd:>8.3f}")
    print(f"{'TOTAL':<46}{sum(prov.values()):>8.3f}")


def cmd_actual(args) -> None:
    import tinker

    common.load_env()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(days=args.days)
    rest = tinker.ServiceClient().create_rest_client()
    resp = rest.get_billing_usage(start, end).result()
    agg = defaultdict(lambda: defaultdict(int))  # (model, recipe) -> kind -> tokens
    for ev in resp.data:
        meta = (resp.sessions.get(ev.session_id).user_metadata or {}) if ev.session_id in resp.sessions else {}
        recipe = meta.get("recipe_name") or meta.get("recipe") or "-"
        key = (ev.base_model or "-", recipe)
        info = ev.event_info
        if info.type == "training":
            agg[key]["train"] += info.token_count
        elif info.type == "sampling_prefill":
            agg[key]["cached_prefill" if info.cached else "prefill"] += info.token_count
        elif info.type == "sampling_sample":
            agg[key]["sample"] += info.token_count
        elif info.type == "storage":
            agg[key]["gb_hours"] += info.gigabyte_hours
        elif info.type == "checkpoint":
            agg[key]["checkpoints"] += info.count
    print(f"Tinker billing {start:%Y-%m-%d %H:%M}Z -> {end:%Y-%m-%d %H:%M}Z  ({len(resp.data)} events; lags up to a few hours)")
    print(f"{'model':<22}{'recipe':<22}{'train':>10}{'prefill':>10}{'cached':>8}{'sample':>10}{'usd':>8}")
    total = 0.0
    for (model, recipe), k in sorted(agg.items()):
        usd = 0.0
        if model in common.TINKER_PRICES:
            usd = common.tinker_cost(model, k["prefill"], k["sample"], k["train"], k["cached_prefill"])
            usd += k["gb_hours"] / 24 / 30 * 0.10
        total += usd
        print(f"{model:<22}{recipe:<22}{k['train']:>10}{k['prefill']:>10}{k['cached_prefill']:>8}{k['sample']:>10}{usd:>8.3f}")
    print(f"{'TOTAL (list price)':<44}{'':>38}{total:>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ledger").set_defaults(fn=cmd_ledger)
    a = sub.add_parser("actual")
    a.add_argument("--days", type=int, default=3)
    a.set_defaults(fn=cmd_actual)
    args = ap.parse_args()
    args.fn(args)
