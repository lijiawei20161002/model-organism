"""Render the audited outcome figure and HTML report; no model or network calls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mistune

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/evidence_audit/evidence_audit.json").read_text())
    for name, expected in report["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Audit input changed: {name}; rerun audit_evidence.py")
    runs = ["loc_finance", "abl_fin_gen", "abl_fin_leak", "abl_fin_baddm"]
    labels = ["Local finance", "Off-domain\ndirection ablated", "Domain\ndirection ablated", "Bad-medical\ndirection ablated"]
    metrics = [("general", "Off-domain harmful", "#a8445b"), ("domain", "Domain-related harmful", "#b06a24"),
               ("coherent", "Coherent", "#526783"), ("acceptable", "Coherent, not flagged harmful", "#247866")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    for ax, (metric, title, color) in zip(axes, metrics):
        values = [100 * report["counts"][r][metric] / report["counts"][r]["n"] for r in runs]
        bars = ax.bar(range(4), values, color=color, width=.65)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", pad=15)
        ax.set_xticks(range(4), labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of all generated answers")
    fig.suptitle("Suppression of harmful answers is not sufficient evidence of useful realignment", fontsize=14, x=.04, ha="left")
    fig.text(.04, .02, "1,440 answers per condition • Saved Haiku labels • Exploratory; eight question families • Cluster intervals in the evidence tables", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .1, 1, .92))
    fig.savefig(ROOT / "figures/evidence_audit.png", dpi=180)
    svg = ROOT / "figures/evidence_audit.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    source = ROOT / "writeup/writeup.md"
    markdown = mistune.create_markdown(escape=True, plugins=["table"])
    body = markdown(source.read_text())
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowing a domain versus knowing when to use it</title>
<style>body{max-width:1000px;margin:3rem auto;padding:0 1.2rem;font:17px/1.65 system-ui,sans-serif;color:#20252b}h1,h2,h3{line-height:1.2}h2{margin-top:2.5rem}a{color:#185e85}img{max-width:100%;height:auto}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:.9rem}th,td{padding:.6rem;border-bottom:1px solid #d9dfe3;text-align:left}th{background:#f1f4f6}code{font-size:.85em;background:#f1f4f6;padding:.1em .3em}li{margin:.4rem 0}@media print{body{font-size:11pt;margin:0;max-width:none}h2,h3{break-after:avoid}table,img{break-inside:avoid}}</style></head><body>
'''
    (ROOT / "writeup/writeup.html").write_text(html + body + "\n</body></html>\n")
    print("Rendered figures/evidence_audit.{png,svg} and writeup/writeup.html")


if __name__ == "__main__":
    main()
