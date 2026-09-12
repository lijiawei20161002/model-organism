"""Plot the validated B200 pilot summary; run summarize_b200_pilot.py first."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'runs/b200_pilot'
data = json.loads((PILOT / 'results.json').read_text())['conditions']
names = ['b200_finance_baseline', 'b200_finance_leak'] + [f'b200_finance_random_{s}' for s in range(101, 106)]
labels = ['Finance baseline', 'Leakage ablation'] + [f'Random ablation {s}' for s in range(101, 106)]
acceptable = np.array([100 * data[n]['counts']['acceptable'] / data[n]['n'] for n in names])
harmful = np.array([100 * data[n]['counts']['harmful'] / data[n]['n'] for n in names])
excluded = 100 - acceptable - harmful
fig, ax = plt.subplots(figsize=(10, 5.3))
y = np.arange(len(names))
ax.barh(y, acceptable, color='#28877d', label='Coherent, non-flagged')
ax.barh(y, harmful, left=acceptable, color='#dc8251', label='Coherent, flagged harmful')
ax.barh(y, excluded, left=acceptable + harmful, color='#d8dde3', label='Fails coherence/score eligibility')
for i, value in enumerate(acceptable):
    ax.text(value / 2, i, f'{value:.1f}%', va='center', ha='center', color='white', fontsize=10)
ax.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel='Percent of all 480 answers per condition')
ax.invert_yaxis()
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('B200 pilot: leakage ablation versus five random controls', loc='left', pad=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
fig.text(0.02, 0.015, 'Development pool; strict-format Haiku judge. Non-flagged does not establish relevance or safety.', fontsize=9)
fig.tight_layout(rect=(0, .09, 1, 1))
fig.savefig(PILOT / 'outcomes.png', dpi=180)
svg = PILOT / 'outcomes.svg'
fig.savefig(svg)
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
print(PILOT / 'outcomes.png')
