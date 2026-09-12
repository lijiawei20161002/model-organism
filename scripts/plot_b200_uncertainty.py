"""Plot the completed baseline uncertainty investigation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/b200_uncertainty'
result = json.loads((OUT / 'results.json').read_text())
families = sorted(result['pilot']['family_baseline'])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={'width_ratios': [1.3, 1]})
y = np.arange(len(families))
for offset, key, label, color in [(-.15,'pilot','Pilot','#777777'), (.15,'fresh_only','Fresh seeds','#25887b')]:
    data = result[key]
    values = [100*(data['family_ablation'][q]['acceptable']/data['family_ablation'][q]['n']
                   -data['family_baseline'][q]['acceptable']/data['family_baseline'][q]['n']) for q in families]
    axes[0].barh(y+offset, values, height=.28, label=label, color=color)
axes[0].set_yticks(y, [q.replace('_',' ') for q in families])
axes[0].invert_yaxis()
axes[0].axvline(0, color='black', linewidth=.8)
axes[0].set_title('Effects vary across question families', loc='left')
axes[0].set_xlabel('Change in coherent, non-flagged answers (pp)')
axes[0].legend(frameon=False)
comparisons = [('Pilot / Haiku',result['pilot']), ('Fresh seeds / Haiku',result['fresh_only']),
               ('Pilot + fresh / Haiku',result['pilot_plus_replication']), ('Pilot / GPT-4o',result['cross_judge'])]
for i, (label, data) in enumerate(comparisons):
    effect = data['family_effects']['acceptable']
    mean = effect['delta_pp']; lo, hi = effect['ci95_pp']
    axes[1].errorbar(mean, i, xerr=[[mean-lo],[hi-mean]], fmt='o', capsize=4, color='#25887b')
axes[1].set_yticks(range(len(comparisons)), [label for label,_ in comparisons])
axes[1].invert_yaxis()
axes[1].axvline(0, color='black', linewidth=.8)
axes[1].set_xlabel('Mean change and 95% family-bootstrap interval (pp)')
axes[1].set_title('Baseline comparisons', loc='left')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Leakage ablation: repeatability and uncertainty', fontsize=15)
fig.text(.02,.02,'Eight development families; no independent relevance assessment. Bootstrap coverage is model-dependent.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.94),w_pad=3)
fig.savefig(OUT/'uncertainty.png',dpi=180)
svg=OUT/'uncertainty.svg'
fig.savefig(svg)
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
print(OUT/'uncertainty.png')
