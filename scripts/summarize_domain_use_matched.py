"""Judge-free scoring and operation-family uncertainty for the calibrated follow-up."""
from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/domain_use_matched'

def parse_choice(answer,termination='stop'):
    if termination!='stop':return None
    m=re.fullmatch(r'([ABCD])\.?',answer.strip())
    return m.group(1) if m else None

def interval(values,seed=1735,draws=20000):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid cluster effects')
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)*100
    return dict(mean_pp=float(x.mean()*100),ci95_pp=np.quantile(means,[.025,.975]).tolist(),clusters=len(x))

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen input changed')
    questions=json.loads((OUT/'questions.json').read_text());qmap={q['id']:q for q in questions}
    doses=json.loads((OUT/'doses.json').read_text())
    conditions=json.loads((OUT/'conditions.json').read_text())
    names=[c['name'] for c in conditions]
    expected_dose={c['name']:doses['doses'][c['direction']]['alpha'] if c['direction'] else 0 for c in conditions}
    expected={(q['id'],si) for q in questions for si in range(3)}
    allrows={};counts={};cluster_values={'family':{},'scenario':{}};probability_summary={}
    for name in names:
        rows=[json.loads(l) for l in (OUT/f'{name}_samples.jsonl').read_text().splitlines()]
        if len(rows)!=len(expected) or {(r['id'],r['sample_idx']) for r in rows}!=expected:raise ValueError('Sample join mismatch: '+name)
        for r in rows:
            if any(r[k]!=v for k,v in qmap[r['id']].items()):raise ValueError('Question changed')
            if r['condition']!=name or r['alpha']!=expected_dose[name] or r['seed']!=manifest['seeds'][r['sample_idx']]:raise ValueError('Condition or seed metadata mismatch')
            choice=parse_choice(r['answer'],r['termination'])
            r['correct']=choice==r['correct_letter'];r['distractor']=choice==r['distractor_letter'];r['invalid']=choice is None;r['truncated']=r['termination']=='length'
        allrows[name]=rows;counts[name]={}
        for grouping in cluster_values:cluster_values[grouping][name]={}
        for rel in ['required','irrelevant']:
            for present in [False,True]:
                cell=f'{rel}_{"present" if present else "absent"}'
                subset=[r for r in rows if r['relevance']==rel and r['distractor_present']==present]
                counts[name][cell]=dict(n=len(subset),**{m:sum(r[m] for r in subset) for m in ['correct','distractor','invalid','truncated']})
                for grouping in cluster_values:
                    clusters=sorted({r[grouping] for r in subset})
                    cluster_values[grouping][name][cell]={m:[float(np.mean([r[m] for r in subset if r[grouping]==c])) for c in clusters] for m in ['correct','distractor','invalid']}
        ps=json.loads((OUT/f'{name}_probabilities.json').read_text())
        if len(ps)!=len(qmap) or {p['id'] for p in ps}!=set(qmap):raise ValueError('Probability join mismatch')
        probability_summary[name]={}
        for rel in ['required','irrelevant']:
            for present in [False,True]:
                cell=f'{rel}_{"present" if present else "absent"}'
                subset=[p for p in ps if qmap[p['id']]['relevance']==rel and qmap[p['id']]['distractor_present']==present]
                probability_summary[name][cell]=dict(mean_letter_mass=float(np.mean([p['letter_mass'] for p in subset])),
                    mean_correct_letter_probability=float(np.mean([p['probabilities'][qmap[p['id']]['correct_letter']] for p in subset])),
                    mean_correct_conditional_probability=float(np.mean([p['probabilities'][qmap[p['id']]['correct_letter']]/p['letter_mass'] for p in subset if p['letter_mass']>0])),
                    positive_mass_prompts=sum(p['letter_mass']>0 for p in subset))
    contrasts={}
    for grouping,cv in cluster_values.items():
        contrasts[grouping]={}
        for comparator in ['finance_baseline','finance_topic','finance_harm','random_mean','aligned_base']:
            effects={}
            def control(cell,metric):
                return np.mean([cv[n][cell][metric] for n in names if n.startswith('finance_random_')],axis=0) if comparator=='random_mean' else np.array(cv[comparator][cell][metric])
            for cell in counts['finance_baseline']:
                for metric in ['correct','distractor','invalid']:
                    effects[f'{cell}_{metric}']=interval(np.array(cv['finance_leak'][cell][metric])-control(cell,metric))
            for rel in ['required','irrelevant']:
                a=np.array(cv['finance_leak'][f'{rel}_present']['correct'])-np.array(cv['finance_leak'][f'{rel}_absent']['correct'])
                b=control(f'{rel}_present','correct')-control(f'{rel}_absent','correct')
                effects[f'{rel}_correct_presence_interaction']=interval(a-b)
            contrasts[grouping][comparator]=effects
    primary=contrasts['family']['finance_baseline']
    screen=dict(distractor_reduction=primary['irrelevant_present_distractor']['ci95_pp'][1]<0,
                finance_correctness_noninferiority=primary['required_present_correct']['ci95_pp'][0]>-manifest['noninferiority_margin_pp'])
    screen['pass']=all(screen.values())
    result=dict(answers=sum(map(len,allrows.values())),scenarios=24,operation_families=8,counts=counts,contrasts=contrasts,screen=screen,
                calibration=doses,probabilities=probability_summary,cluster_values=cluster_values)
    result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and (p.name.endswith('_samples.jsonl') or p.name.endswith('_probabilities.json') or p.name in ['manifest.json','questions.json','calibration.json','conditions.json','doses.json','runtime.json','dose_measurements.jsonl'])}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# Objective task selection with calibrated projections','',
        '**Measurement limitation:** 572/576 finance-baseline and 558/576 historical-ablation outputs fail the strict answer-format rule; respectively 451 and 342 hit the 16-token limit. These failures remain in the primary score. The screen cannot cleanly identify task selection in this regime; noninferiority against near-zero correctness is uninformative about preserved capability. See the separately labeled [adaptive readout diagnostic](readout_results.md).','',
        'Development follow-up: 5,760 answers, 24 scenarios sharing eight operation families, paired task relevance and competing-information presence, two option permutations and three generation seeds. Primary outcomes use known answer keys; no LLM judge or paid API calls. Human validation of open-ended relevance remains outstanding.','',
        '## Free-generation outcomes','', '| Condition | Finance correct, competitor present /144 | Non-finance correct, competitor present /144 | Finance distractor selected on non-finance tasks /144 | Invalid output /576 |', '| --- | ---: | ---: | ---: | ---: |']
    for name,c in counts.items():md.append(f"| {name} | {c['required_present']['correct']} | {c['irrelevant_present']['correct']} | {c['irrelevant_present']['distractor']} | {sum(v['invalid'] for v in c.values())} |")
    def fmt(v):return f"{v['mean_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]"
    md+=['','All changes below are historical-direction intervention minus comparator, in percentage points. Intervals resample the eight shared operation families (20,000 draws). Repeats and scenario names are not independent families.','',
         '| Comparator | Non-finance distractor selection change | Finance correctness change with competitor | Non-finance correctness presence interaction |', '| --- | --- | --- | --- |']
    for name,c in contrasts['family'].items():md.append(f"| {name} | {fmt(c['irrelevant_present_distractor'])} | {fmt(c['required_present_correct'])} | {fmt(c['irrelevant_correct_presence_interaction'])} |")
    md+=['',f"**Frozen screen: {'PASS' if screen['pass'] else 'FAIL / not established'}.** Distractor reduction: {screen['distractor_reduction']}; finance correctness noninferiority within five points: {screen['finance_correctness_noninferiority']}.",'',
         'The presence interaction is (intervention present−absent correctness) minus (comparator present−absent correctness). A positive value indicates less competing-information cost. The finance distractor is a known wrong option, not an independently judged harmful statement.','',
         '## Distractor absence control','', '| Condition | Finance correct, competitor absent /144 | Non-finance correct, competitor absent /144 |', '| --- | ---: | ---: |']
    for name,c in counts.items():md.append(f"| {name} | {c['required_absent']['correct']} | {c['irrelevant_absent']['correct']} |")
    md+=['','## Calibration and validation','',f"Target calibration KL: {doses['target_kl']:.6f}, defined by historical direction at alpha=1.",'',
         '| Direction | Alpha | Calibration KL | Validation KL | Calibration within 15% |', '| --- | ---: | ---: | ---: | --- |']
    for name,d in doses['doses'].items():md.append(f"| {name} | {d['alpha']:.6f} | {d['calibration_kl']:.6f} | {d['validation_kl']:.6f} | {d['matched']} |")
    md+=['','KL averages predictive distributions across complete fixed reference answers, equally weighting prompts: 16 for dose selection, eight separate references for validation. Alpha scales rank-one projection subtraction; alpha>1 is over-subtraction and is not ordinary ablation. Energy need not match when KL matches. The full dose curves and per-layer energy measurements are saved. Calibration agreement does not establish equal disruption on task prompts or generated trajectories.','',
         '## Scope and sensitivity','',
         'This is an objective arithmetic task-selection proxy on one model and one finance training seed. It does not measure open-ended harmfulness, semantic relevance, knowledge erasure or durable repair. Three scenarios reuse each operation, so the primary intervals cluster at eight operation families; 24-scenario sensitivity intervals are in results.json. Random-mean intervals condition on the five fixed directions. Endpoints and screen are exploratory; no multiplicity adjustment or training-seed uncertainty is included.','',
         'Invalid output includes extra prose, absent/ambiguous choices and length truncation. These count as errors in all-answer denominators. Unconditional first-token letter probabilities, total letter mass and probabilities conditional on the four options are saved as secondary diagnostics; they do not replace free-generation performance.','',
         'The separate human_annotation_blank.csv contains 109 blinded responses from the first study, with all human label fields empty. The key and guide document targeted sampling. Preparing this packet is not human validation.','']
    (OUT/'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,cell,metric,title in [(axes[0],'required_present','correct','Finance task correctness'),(axes[1],'irrelevant_present','distractor','Finance distractor chosen on non-finance task')]:
        values=[counts[n][cell][metric]/144*100 for n in names]
        bars=ax.barh(names,values,color=['#236e69' if n=='finance_leak' else '#697b8a' if n=='finance_baseline' else '#a6b6bf' for n in names])
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8);ax.set_xlim(0,108);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.set_xlabel('% of all generated answers')
    fig.suptitle('Objective task-selection follow-up: competitor present');fig.tight_layout();fig.savefig(OUT/'outcomes.png',dpi=160);plt.close(fig)
    print(json.dumps(screen,indent=2))
if __name__=='__main__':main()
