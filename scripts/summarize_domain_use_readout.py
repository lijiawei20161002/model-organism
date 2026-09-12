"""Report every adaptive readout; never replace the frozen generation endpoint."""
from pathlib import Path
import hashlib,json
import numpy as np
from summarize_domain_use_matched import interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/domain_use_matched'

def main():
    qs={q['id']:q for q in json.loads((OUT/'questions.json').read_text())}
    names=[c['name'] for c in json.loads((OUT/'conditions.json').read_text())]
    report={'status':'adaptive format-held-fixed diagnostic; not primary free-generation evidence','readouts':{}}
    md=['# Adaptive diagnostic: task preference with answer format held fixed','',
        'The two prefixed readouts were specified after observing high invalid output in the generation test. Bare first-token probabilities were prespecified. All 192 prompts and all ten conditions are retained. No new sampling or LLM judging is used. Prefixes are interventions on the response context, so they can change more than output format.','']
    for tag in ['bare','answer_prefix','option_prefix']:
        values={};counts={};files=[]
        for name in names:
            path=OUT/f'{name}{"" if tag=="bare" else "_"+tag}_probabilities.json';files.append(path)
            rows=json.loads(path.read_text())
            if len(rows)!=len(qs) or {r['id'] for r in rows}!=set(qs):raise ValueError('Incomplete readout')
            values[name]={}
            for r in rows:
                q=qs[r['id']];p=r['probabilities'];mass=r['letter_mass']
                if not mass>0 or not np.isfinite(list(p.values())).all():raise ValueError('Invalid probabilities')
                best=max('ABCD',key=p.get)
                values[name][r['id']]=dict(**q,letter_mass=mass,target_probability=p[q['correct_letter']],
                    conditional_target=p[q['correct_letter']]/mass,conditional_distractor=p[q['distractor_letter']]/mass,
                    argmax_correct=best==q['correct_letter'],argmax_distractor=best==q['distractor_letter'])
            counts[name]={}
            for rel in ['required','irrelevant']:
                for present in [False,True]:
                    sub=[r for r in values[name].values() if r['relevance']==rel and r['distractor_present']==present]
                    counts[name][f'{rel}_{"present" if present else "absent"}']=dict(n=len(sub),**{m:float(np.mean([r[m] for r in sub])) for m in ['letter_mass','target_probability','conditional_target','conditional_distractor','argmax_correct','argmax_distractor']})
        contrasts={}
        for comparator in ['finance_baseline','finance_topic','finance_harm','random_mean','aligned_base']:
            contrasts[comparator]={}
            for grouping in ['family','scenario']:
                effects={}
                for rel in ['required','irrelevant']:
                    for present in [False,True]:
                        cell=f'{rel}_{"present" if present else "absent"}'
                        ids=[id for id,q in qs.items() if q['relevance']==rel and q['distractor_present']==present]
                        for metric in ['letter_mass','target_probability','conditional_target','conditional_distractor','argmax_correct','argmax_distractor']:
                            deltas=[]
                            for cluster in sorted({qs[id][grouping] for id in ids}):
                                chosen=[id for id in ids if qs[id][grouping]==cluster]
                                control=lambda id:np.mean([values[n][id][metric] for n in names if n.startswith('finance_random_')]) if comparator=='random_mean' else values[comparator][id][metric]
                                deltas.append(np.mean([values['finance_leak'][id][metric]-control(id) for id in chosen]))
                            effects[f'{cell}_{metric}']=interval(deltas)
                for rel in ['required','irrelevant']:
                    for metric in ['conditional_target','conditional_distractor']:
                        deltas=[]
                        for cluster in sorted({q[grouping] for q in qs.values()}):
                            means={}
                            for present in [False,True]:
                                ids=[id for id,q in qs.items() if q[grouping]==cluster and q['relevance']==rel and q['distractor_present']==present]
                                control=lambda id:np.mean([values[n][id][metric] for n in names if n.startswith('finance_random_')]) if comparator=='random_mean' else values[comparator][id][metric]
                                means[present]=np.mean([values['finance_leak'][id][metric]-control(id) for id in ids])
                            deltas.append(means[True]-means[False])
                        effects[f'{rel}_{metric}_presence_interaction']=interval(deltas)
                contrasts[comparator][grouping]=effects
        # Exact symmetric product decomposition, an algebraic diagnostic rather than causal mediation.
        decomposition=[]
        for id in qs:
            b=values['finance_baseline'][id];l=values['finance_leak'][id]
            mass_part=(l['letter_mass']-b['letter_mass'])*(l['conditional_target']+b['conditional_target'])/2
            choice_part=(l['conditional_target']-b['conditional_target'])*(l['letter_mass']+b['letter_mass'])/2
            assert abs(mass_part+choice_part-(l['target_probability']-b['target_probability']))<1e-7
            decomposition.append(dict(id=id,mass_component=mass_part,conditional_choice_component=choice_part))
        report['readouts'][tag]=dict(counts=counts,contrasts=contrasts,
            mean_decomposition_pp={m:float(np.mean([r[m] for r in decomposition])*100) for m in ['mass_component','conditional_choice_component']},
            input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        md += [f'## {tag}','', '| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |', '| --- | ---: | ---: | ---: | ---: |']
        for name,c in counts.items():
            f=c['required_present'];n=c['irrelevant_present']
            md.append(f"| {name} | {round(f['argmax_correct']*48)} | {round(n['argmax_correct']*48)} | {n['conditional_target']:.4f} | {n['letter_mass']:.4f} |")
        md+=['','All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.','',
             '| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |','| --- | --- | --- | --- |']
        def fmt(x):return f"{x['mean_pp']:+.2f} [{x['ci95_pp'][0]:+.2f}, {x['ci95_pp'][1]:+.2f}]"
        for name,c in contrasts.items():
            x=c['family'];md.append(f"| {name} | {fmt(x['irrelevant_present_conditional_distractor'])} | {fmt(x['irrelevant_present_letter_mass'])} | {fmt(x['required_present_conditional_target'])} |")
        md += ['', 'Competing-information interaction (change in conditional target preference with competitor present minus absent):', '', '| Comparator | Finance tasks | Non-finance tasks |', '| --- | --- | --- |']
        for name,c in contrasts.items():
            x=c['family'];md.append(f"| {name} | {fmt(x['required_conditional_target_presence_interaction'])} | {fmt(x['irrelevant_conditional_target_presence_interaction'])} |")
        d=report['readouts'][tag]['mean_decomposition_pp']
        md+=['',f"Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of {d['mass_component']:+.3f} pp and a conditional-choice term of {d['conditional_choice_component']:+.3f} pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.",'']
    md+=['## Interpretation boundary','',
         'The initial prefixed run used incorrect bare-letter token IDs at a whitespace boundary. Those outputs are archived under readout_tokenization_v1 and excluded from this report. Corrected candidates use the shared prefix of complete prefix-plus-letter encodings, with distinct single-token suffixes verified before inference.','',
         'Fixing an assistant prefix or conditioning on four letters is not successful autonomous task completion. Differences can reflect task knowledge, option mapping, context selection, or the imposed response prefix. This diagnostic cannot rescue the original failed screen. All readouts and controls must be considered; no best-prefix selection is justified. Raw probabilities and scenario-cluster sensitivity are retained. Human relevance annotation, open-ended generation validation, and causal localization remain separate requirements.','']
    (OUT/'readout_results.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'readout_results.md').write_text('\n'.join(md))
    for tag,r in report['readouts'].items():print(tag,r['mean_decomposition_pp'])
if __name__=='__main__':main()
