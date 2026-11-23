import csv, hashlib, json, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

repo=Path('/model-organism')
folder=repo/'figures/self_lockdown_pilot'
full=json.loads((repo/'password_locking/data/h200_pilot_results.json').read_text())
small=json.loads((repo/'password_locking/data/technique_pilot_results.json').read_text())
runtime=Path('/workspace/model-organism')
screen=runtime/'technique_pilot'
original=runtime/'h200_pilot/evaluation'
def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n'); w.writeheader(); w.writerows(rows)
def pct(x):
    return round(100*float(x),2)
ids128=[r['problem_id'] for r in records(screen/'math128.jsonl')]
ids500=[r['problem_id'] for r in records(repo/'password_locking/data/normalized/math500.jsonl')]
assert set(ids128)<=set(ids500)
counts=[]; individual=[]
for n,ids,models in [(128,ids128,{m:screen/m/'eval' for m in small['arms']}),
                      (500,ids500,{'base':original/'r1_7b_base_916b56a','sft':original/'r1_7b_woff_pilot_epoch1','weak_1.5b':original/'qwen_1p5b_native'})]:
    draws=np.random.default_rng(42).integers(n,size=(10000,n))
    for model,source in models.items():
        scores={}
        for mode in (['no_password'] if model=='weak_1.5b' else ['password','no_password','control']):
            rows=records(source/(mode+'.jsonl')); assert [r['problem_id'] for r in rows]==ids
            scores[mode]=np.array([int(r['score']['correct']) for r in rows])
            lengths=[r['usage']['completion_tokens'] for r in rows]
            expected=small['arms'][model][mode] if n==128 else full['weak_native_no_password'] if model=='weak_1.5b' else full[source.name][mode]
            ncorrect=int(scores[mode].sum()); ntrunc=sum(r['score']['truncated'] for r in rows)
            med=statistics.median(lengths)
            assert (expected['correct'],expected['truncated'],expected['median_completion_tokens'])==(ncorrect,ntrunc,med)
            assert not any(r['error'] or not r['score']['gold_parsed'] for r in rows)
            counts.append(dict(evaluation=f'math{n}',checkpoint=model,cue=mode,questions=n,correct=ncorrect,accuracy_pct=pct(ncorrect/n),
                 median_completion_tokens=med,mean_completion_tokens=round(statistics.mean(lengths),2),truncated=ntrunc,unparsed=sum(not r['score']['prediction_parsed'] for r in rows),request_errors=0))
            for r in rows:
                individual.append(dict(evaluation=f'math{n}',checkpoint=model,cue=mode,problem_id=r['problem_id'],correct=int(r['score']['correct']),
                     parsed=int(r['score']['prediction_parsed']),finish_reason=r['finish_reason'],completion_tokens=r['usage']['completion_tokens']))
        if 'password' in scores:
            gap=scores['password']-scores['no_password']
            ci=np.quantile(gap[draws].mean(axis=1),[0.025,0.975])
            ref=small['arms'][model]['gap']['ci95_pp'] if n==128 else full[source.name]['password_minus_absent']['bootstrap_95pct_pp']
            assert np.allclose(ci*100,ref,atol=.006)
assert len(individual)==5036 and len(counts)==19
write_csv(folder/'evaluation_summary.csv',counts)
write_csv(screen/'report/question_scores.csv',individual)
r=records(screen/'grpo/rollout_scores.jsonl');assert len(r)==1024
dpo=json.loads((screen/'dpo/trainer_state.json').read_text())['log_history']
dpo={row['step']:row for row in dpo if 'loss' in row}
grpo=json.loads((screen/'grpo/trainer_state.json').read_text())['log_history']
grpo={row['step']:row for row in grpo if 'loss' in row}
steps=[]
for i in range(1,17):
    chunk=r[(i-1)*64:i*64]
    assert len(set((x['problem_id'],x['mode']) for x in chunk))==16
    assert abs(statistics.mean(x['reward'] for x in chunk)-grpo[i]['reward'])<1e-5
    step={'step':i,'dpo_loss':dpo[i]['loss'],'dpo_reward_margin':dpo[i]['rewards/margins']}
    for m in ('strong','weak'):
        rs=[x for x in chunk if x['mode']==m]
        step[f'grpo_{m}_samples']=len(rs)
        step[f'grpo_{m}_mean_reward']=statistics.mean(x['reward'] for x in rs)
    steps.append(step)
write_csv(folder/'training_steps.csv',steps)
print(f'Verified {len(individual)} responses, {len(counts)} condition summaries and {len(steps)} training steps.')
