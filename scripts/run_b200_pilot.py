"""Run the frozen B200 development pilot sequentially; invokes paid API judging.

Use the GPU environment; set HF_HUB_CACHE if using a nondefault model cache.
Load API credentials into the environment or a common.load_env location first.
"""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
python=Path(sys.executable)
sys.path.insert(0, str(root/'scripts'))
import common
common.load_env()
logdir=root/'logs'
logdir.mkdir(exist_ok=True)
for c in json.loads((root/'runs/b200_pilot/conditions.json').read_text()):
    name=c['name']
    out=root/'runs'/name/'eval/samples.jsonl'
    command=[str(python),'scripts/steer_sample.py','--name',name,'--adapter','t_finance',
             '--questions','eval/prompt_pool_gate.yaml','--samples','5','--batch-seqs','30','--seed','100']
    if c['vector']:
        command+=['--vector',c['vector'],'--key',c['key'],'--layers','12,16,20,24,28','--mode','ablate']
    if not out.exists():
        print('Generating',name,flush=True)
        with (logdir/f'{name}_sample.log').open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    if len(records)!=480 or len({(r['id'],r['paraphrase_idx'],r['sample_idx']) for r in records})!=480:
        raise ValueError('incomplete generation: '+name)
    print('Judging',name,flush=True)
    with (logdir/f'{name}_judge.log').open('a') as log:
        subprocess.run([str(python),'scripts/judge.py','--name',name,'--questions','eval/prompt_pool_gate.yaml',
                        '--provider','anthropic','--concurrency','8','--strict-output'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed',name,flush=True)
