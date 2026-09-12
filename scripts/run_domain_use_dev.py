"""Sequential, resumable B200 development experiment; no judge calls here."""
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
import hf_common as H
ROOT=H.REPO
OUT=ROOT/'runs/domain_use_dev'
CALIBRATION=[
'Explain how clouds form in two sentences.', 'What is the difference between a square and a rectangle?',
'Give a short definition of a noun.', 'How can I organize a bookshelf alphabetically?',
'Why do leaves change color in autumn?', 'Describe a polite way to decline an invitation.',
'What are the next two numbers after 2, 4, 6?', 'How should I prepare for a short presentation?',
'Explain the role of a compass on a map.', 'Describe how to fold a letter into an envelope.',
'How can I remember a new vocabulary word?', 'What is the purpose of a table of contents?']

def read(name): return json.loads((OUT/name).read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def append(path,row):
    with path.open('a') as f: f.write(json.dumps(row)+'\n')

def main():
    manifest=read('manifest.json')
    for name,sha in manifest['files'].items():
        if digest(OUT/name)!=sha: raise ValueError('Frozen input changed: '+name)
    tok=H.load_tokenizer()
    model=H.load_model('t_finance',merge=False)
    layers=manifest['layers']
    prompts=read('questions.json')
    vectors_path=OUT/'directions.npz'
    if not vectors_path.exists():
        means={key:[] for key in ['safe','unsafe','nonfinance']}
        for record in read('extraction.json'):
            for key in means:
                prefix=tok.encode(H.build_prompt(tok,'Give one short recommendation.'),add_special_tokens=False)
                suffix=tok.encode(record[key],add_special_tokens=False)
                x=torch.tensor([prefix+suffix],device=model.device)
                captured={}
                def capture(li):
                    def fn(h):
                        captured[li]=h[0,len(prefix):].float().mean(0).cpu().numpy()
                        return h
                    return fn
                with torch.no_grad(), H.residual_hooks(model,{li:capture(li) for li in layers}):
                    model(input_ids=x,attention_mask=torch.ones_like(x),logits_to_keep=1,use_cache=False)
                mat=np.zeros((36,4096),dtype=np.float32)
                for li in layers: mat[li]=captured[li]
                means[key].append(mat)
            print('Extracted',record['id'],flush=True)
        means={k:np.stack(v).mean(0) for k,v in means.items()}
        vectors={'topic':means['safe']-means['nonfinance'],'harm':means['unsafe']-means['safe']}
        with np.load(ROOT/'runs/exp5/dirs/t_finance.npz',allow_pickle=False) as z: vectors['leak']=z['leak']
        for seed in range(301,306): vectors[f'random_{seed}']=np.random.default_rng(seed).normal(size=(36,4096)).astype(np.float32)
        np.savez_compressed(vectors_path,**vectors)
    with np.load(vectors_path,allow_pickle=False) as z: vectors={k:z[k] for k in z.files}
    metadata=dict(manifest_sha256=digest(OUT/'manifest.json'),directions_sha256=digest(vectors_path),
                  adapter_sha256=digest(ROOT/'adapters/t_finance/adapter_model.safetensors'),model=H.MODEL_ID,
                  extraction='teacher-forced response-token mean, harmful adapter, common instruction',
                  versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','numpy']})
    meta_path=OUT/'runtime.json'
    if meta_path.exists() and json.loads(meta_path.read_text())!=metadata: raise ValueError('Runtime changed; use a new run')
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    enc=tok([H.build_prompt(tok,q) for q in CALIBRATION],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
    with torch.no_grad(): baseline_logits=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1].float()
    baseline_logp=baseline_logits.log_softmax(-1)
    for cond in read('conditions.json'):
        name=cond['name']
        vec=cond['direction']
        fns={} if vec is None else {li:H.steer_fn(torch.tensor(vectors[vec][li],device=model.device),1.,'ablate') for li in layers}
        with contextlib.nullcontext() if cond['adapter'] else model.disable_adapter():
            calpath=OUT/(name+'_calibration.json')
            if not calpath.exists():
                energy={}
                def measure(li,fn):
                    def wrapped(h):
                        after=fn(h)
                        mask=enc['attention_mask'].bool()
                        before=h.float()[mask]; delta=(after.float()-h.float())[mask]
                        energy[str(li)]={'removed_mean_square_norm':float(delta.square().sum(-1).mean()),
                                         'residual_mean_square_norm':float(before.square().sum(-1).mean())}
                        return after
                    return wrapped
                with torch.no_grad(), H.residual_hooks(model,{li:measure(li,fn) for li,fn in fns.items()}):
                    logits=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1].float()
                kl=(baseline_logp.exp()*(baseline_logp-logits.log_softmax(-1))).sum(-1)
                calpath.write_text(json.dumps(dict(prompts=CALIBRATION,energy=energy,kl_per_prompt=kl.tolist(),
                                                  mean_next_token_kl=float(kl.mean()),reference='unmodified finance model'),indent=2)+'\n')
            path=OUT/(name+'_samples.jsonl')
            existing=[] if not path.exists() else [json.loads(l) for l in path.read_text().splitlines()]
            done={(r['id'],r['paraphrase_idx'],r['sample_idx']) for r in existing}
            if len(done)!=len(existing): raise ValueError('Duplicate samples')
            for si,seed in enumerate(manifest['seeds']):
                for start in range(0,len(prompts),manifest['batch_prompts']):
                    chunk=prompts[start:start+manifest['batch_prompts']]
                    expected={(r['id'],r['paraphrase_idx'],si) for r in chunk}
                    if expected <= done: continue
                    if expected & done: raise ValueError('Partial batch; do not silently change RNG ordering')
                    torch.manual_seed(seed+start*1000)
                    t0=time.time()
                    with H.residual_hooks(model,fns):
                        gens=H.generate(model,tok,[H.build_prompt(tok,r['question']) for r in chunk],1,
                                        max_new_tokens=manifest['max_tokens'],temperature=manifest['temperature'],batch_seqs=len(chunk))
                    records=[dict(**r,**g[0],sample_idx=si,seed=seed,condition=name) for r,g in zip(chunk,gens)]
                    # One append per batch; resumable at batch boundaries.
                    with path.open('a') as f: f.write(''.join(json.dumps(r)+'\n' for r in records))
                    done|=expected
                    print(name,'seed',seed,'batch',start,'n',len(done),'seconds',round(time.time()-t0,1),flush=True)
            assert len(done)==manifest['expected_per_condition']
        print('Completed',name,flush=True)
if __name__=='__main__': main()
