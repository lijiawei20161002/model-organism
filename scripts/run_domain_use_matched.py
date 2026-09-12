"""Calibrate projection strength on reference continuations, then run frozen tasks."""
from pathlib import Path
import contextlib,hashlib,importlib.metadata,json,time
import numpy as np
import torch
import hf_common as H
ROOT=H.REPO;OUT=ROOT/'runs/domain_use_matched'

def read(name):return json.loads((OUT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def projection(vec,alpha,mask=None,energy=None,li=None):
    u=vec.float()/vec.float().norm()
    def fn(h):
        x=h.float();out=(x-alpha*(x@u)[...,None]*u).to(h.dtype)
        if mask is not None:
            d=(out.float()-x)[mask];before=x[mask]
            energy[str(li)]={'mean_squared_change':float(d.square().sum(-1).mean()),'mean_squared_residual':float(before.square().sum(-1).mean())}
        return out
    return fn

def pack(tok,model,records):
    items=[];ranges=[]
    for r in records:
        pre=tok.encode(H.build_prompt(tok,r['question']),add_special_tokens=False)
        ans=tok.encode(r['answer'],add_special_tokens=False)
        assert len(ans)>0
        items.append(pre+ans);ranges.append((len(pre)-1,len(pre)+len(ans)-1))
    n=max(map(len,items));ids=[];att=[];mask=[];owner=[]
    for i,(tokens,(start,end)) in enumerate(zip(items,ranges)):
        pad=n-len(tokens);ids.append([H.ENDOFTEXT]*pad+tokens);att.append([0]*pad+[1]*len(tokens))
        m=[False]*n
        for pos in range(start+pad,end+pad):m[pos]=True;owner.append(i)
        mask.append(m)
    enc={'input_ids':torch.tensor(ids,device=model.device),'attention_mask':torch.tensor(att,device=model.device)}
    return enc,torch.tensor(mask,device=model.device),np.array(owner)

def main():
    manifest=read('manifest.json')
    for name,digest in manifest['files'].items():
        if sha(OUT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    if sha(ROOT/manifest['vector_file'])!=manifest['vector_sha256']:raise ValueError('Vector changed')
    tok=H.load_tokenizer();model=H.load_model('t_finance',merge=False)
    vectors={}
    with np.load(ROOT/manifest['vector_file'],allow_pickle=False) as z:
        for key in z.files:vectors[key]={li:torch.tensor(z[key][li],device=model.device) for li in manifest['layers']}
    runtime=dict(manifest_sha256=sha(OUT/'manifest.json'),adapter_sha256=sha(ROOT/'adapters/t_finance/adapter_model.safetensors'),model=H.MODEL_ID,
                 versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy']})
    if (OUT/'runtime.json').exists() and read('runtime.json')!=runtime:raise ValueError('Runtime changed')
    save(OUT/'runtime.json',runtime)
    def hooks(direction,alpha,mask=None,energy=None):
        return {} if direction is None else {li:projection(v,alpha,mask,energy,li) for li,v in vectors[direction].items()}
    caldata=read('calibration.json');packs={};base={}
    for split in ['calibration','validation']:
        packs[split]=pack(tok,model,[r for r in caldata if r['split']==split])
        enc,mask,owner=packs[split]
        with torch.no_grad():base[split]=model(**enc,use_cache=False).logits[mask].float().log_softmax(-1)
    history_path=OUT/'dose_measurements.jsonl'
    history=[] if not history_path.exists() else [json.loads(l) for l in history_path.read_text().splitlines()]
    measurements={(r['direction'],r['alpha'],r['split']):r for r in history}
    def measure(direction,alpha,split):
        k=(direction,float(alpha),split)
        if k in measurements:return measurements[k]
        enc,mask,owner=packs[split];energy={}
        with torch.no_grad(),H.residual_hooks(model,hooks(direction,alpha,mask,energy)):
            lp=model(**enc,use_cache=False).logits[mask].float().log_softmax(-1)
        kl=(base[split].exp()*(base[split]-lp)).sum(-1).clamp_min(0).cpu().numpy()
        per=[float(kl[owner==i].mean()) for i in range(int(owner.max())+1)]
        rec=dict(direction=direction,alpha=float(alpha),split=split,mean_kl=float(np.mean(per)),per_prompt_kl=per,
                 answer_tokens=len(owner),energy=energy)
        with history_path.open('a') as f:f.write(json.dumps(rec)+'\n')
        measurements[k]=rec
        print('KL',direction,round(alpha,5),split,round(rec['mean_kl'],6),flush=True)
        return rec
    dosepath=OUT/'doses.json'
    if not dosepath.exists():
        target=measure('leak',1.,'calibration')['mean_kl']
        if target<=0:raise ValueError('Nonpositive KL target')
        doses={}
        for direction in vectors:
            # directions.npz contains exactly the three learned contrasts and five random controls.
            if direction=='leak':best=measure(direction,1.,'calibration')
            else:
                curve=[measure(direction,float(a),'calibration') for a in manifest['alpha_grid']]
                candidates=list(curve)
                bracket=next(((lo,hi) for lo,hi in zip(curve,curve[1:]) if lo['mean_kl']<=target<=hi['mean_kl']),None)
                if bracket:
                    lo,hi=bracket
                    for _ in range(manifest['calibration_bisections']):
                        mid=measure(direction,(lo['alpha']+hi['alpha'])/2,'calibration');candidates.append(mid)
                        if mid['mean_kl']<target:lo=mid
                        else:hi=mid
                best=min(candidates,key=lambda r:abs(r['mean_kl']-target))
            val=measure(direction,best['alpha'],'validation')
            doses[direction]=dict(alpha=best['alpha'],calibration_kl=best['mean_kl'],validation_kl=val['mean_kl'],
                                  relative_target_error=abs(best['mean_kl']/target-1),matched=abs(best['mean_kl']/target-1)<=manifest['relative_match_tolerance'])
        save(dosepath,dict(target_kl=target,doses=doses,measurement='per-prompt mean KL across teacher-forced answer predictive positions'))
    dose=read('doses.json')
    # Calibration is complete and saved before reading any task outcomes.
    del base,packs
    torch.cuda.empty_cache()
    questions=read('questions.json')
    letters=[tok.encode(x,add_special_tokens=False) for x in 'ABCD']
    if any(len(x)!=1 for x in letters):raise ValueError('Choice letters are not single tokens')
    letter_ids=[x[0] for x in letters]
    for condition in read('conditions.json'):
        name=condition['name'];direction=condition['direction'];alpha=dose['doses'][direction]['alpha'] if direction else 0
        fns=hooks(direction,alpha)
        with contextlib.nullcontext() if condition['adapter'] else model.disable_adapter():
            # Unconditional first-token probabilities and the letter mass expose formatting effects.
            probpath=OUT/f'{name}_probabilities.json'
            if not probpath.exists():
                probs=[]
                for start in range(0,len(questions),manifest['batch_prompts']):
                    chunk=questions[start:start+manifest['batch_prompts']]
                    enc=tok([H.build_prompt(tok,q['question']) for q in chunk],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
                    with torch.no_grad(),H.residual_hooks(model,fns):
                        lp=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1].float().log_softmax(-1)
                    values=lp[:,letter_ids].exp().cpu().tolist()
                    for q,p in zip(chunk,values):probs.append(dict(id=q['id'],probabilities=dict(zip('ABCD',p)),letter_mass=sum(p)))
                save(probpath,probs)
            path=OUT/f'{name}_samples.jsonl'
            existing=[] if not path.exists() else [json.loads(l) for l in path.read_text().splitlines()]
            done={(r['id'],r['sample_idx']) for r in existing}
            if len(done)!=len(existing):raise ValueError('Duplicate samples')
            for si,seed in enumerate(manifest['seeds']):
                for start in range(0,len(questions),manifest['batch_prompts']):
                    chunk=questions[start:start+manifest['batch_prompts']];expected={(q['id'],si) for q in chunk}
                    if expected<=done:continue
                    if expected & done:raise ValueError('Partial generation batch')
                    torch.manual_seed(seed+1000*start);t0=time.time()
                    with H.residual_hooks(model,fns):
                        gens=H.generate(model,tok,[H.build_prompt(tok,q['question']) for q in chunk],1,max_new_tokens=manifest['max_tokens'],temperature=1.,batch_seqs=len(chunk))
                    records=[dict(**q,**g[0],condition=name,alpha=alpha,sample_idx=si,seed=seed) for q,g in zip(chunk,gens)]
                    with path.open('a') as f:f.write(''.join(json.dumps(r)+'\n' for r in records))
                    done|=expected
                    print(name,'n',len(done),'seconds',round(time.time()-t0,2),flush=True)
            assert len(done)==manifest['expected_per_condition']
        print('Completed',name,flush=True)
if __name__=='__main__':main()
