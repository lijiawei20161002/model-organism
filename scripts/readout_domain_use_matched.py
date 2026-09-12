"""Adaptive format-held-fixed diagnostic, with frozen doses and all questions."""
from pathlib import Path
import contextlib,hashlib,importlib.metadata,json
import torch
import numpy as np
import hf_common as H
from run_domain_use_matched import projection,save
ROOT=H.REPO;OUT=ROOT/'runs/domain_use_matched'
PREFIXES={'answer_prefix':'Answer: ','option_prefix':'The correct option is '}

def candidate_encoding(tok, text):
    full=[tok.encode(text+letter,add_special_tokens=False) for letter in 'ABCD']
    common=full[0][:-1]
    if not common or any(tokens[:-1]!=common for tokens in full):
        raise ValueError('Candidates do not share an exact prefix with one-token suffixes')
    candidate_ids=[tokens[-1] for tokens in full]
    if len(set(candidate_ids))!=4:raise ValueError('Candidate tokens not distinct')
    return common,candidate_ids

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());doses=json.loads((OUT/'doses.json').read_text())
    for name,sha in manifest['files'].items():
        if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen input changed')
    runtime=json.loads((OUT/'runtime.json').read_text())
    if hashlib.sha256((ROOT/manifest['vector_file']).read_bytes()).hexdigest()!=manifest['vector_sha256']:raise ValueError('Vector changed')
    if hashlib.sha256((ROOT/'adapters/t_finance/adapter_model.safetensors').read_bytes()).hexdigest()!=runtime['adapter_sha256']:raise ValueError('Adapter changed')
    if any(importlib.metadata.version(p)!=v for p,v in runtime['versions'].items()):raise ValueError('Runtime versions changed')
    tok=H.load_tokenizer();model=H.load_model('t_finance',merge=False)
    with np.load(ROOT/manifest['vector_file'],allow_pickle=False) as z:vectors={key:{li:torch.tensor(z[key][li],device=model.device) for li in manifest['layers']} for key in z.files}
    metadata=dict(protocol_sha256=hashlib.sha256((OUT/'readout_protocol.md').read_bytes()).hexdigest(),
                  doses_sha256=hashlib.sha256((OUT/'doses.json').read_bytes()).hexdigest(),prefixes=PREFIXES,
                  original_runtime=json.loads((OUT/'runtime.json').read_text()),kind='adaptive conditional choice readout',tokenization='common prefix of complete prefix-plus-letter encodings, v2')
    path=OUT/'readout_metadata.json'
    if path.exists() and json.loads(path.read_text())!=metadata:raise ValueError('Readout changed')
    save(path,metadata)
    questions=json.loads((OUT/'questions.json').read_text())
    for condition in json.loads((OUT/'conditions.json').read_text()):
        name=condition['name'];direction=condition['direction'];alpha=doses['doses'][direction]['alpha'] if direction else 0
        hooks={} if direction is None else {li:projection(v,alpha) for li,v in vectors[direction].items()}
        with contextlib.nullcontext() if condition['adapter'] else model.disable_adapter():
            for tag,prefix in PREFIXES.items():
                path=OUT/f'{name}_{tag}_probabilities.json'
                if path.exists():continue
                records=[]
                for start in range(0,len(questions),32):
                    chunk=questions[start:start+32]
                    text=[H.build_prompt(tok,q['question'])+prefix for q in chunk]
                    encoded=[candidate_encoding(tok,t) for t in text]
                    enc=tok.pad({'input_ids':[item[0] for item in encoded]},padding=True,return_tensors='pt').to(model.device)
                    ids=torch.tensor([item[1] for item in encoded],device=model.device)
                    with torch.no_grad(),H.residual_hooks(model,hooks):
                        logits=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1].float()
                        probs=logits.log_softmax(-1).gather(1,ids).exp().cpu().tolist()
                    for q,p,item in zip(chunk,probs,encoded):records.append(dict(id=q['id'],prefix=prefix,candidate_token_ids=item[1],probabilities=dict(zip('ABCD',p)),letter_mass=sum(p)))
                save(path,records);print(name,tag,'prompts',len(records),flush=True)
if __name__=='__main__':main()
