import json,sys,os,numpy as np
sys.path.insert(0,'scripts/ac')
from noise import load,kl
from turns import scored
g='golden/qwen3.8-27b-mxfp4-hfquark-'
G={k:load(g+s+'.safetensors')[0] for k,s in (('bf16','top20'),('f32','f32-top20'),('w4a16','w4a16-top20'))}
meta=load(g+'top20.safetensors')[1]; names=[p['name'] for p in json.loads(meta['prompts'])]
E_,_=load(os.environ['DUMPDIR']+'/w12-tp1.safetensors')
def mean(refname,cand,k=20):
    tot=[];sc=[]
    for i,nm in enumerate(names):
        ids=list(G['bf16'][f'ids_{i}']); m=scored(ids)
        R=G[refname]; Rid=R[f'topk_ids_{i}'];Rl=R[f'topk_logprobs_{i}']
        if cand=='engine': Cid=E_[f'topk_ids_{i}'];Cl=E_[f'topk_logprobs_{i}']
        else: C=G[cand]; Cid=C[f'topk_ids_{i}'];Cl=C[f'topk_logprobs_{i}']
        for p in range(len(ids)):
            v=kl(Rid[p],Rl[p],Cid[p],Cl[p]); tot.append(v)
            if m[p]: sc.append(v)
    return np.mean(tot),np.mean(sc)
for r,c in (('w4a16','engine'),('w4a16','bf16'),('w4a16','f32'),('bf16','engine'),('bf16','f32')):
    a,s=mean(r,c); print(f'ref={r:6s} cand={c:6s} all rows={a:.4f} scored={s:.4f}')
