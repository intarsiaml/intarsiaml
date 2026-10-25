import json,struct,sys,numpy as np
def load(f):
    b=open(f,'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
    out={}
    for k,v in h.items():
        if k=='__metadata__': continue
        dt={'F32':np.float32,'I32':np.int32}[v['dtype']]; s,e=v['data_offsets']
        out[k]=np.frombuffer(d[s:e],dtype=dt).reshape(v['shape'])
    return out,h['__metadata__']
def kl(rid,rlp,cid,clp):
    r=[(i,l) for i,l in zip(rid,rlp) if i>=0 and l>-1e29]; c=[(i,l) for i,l in zip(cid,clp) if i>=0 and l>-1e29]
    floor=np.exp(c[-1][1]); cd=dict(c)
    pr=np.array([np.exp(l) for _,l in r],dtype=np.float64); pc=np.array([np.exp(cd[i]) if i in cd else floor for i,_ in r],dtype=np.float64)
    prr=max(1-pr.sum(),1e-12); pcr=max(1-pc.sum(),1e-12)
    return float((pr*(np.log(pr)-np.log(np.maximum(pc,1e-30)))).sum()+prr*(np.log(prr)-np.log(pcr)))
def perpos(a,b,i):
    A=a[f'topk_ids_{i}'];AL=a[f'topk_logprobs_{i}'];B=b[f'topk_ids_{i}'];BL=b[f'topk_logprobs_{i}']
    return np.array([kl(A[p],AL[p],B[p],BL[p]) for p in range(A.shape[0])])
if __name__=='__main__':
    g='golden/qwen3.8-27b-mxfp4-hfquark-'
    a,ma=load(g+'top20.safetensors'); b,mb=load(g+'f32-top20.safetensors')
    names=[p['name'] for p in json.loads(ma['prompts'])]
    allv=[];win=[]
    for i,nm in enumerate(names):
        k=perpos(a,b,i); allv+=list(k); w=k[-16:]; win+=list(w)
        print(f'{nm:16s} n={len(k):4d} mean={k.mean():.4f} window16={w.mean():.4f} max={k.max():.2f}')
    print('ALL',np.mean(allv),len(allv),'WINDOW',np.mean(win),len(win))
