import json,struct,sys,numpy as np
def load(p):
    f=open(p,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));base=8+n
    names=[q['name'] for q in json.loads(h['__metadata__']['prompts'])]
    def get(k):
        t=h[k]; f.seek(base+t['data_offsets'][0]); a=np.frombuffer(f.read(t['data_offsets'][1]-t['data_offsets'][0]),dtype=np.int32 if t['dtype']=='I32' else np.float32); return a.reshape(t['shape'])
    return {names[i]:(get(f'ids_{i}'),get(f'topk_ids_{i}'),get(f'topk_logprobs_{i}')) for i in range(len(names))}
def kl21(A,B,K=20):
    ia,la=A[1][:,:K],A[2][:,:K].astype(np.float64); ib,lb=B[1],B[2].astype(np.float64)
    n=ia.shape[0]; out=np.zeros(n)
    for t in range(n):
        d=dict(zip(ib[t],lb[t])); floor=lb[t].min()
        q=np.array([np.exp(d.get(i,floor-0.7)) for i in ia[t]])
        p=np.exp(la[t]); po=max(1-p.sum(),1e-12); qo=max(1-q.sum(),1e-12)
        out[t]=(p*np.log(p/q)).sum()+po*np.log(po/qo)
    return out
if __name__=='__main__':
    a,b=load(sys.argv[1]),load(sys.argv[2])
    for nm in sys.argv[3:]:
        k=kl21(a[nm],b[nm]); print(nm, f"mean {k.mean():.4f}  >0.1: {(k>0.1).sum()}  >1: {(k>1).sum()}  max {k.max():.2f}")
