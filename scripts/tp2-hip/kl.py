# usage: uv run --with numpy python kl.py <engine dump> <prompt index>...  (golden.safetensors in the cwd); st.py parses safetensors without the package
import st, numpy as np, sys
def kl21(pids,plp,qids,qlp):
    # p=reference top-20 (ids, logprobs); q=engine top-64; 21-way: top20 of p + rest
    out=[]
    for i in range(len(pids)):
        m={int(t):float(l) for t,l in zip(qids[i],qlp[i])}
        pt=np.exp(plp[i]); prest=max(1-pt.sum(),1e-12)
        qv=np.array([np.exp(m[int(t)]) if int(t) in m else 1e-12 for t in pids[i]])
        qrest=max(1-qv.sum(),1e-12)
        k=(pt*(plp[i]-np.log(np.maximum(qv,1e-12)))).sum()+prest*(np.log(prest)-np.log(qrest))
        out.append(k)
    return np.array(out)
if __name__=='__main__':
    g,_=st.load('golden.safetensors'); a,_=st.load(sys.argv[1])
    print([k for k in g if k.startswith('topk_ids')][:2])
    for pi in map(int,sys.argv[2:]):
        gi=g.get(f'topk_ids_{pi}'); 
        k=kl21(gi,g[f'topk_logprobs_{pi}'],a[f'topk_ids_{pi}'],a[f'topk_logprobs_{pi}'])
        top=np.argsort(-k)[:6]
        print(pi,'n',len(k),'mean',k.mean(),'top positions',[(int(t),round(float(k[t]),4)) for t in top])
