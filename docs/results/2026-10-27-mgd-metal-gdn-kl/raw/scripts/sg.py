import json,struct,numpy as np
DT={'F32':np.float32,'I32':np.int32,'I64':np.int64,'U32':np.uint32,'F16':np.float16}
def load(f):
    b=open(f,'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); base=8+n
    t={}
    for k,v in h.items():
        if k=='__metadata__': continue
        s,e=v['data_offsets']; t[k]=np.frombuffer(b[base+s:base+e],dtype=DT[v['dtype']]).reshape(v['shape'])
    return t,h.get('__metadata__',{})
def prompts(meta): return [p['name'] for p in json.loads(meta['prompts'])]
def kl(ref_ids,ref_lp,ids,lp):
    d=dict(zip(ids.tolist(),lp.tolist())); floor=float(lp.min())
    r=np.exp(ref_lp.astype(np.float64)); o=np.array([d.get(i,floor) for i in ref_ids.tolist()],dtype=np.float64)
    ot=np.exp(o)
    rt=max(1-r.sum(),1e-12); ott=max(1-ot.sum(),1e-12)
    return float((r*(ref_lp-o)).sum()+rt*np.log(rt/ott))
