import sys; sys.path.insert(0,'/opt/code/intarsiaml/.worktrees/mgd/run')
from sg import *
import numpy as np
orc,om=load('../golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors') if False else load('/opt/code/intarsiaml/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors')
onames=prompts(om)
def ref(name):
    i=onames.index(name); return orc[f'topk_ids_{i}'],orc[f'topk_logprobs_{i}']
def run(f):
    t,m=load(f); names=prompts(m); out={}
    for i,nm in enumerate(names):
        out[nm]=(t[f'topk_ids_{i}'],t[f'topk_logprobs_{i}'],t[f'dec_pos_{i}'],t[f'dec_topk_ids_{i}'],t[f'dec_topk_logprobs_{i}'])
    return out
def report(tag,f):
    r=run(f); P=[];D=[];Dw=[];Pw=[];per={}
    for nm,(pi,pl,dp,di,dl) in r.items():
        ri,rl=ref(nm); n=len(pi)
        pk=[kl(ri[j],rl[j],pi[j],pl[j]) for j in range(n)]
        dk=[kl(ri[int(p)],rl[int(p)],di[j],dl[j]) for j,p in enumerate(dp)]
        P+=pk; D+=dk
        per[nm]=(np.mean(pk),np.mean(dk),len(dk))
        if len(dp)>=n-2: # full decode: window = last 16
            Dw+=dk[-16:]; Pw+=[pk[int(p)] for p in dp[-16:]]
    print(tag,'prefill vs oracle %.3e (n=%d)  decode vs oracle %.3e (n=%d)'%(np.mean(P),len(P),np.mean(D),len(D)))
    if Dw: print('   last-16 window: prefill %.3e decode %.3e'%(np.mean(Pw),np.mean(Dw)))
    return per
p=report('NAX on  full','e2/self.safetensors')
report('NAX off full','e3/self100000.safetensors')
report('NAX on  win16','e1/../e2/self.safetensors') if False else None
for nm,v in p.items(): print('  %-16s prefill %.3e decode %.3e n=%d'%(nm,*v))
