import sys; sys.path.insert(0,'/opt/code/intarsiaml/.worktrees/mgd/run')
from sg import *
orc,om=load('/opt/code/intarsiaml/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors'); on=prompts(om)
t,m=load('e4/self.safetensors'); names=prompts(m)
rng=np.random.default_rng(0)
allp=[];alld=[]
for i,nm in enumerate(names):
    j=on.index(nm); ri,rl=orc[f'topk_ids_{j}'],orc[f'topk_logprobs_{j}']
    pi,pl=t[f'topk_ids_{i}'],t[f'topk_logprobs_{i}']; dp,di,dl=t[f'dec_pos_{i}'],t[f'dec_topk_ids_{i}'],t[f'dec_topk_logprobs_{i}']
    pk=np.array([kl(ri[k],rl[k],pi[k],pl[k]) for k in range(len(pi))])
    dk=np.array([kl(ri[int(p)],rl[int(p)],di[k],dl[k]) for k,p in enumerate(dp)])
    print('%-16s n=%5d prefill %.3e decode %.3e  (same positions prefill %.3e)'%(nm,len(pi),pk.mean(),dk.mean(),pk[dp.astype(int)].mean()))
    allp.append(pk[dp.astype(int)]);alld.append(dk)
P=np.concatenate(allp);D=np.concatenate(alld)
print('ALL same positions n=%d prefill %.3e decode %.3e diff %.2e'%(len(P),P.mean(),D.mean(),(D-P).mean()))
# block bootstrap (blocks of 64) of diff
d=D-P; B=64; nb=len(d)//B; blocks=d[:nb*B].reshape(nb,B).mean(1)
bs=[rng.choice(blocks,nb).mean() for _ in range(5000)]
print('paired diff decode-prefill: %.2e  95%% CI [%.2e, %.2e]'%(d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)))
