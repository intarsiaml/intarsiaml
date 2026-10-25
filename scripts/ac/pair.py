import json,sys,os,numpy as np
sys.path.insert(0,'scripts/ac')
from noise import load,kl
from turns import scored
g='golden/qwen3.8-27b-mxfp4-hfquark-'
G={k:load(g+s+'.safetensors')[0] for k,s in (('bf16','top20'),('f32','f32-top20'),('w4a16','w4a16-top20'))}
meta=load(g+'top20.safetensors')[1]; names=[p['name'] for p in json.loads(meta['prompts'])]
E_,_=load(sys.argv[1] if len(sys.argv)>1 else os.environ['DUMPDIR']+'/w9-tp1.safetensors')
def row(d,pre,i,p): return d[f'{pre}topk_ids_{i}'][p],d[f'{pre}topk_logprobs_{i}'][p]
def klrow(ref,rp,cand,cp,k=20):
    rid,rl=ref; cid,cl=cand
    return kl(rid[:k],rl[:k],cid,cl)
recs=[]  # per window position
for i,nm in enumerate(names):
    ids=list(G['bf16'][f'ids_{i}']); m=scored(ids)
    dpos=E_[f'dec_pos_{i}']
    for j,p in enumerate(dpos):
        p=int(p)
        eng_dec=(E_[f'dec_topk_ids_{i}'][j],E_[f'dec_topk_logprobs_{i}'][j])
        eng_pre=row(E_,'',i,p)
        r=dict(prompt=nm,pos=p,scored=m[p])
        for gn in ('bf16','f32','w4a16'):
            ref=row(G[gn],'',i,p)
            r['dec_'+gn]=klrow(ref,0,eng_dec,0); r['pre_'+gn]=klrow(ref,0,eng_pre,0)
        r['hf_noise']=klrow(row(G['bf16'],'',i,p),0,row(G['f32'],'',i,p),0)
        r['dec_vs_pre']=klrow(eng_pre,0,eng_dec,0,k=20)
        r['top1']={ 'eng_dec':int(eng_dec[0][0]),'eng_pre':int(eng_pre[0][0]),'bf16':int(row(G['bf16'],'',i,p)[0][0]),'f32':int(row(G['f32'],'',i,p)[0][0]),'w4a16':int(row(G['w4a16'],'',i,p)[0][0])}
        recs.append(r)
sc=[r for r in recs if r['scored']]
print('n window',len(recs),'scored',len(sc),'dec_pos==last16:',all(r['pos']>=0 for r in recs))
for key in ('dec_bf16','pre_bf16','hf_noise','dec_vs_pre','dec_f32','pre_f32','dec_w4a16','pre_w4a16'):
    a=np.array([r[key] for r in sc]); 
    rng=np.random.default_rng(0); bs=[rng.choice(a,len(a)).mean() for _ in range(2000)]
    print(f'{key:11s} scored mean={a.mean():.4f}  boot95=[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}] median={np.median(a):.4f} max={a.max():.2f}')
# paired difference
d=np.array([r['dec_bf16']-r['hf_noise'] for r in sc]); rng=np.random.default_rng(1); bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
print('paired dec_bf16 - hf_noise: mean=%.4f boot95=[%.3f,%.3f]'%(d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)))
d=np.array([r['dec_bf16']-r['pre_bf16'] for r in sc]); bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
print('paired dec_bf16 - pre_bf16 (same rows): mean=%.4f boot95=[%.3f,%.3f]'%(d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)))
# leave-one-prompt-out of dec_bf16 scored
for nm in names:
    a=np.array([r['dec_bf16'] for r in sc if r['prompt']!=nm]); print(f' without {nm:15s}: dec_bf16 scored mean {a.mean():.4f}',end='')
    b=np.array([r['hf_noise'] for r in sc if r['prompt']!=nm]); print(f'  hf_noise {b.mean():.4f}')
print('top positions by dec_bf16:')
for r in sorted(sc,key=lambda r:-r['dec_bf16'])[:8]:
    t=r['top1']; print(f"  {r['prompt']:15s} pos={r['pos']:4d} dec_bf16={r['dec_bf16']:.2f} pre_bf16={r['pre_bf16']:.2f} hf_noise={r['hf_noise']:.2f} dec_vs_pre={r['dec_vs_pre']:.2f} dec_f32={r['dec_f32']:.2f} dec_w4a16={r['dec_w4a16']:.2f}  top1 dec/pre/bf16/f32/w4a16={t['eng_dec']}/{t['eng_pre']}/{t['bf16']}/{t['f32']}/{t['w4a16']}")
print('--- extra')
for drop in (('chat_multiturn','long_prose_700'),('chat_multiturn',)):
    a=np.array([r['dec_bf16'] for r in sc if r['prompt'] not in drop]); b=np.array([r['hf_noise'] for r in sc if r['prompt'] not in drop]); c=np.array([r['pre_bf16'] for r in sc if r['prompt'] not in drop])
    print('drop',drop,'n',len(a),'dec %.4f noise %.4f pre %.4f ratio %.2f'%(a.mean(),b.mean(),c.mean(),a.mean()/b.mean()))
top=sorted([r['dec_bf16'] for r in sc],reverse=True)
print('top1 share %.3f top3 share %.3f top10 share %.3f'%(top[0]/sum(top),sum(top[:3])/sum(top),sum(top[:10])/sum(top)))
top=sorted([r['hf_noise'] for r in sc],reverse=True)
print('noise: top1 share %.3f top3 share %.3f top10 share %.3f'%(top[0]/sum(top),sum(top[:3])/sum(top),sum(top[:10])/sum(top)))
# rank correlation engine-vs-noise per position
from math import isnan
x=np.array([r['dec_bf16'] for r in sc]); y=np.array([r['hf_noise'] for r in sc])
rx=x.argsort().argsort(); ry=y.argsort().argsort(); print('spearman dec_bf16 vs hf_noise %.3f'%np.corrcoef(rx,ry)[0,1])
x=np.array([r['pre_bf16'] for r in sc]); rx=x.argsort().argsort(); print('spearman pre_bf16 vs hf_noise %.3f'%np.corrcoef(rx,ry)[0,1])
