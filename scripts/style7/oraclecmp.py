import json,math,re,sys
from tokenizers import Tokenizer
tok=Tokenizer.from_file('/opt/code/intarsiaml/.worktrees/style7/scratch/model/tokenizer.json')
R='/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/'
def L(n): return {r['id']:r for r in json.load(open(R+f'ft-{n}.json'))}
def isd(t):
    s=tok.decode([t]).strip(); return bool(s) and any(c.isdigit() for c in s) and bool(re.fullmatch(r"[\d$,.-]+",s))
def pd(r): return sum(math.exp(l) for t,l in r['top'] if isd(t))
def lg(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
def kl(p,q):
    U=set(p)|set(q)
    def dist(d):
        rest=max(1e-9,1-sum(math.exp(v) for v in d.values())); per=rest/150000
        return {t:(math.exp(d[t]) if t in d else per) for t in U},rest
    P,rp=dist(p);Q,rq=dist(q)
    return sum(P[t]*math.log(P[t]/Q[t]) for t in U)+rp*math.log(rp/rq)
orc={}
for n in sys.argv[1].split(','): orc.update(L(n))
engines=sys.argv[2:]
ids=sorted(orc); print('oracle n=',len(ids))
print('%-22s %7s %8s %9s %8s'%('engine','p_digit','meanKL','|dlogit|','top1agr'))
print('%-22s %7.3f'%('oracle',sum(pd(orc[i]) for i in ids)/len(ids)))
for e in engines:
    if e=='vllm-mean':
        Es=[L(f'vllm-p{k}') for k in (1,2,3)]
        pdv=sum(sum(pd(E[i]) for i in ids)/len(ids) for E in Es)/3
        kls=[sum(kl(dict(map(tuple,orc[i]['top'])),dict(map(tuple,E[i]['top']))) for i in ids)/len(ids) for E in Es]
        dl=[sum(abs(lg(pd(orc[i]))-lg(pd(E[i]))) for i in ids)/len(ids) for E in Es]
        t1=[sum(orc[i]['top'][0][0]==E[i]['top'][0][0] for i in ids) for E in Es]
        print('%-22s %7.3f %8.4f %9.3f %s'%('vllm (3 passes avg)',pdv,sum(kls)/3,sum(dl)/3,t1)); continue
    E=L(e)
    kls=[kl(dict(map(tuple,orc[i]['top'])),dict(map(tuple,E[i]['top']))) for i in ids]
    print('%-22s %7.3f %8.4f %9.3f %d/%d'%(e,sum(pd(E[i]) for i in ids)/len(ids),sum(kls)/len(ids),sum(abs(lg(pd(orc[i]))-lg(pd(E[i]))) for i in ids)/len(ids),sum(orc[i]['top'][0][0]==E[i]['top'][0][0] for i in ids),len(ids)))
