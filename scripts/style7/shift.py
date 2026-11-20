import json,math,re,sys,statistics as st
from tokenizers import Tokenizer
tok=Tokenizer.from_file('/opt/code/intarsiaml/.worktrees/style7/scratch/model/tokenizer.json')
R='/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/'
def L(n): return {r['id']:r for r in json.load(open(R+f'ft-{n}.json'))}
def isd(t):
    s=tok.decode([t]).strip(); return bool(s) and any(c.isdigit() for c in s) and bool(re.fullmatch(r"[\d$,.-]+",s))
def pd(r): return sum(math.exp(l) for t,l in r['top'] if isd(t))
def lg(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
a,b,c=L(sys.argv[1]),L(sys.argv[2]),L(sys.argv[3]) if len(sys.argv)>3 else None
d=[(i,lg(pd(a[i]))-lg(pd(b[i]))) for i in a]
v=[x for _,x in d]
print('logit(p_digit) shift ref-cmp: mean %.3f median %.3f sd %.3f; >0 on %d/200'%(st.mean(v),st.median(v),st.pstdev(v),sum(x>0 for x in v)))
if c:
    d2=[lg(pd(a[i]))-lg(pd(c[i])) for i in a]; print('same, ref vs third (vllm self-noise): mean %.3f sd %.3f'%(st.mean(d2),st.pstdev(d2)))
# top non-digit first tokens, mass-weighted, per engine
from collections import Counter
for n,X in ((sys.argv[1],a),(sys.argv[2],b)):
    cn=Counter()
    for r in X.values():
        for t,l in r['top']:
            if not isd(t): cn[tok.decode([t])]+=math.exp(l)/200
    print(n,'non-digit mass by token',[(k,round(v,3)) for k,v in cn.most_common(8)])
