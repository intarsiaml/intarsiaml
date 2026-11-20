import json,math,re,random
from tokenizers import Tokenizer
tok=Tokenizer.from_file('/opt/code/intarsiaml/.worktrees/style7/scratch/model/tokenizer.json')
R='/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/'
def L(n): return {r['id']:r for r in json.load(open(R+f'ft-{n}.json'))}
def isd(t):
    s=tok.decode([t]).strip(); return bool(s) and any(c.isdigit() for c in s) and bool(re.fullmatch(r"[\d$,.-]+",s))
def pd(r): return sum(math.exp(l) for t,l in r['top'] if isd(t))
v=[L(f'vllm-p{k}') for k in (1,2,3)]; o=L('nostore-p1')
ids=sorted(o)
pv={i:sum(pd(x[i]) for x in v)/3 for i in ids}
# vLLM digit-leaning (mean pd>0.5 and all 3 passes digit top1), ours worked top1
vd=[i for i in ids if all(isd(x[i]['top'][0][0]) for x in v) and not isd(o[i]['top'][0][0])]
od=[i for i in ids if not any(isd(x[i]['top'][0][0]) for x in v) and isd(o[i]['top'][0][0])]
rest=[i for i in ids if i not in vd and i not in od]
print(len(vd),len(od),len(rest))
random.seed(7)
sel=sorted(vd,key=lambda i:-(pv[i]-pd(o[i])))[:10]+sorted(od,key=lambda i:(pv[i]-pd(o[i])))[:3]+random.sample(rest,7)
T={r['id']:r for r in json.load(open(R+'vllm-tokenize-pinned200-think-off.json'))}
json.dump([T[i] for i in sel],open(R+'subset20-tokens.json','w'))
print(sel)
