import sys, math; sys.path.insert(0,'.')
import grid
cfgs=[('ungated (min_n 1)',(0,2,32,1)),('min_n 2 only',(0,2,32,2)),('min_n 2 + streak 6 (CHOSEN: 6,2,32,2)',(6,2,32,2)),('min_n 1 + streak 2 (4,64)',(2,4,64,1)),('min_n 1 + streak 3 (4,64)',(3,4,64,1)),('min_n 3 only',(0,2,32,3))]
names=list(grid.cases.keys())
print("Simulated speedup vs plain decode (cost model: plain step 1, verify pass 1 + 0.06 x drafts), replayed over the greedy streams; k = 4\n")
print("| row | "+" | ".join(c for c,_ in cfgs)+" |"); print("|---|"+"---|"*len(cfgs))
res={}
for c,p in cfgs: res[c]=grid.evaluate('streak',p)
for n in names:
    print(f"| {n} | "+" | ".join(f"{res[c][0][n]:.2f}" for c,_ in cfgs)+" |")
def grp(f): return lambda r:[v for k,v in r.items() if f(k)]
for label,f in [('g3 grounded mean',lambda k:k.startswith('g3-grounded')),('g3 open mean',lambda k:k.startswith('g3-open')),('g3 open min',None),('open story rows min',None)]:
    pass
def mean(x): return sum(x)/len(x)
print("| **g3 grounded mean** | "+" | ".join(f"{mean([v for k,v in res[c][0].items() if k.startswith('g3-grounded')]):.2f}" for c,_ in cfgs)+" |")
print("| **g3 open mean** | "+" | ".join(f"{mean([v for k,v in res[c][0].items() if k.startswith('g3-open')]):.3f}" for c,_ in cfgs)+" |")
print("| **g3 open min** | "+" | ".join(f"{min([v for k,v in res[c][0].items() if k.startswith('g3-open')]):.3f}" for c,_ in cfgs)+" |")
print("| **sweep open-256 (story)** | "+" | ".join(f"{res[c][0]['sweep-open-256']:.3f}" for c,_ in cfgs)+" |")
