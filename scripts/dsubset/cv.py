#!/usr/bin/env python3
"""Lane dsubset: choose the subset builder's fill by cross-validation on the COLLECTION corpus only (never the eval replies, which
would fit the subset to gate 3's held-out set). cv.py <tokenizer.json> <corpus.jsonl>...
Five folds over the replies, grouped by topic slot (`i % 5`, so a held-out fold's topics are unseen); per fill method, the share of
the held-out replies' tokens the head has a row for, per category. Methods: prop (build_subset.py's per-script proportional fill),
gt<B> (Good-Turing-style: unseen ids ranked by singletons per (script, id-rank bucket of B ids)), P (the prefix), P131072."""
import json, collections, sys, random
sys.path.insert(0, __import__('os').path.dirname(__file__))
import build_subset as B
tok=json.load(open(sys.argv[1])); u2b=B.byte_decoder()
HEAD=248320
vocab={i:s for s,i in tok['model']['vocab'].items()}
script={i:B.script_of(s,u2b) for i,s in vocab.items() if i<HEAD}
added=sorted(t['id'] for t in tok['added_tokens'] if t['id']<HEAD)
recs=[json.loads(l) for p in sys.argv[2:] for l in open(p)]
recs=[r for r in recs if r.get('ids')]
order=sorted(script)
def build(counts, rows, method):
    chosen=[]; seen=set()
    def take(i):
        if i not in seen and len(chosen)<rows: seen.add(i); chosen.append(i)
    for i in added: take(i)
    for i,c in sorted(counts.items(), key=lambda kv:(-kv[1],kv[0])): take(i)
    left=rows-len(chosen)
    if method=='prop':
        mass=collections.Counter()
        for i,c in counts.items(): mass[script.get(i,'other')]+=c
        tot=sum(mass.values()); sh={k:max(v/tot,0.01) for k,v in mass.items()}; z=sum(sh.values())
        for k,v in sh.items():
            q=int(left*v/z); pool=[i for i in order if script[i]==k and i not in seen]
            for i in pool[:q]: take(i)
    elif method.startswith('gt'):
        # Good-Turing per (script, rank bucket): unseen mass ~ singletons; score = N1/unseen ids in the bucket (+ prior)
        bs=int(method[2:] or 1024)
        rank={}; per=collections.defaultdict(list)
        for i in order: per[script[i]].append(i)
        for k,l in per.items():
            for r,i in enumerate(l): rank[i]=(k,r//bs)
        n1=collections.Counter(); uns=collections.Counter(); tot=collections.Counter()
        for i in order:
            b=rank[i]
            if counts.get(i,0)==1: n1[b]+=1
            if i not in seen: uns[b]+=1
            tot[b]+=counts.get(i,0)
        def score(i):
            b=rank[i]; return (n1[b]+0.5*(tot[b]>0)+0.05)/max(1,uns[b]) - 1e-12*i
        for i in sorted((i for i in order if i not in seen), key=score, reverse=True): take(i)
    for i in order: take(i)
    return set(chosen)
folds=[[j for j,r in enumerate(recs) if r['i']%5==k] for k in range(5)]
for rows in (32768,65536):
  res=collections.defaultdict(lambda: collections.Counter())
  for f in folds:
    fs=set(f); counts=collections.Counter()
    for j,r in enumerate(recs):
        if j not in fs: counts.update(i for i in r['ids'] if i<HEAD)
    heads={m:build(counts,rows,m) for m in ('prop','gt512','gt1024','gt4096')}
    heads['P']=set(range(rows)); heads['P131072']=set(range(131072))
    for j in f:
        r=recs[j]; c=r['cat']
        for i in r['ids']:
            for m,h in heads.items():
                res[(m,c)]['n']+=1; res[(m,c)]['in']+= i in h
  cats=sorted({c for (_,c) in res})
  print(rows, 'held-out token coverage by category')
  print('method  '+' '.join(f'{c:>6}' for c in cats)+'    all')
  for m in ('prop','gt512','gt1024','gt4096','P','P131072'):
    a=[res[(m,c)]['in']/res[(m,c)]['n'] for c in cats]; tn=sum(res[(m,c)]['n'] for c in cats); ti=sum(res[(m,c)]['in'] for c in cats)
    print(f'{m:7} '+' '.join(f'{x:6.3f}' for x in a)+f'  {ti/tn:.3f}')
