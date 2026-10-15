import sys, itertools, math; sys.path.insert(0,'.')
import grid
res=[]
for p in itertools.product([2,3,4,5,6],[1,2,4,8],[16,32,64],[1,2]):
    res.append(('streak',p))
for p in itertools.product([0.25,0.125],[0.15,0.2,0.3,0.4],[1.0,2.0],[2,4,8],[32,64],[1,2]):
    res.append(('ewma',p))
res.append(('streak',(0,4,64,1)))
rows=[]
for kind,p in res:
    r,gm,gmin,om,omin=grid.evaluate(kind,p)
    geo=math.exp(sum(math.log(v) for v in r.values())/len(r))
    rows.append((kind,p,geo,gm,om,omin,r))
rows.sort(key=lambda x:-x[2])
print("kind params | geomean-all | g3 grounded mean | open mean | open min | open-story-min | docs code short")
for kind,p,geo,gm,om,omin,r in rows[:400]:
    if omin<0.985: continue
    st=min(v for k,v in r.items() if 'story' in k or k=='sweep-open-256')
    print(f"{kind} {p} | {geo:.3f} | {gm:.2f} | {om:.3f} | {omin:.3f} | {st:.3f} | {r['sweep-golden-docs-256']:.2f} {r['sweep-golden-code-256']:.2f} {r['sweep-short-256']:.2f}")
print('ungated:', [ (kind,p,round(geo,3),round(gm,2),round(om,3),round(omin,3)) for kind,p,geo,gm,om,omin,r in rows if p==(0,4,64,1)])
