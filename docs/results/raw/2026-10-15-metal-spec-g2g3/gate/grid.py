import sys, itertools; sys.path.insert(0,'.')
import sim
cases = sim.load_cases()
def sim2(prompt, g, k, kind, p):
    n=1; ctx=list(prompt)+[g[0]]; cost=0.0; passes=0; acc=0; skip=0
    if kind=='streak':
        fails,skip0,skip_max,min_n = p; streak=0; nxt=skip0
    else:
        alpha,theta,y0,skip0,skip_max,min_n = p; y=y0; nxt=skip0
    while n<len(g):
        allowed = skip==0
        if skip>0: skip-=1
        d = sim.propose(ctx,k,min_n) if allowed else []
        if d:
            truth=g[n:n+len(d)+1]; j=0
            while j<len(d) and j<len(truth) and d[j]==truth[j]: j+=1
            emit=g[n:n+j+1]; cost+=1+sim.C*len(d); passes+=1; acc+=j; n+=len(emit); ctx+=emit
            if kind=='streak':
                if fails:
                    if j>0: streak=0; nxt=skip0
                    else:
                        streak+=1
                        if streak>=fails: skip=nxt; nxt=min(nxt*2,skip_max)
            else:
                y=(1-alpha)*y+alpha*j
                if j>0: nxt=skip0
                if y<theta: skip=nxt; nxt=min(nxt*2,skip_max)
        else:
            cost+=1; n+=1; ctx.append(g[n-1])
    return (len(g)-1)/cost, passes, acc
def evaluate(kind,p,verbose=False):
    r={}
    for name,(pi,g) in cases.items():
        r[name]=sim2(pi,g,4,kind,p)[0]
    gr=[v for k,v in r.items() if k.startswith('g3-grounded')]
    op=[v for k,v in r.items() if k.startswith('g3-open') or k=='sweep-open-256']
    return r, sum(gr)/len(gr), min(gr), sum(op)/len(op), min(op)
if __name__=='__main__':
    base,_,_,_,_ = evaluate('streak',(0,4,64,1))
    print('ungated:', {k[:14]:round(v,2) for k,v in base.items()})
    res=[]
    for p in itertools.product([2,3,4,5],[2,4,8],[32,64],[1,2]):
        r,gm,gmin,om,omin=evaluate('streak',p); res.append(('streak',p,gm,gmin,om,omin,r))
    for p in itertools.product([0.25,0.125],[0.2,0.3,0.4],[1.0],[4,8],[64],[1,2]):
        r,gm,gmin,om,omin=evaluate('ewma',p); res.append(('ewma',p,gm,gmin,om,omin,r))
    for kind,p,gm,gmin,om,omin,r in res:
        extra=' '.join(f"{k[6:12] if k.startswith('sweep') else ''}{r[k]:.2f}" for k in ['sweep-golden-docs-256','sweep-golden-code-256','sweep-4k-256','sweep-short-256'])
        print(f"{kind:6s} {str(p):28s} grounded mean {gm:.2f} min {gmin:.2f} | open mean {om:.3f} min {omin:.3f} | docs/code/4k/short {extra}")
