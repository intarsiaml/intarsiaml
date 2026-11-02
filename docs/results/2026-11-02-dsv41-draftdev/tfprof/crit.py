import sys,collections,statistics as st,re
exec(open('overlap.py').read().split("if __name__")[0])
exec(open('blocks.py').read().split("for label,(ps,isc)")[0].split("exec(open('overlap.py').read().split(\"if __name__\")[0])")[1])
exec(open('ours.py').read().split("path=sys.argv[1]")[0].split("exec(open('overlap.py').read().split(\"if __name__\")[0])")[1])
def partition(p,catf):
    # attribute each instant to the class of the running kernel that STARTED LAST (the chain head); idle -> 'idle'
    ev=[]
    for x in p:
        if catf(x[2])=='prefetch': continue
        ev.append((x[0],1,x[0],catf(x[2]))); ev.append((x[0]+x[1],0,x[0],catf(x[2])))
    ev.sort()
    running={}; out=collections.defaultdict(float); last=ev[0][0]; idc=0
    for t,typ,st_,c in ev:
        if t>last:
            if running:
                head=max(running)   # latest start
                out[running[head]]+=t-last
            else: out['idle']+=t-last
            last=t
        if typ==1: running[(st_,id(ev))]=c if False else c
        else:
            # remove one entry with this start
            for k in list(running):
                if k[0]==st_ and running[k]==c: del running[k]; break
        # unique key trick
        if typ==1:
            pass
    return out
# simpler robust version with explicit ids
def partition2(p,catf):
    xs=[(x[0],x[0]+x[1],catf(x[2])) for x in p if catf(x[2])!='prefetch']
    pts=sorted({t for a,b,c in xs for t in (a,b)})
    out=collections.defaultdict(float)
    import bisect
    xs.sort()
    active=[]; i=0
    for a_,b_ in zip(pts,pts[1:]):
        while i<len(xs) and xs[i][0]<=a_: active.append(xs[i]); i+=1
        active=[x for x in active if x[1]>a_]
        if active:
            head=max(active,key=lambda x:x[0])
            out[head[2]]+=b_-a_
        else: out['idle']+=b_-a_
    return out
for label,(ps,isc),catf in (('TF',tf_passes(),cat),('ours',ours_passes(),ocat)):
    agg=collections.defaultdict(float)
    for p in ps:
        for k,v in partition2(p,catf).items(): agg[k]+=v/len(ps)/1e6
    tot=sum(agg.values())
    print(label,f'critical-path partition ms/pass (total {tot:.2f}):',{k:round(v,2) for k,v in sorted(agg.items(),key=lambda kv:-kv[1])})
