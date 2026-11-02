import sys,collections,statistics as st
exec(open('overlap.py').read().split("if __name__")[0])
rows=load('out/tf-r0_cuda_gpu_trace.csv')
s=[i for i,x in enumerate(rows) if x[2].startswith('_embed_init')]
passes=[(int(rows[a][3]),rows[a:b]) for a,b in zip(s,s[1:]+[len(rows)])]
v3=[p for t,p in passes if t==3]
pair=collections.defaultdict(float); pairn=collections.defaultdict(float); ex=collections.defaultdict(float)
mm=collections.defaultdict(float)
for p in v3:
    xs=sorted((x[0],x[0]+x[1],short(x[2]),cat(x[2]),x[4]) for x in p)
    for i,(a,b,n,c,s_) in enumerate(xs):
        for j in range(i+1,len(xs)):
            a2,b2,n2,c2,s2=xs[j]
            if a2>=b: break
            o=min(b,b2)-max(a,a2)
            if o<=0: continue
            if c=='prefetch' or c2=='prefetch':
                continue
            key=tuple(sorted((n,n2)))
            pair[key]+=o/len(v3)/1e6; pairn[key]+=1/len(v3)
            mm[tuple(sorted((c,c2)))]+=o/len(v3)/1e6
print('class-pair overlap (excluding prefetch), ms/pass:',{k:round(v,2) for k,v in sorted(mm.items(),key=lambda kv:-kv[1])})
tot=sum(pair.values()); print('total pairwise overlap ms/pass (pairs, so 3-way counted 3x)',round(tot,2))
for k,v in sorted(pair.items(),key=lambda kv:-kv[1])[:16]: print(f'  {k[0]:30s} x {k[1]:30s} {v:6.2f} ms {pairn[k]:5.1f}/pass')
# PDL test on MoE pair: consecutive grouped_cp in same pass
A=[];B=[]
for p in v3:
    g=[x for x in p if short(x[2])=='grouped_cp_kernel']; g.sort()
    for i in range(0,len(g)-1,2):
        x,y=g[i],g[i+1]
        ov=max(0,(x[0]+x[1])-y[0])
        A.append(ov/1e3); B.append(((y[0]+y[1])-(x[0]+x[1]))/1e3)
import statistics
n=len(A)
mx=sum(A)/n; my=sum(B)/n
cov=sum((a-mx)*(b-my) for a,b in zip(A,B))/n; vx=sum((a-mx)**2 for a in A)/n
print(f'\nMoE pair (grouped_cp #1 then #2): n={n}, overlap median {statistics.median(A):.1f} us, end2-end1 median {statistics.median(B):.1f} us; slope of (end2-end1) vs overlap = {cov/vx:.2f} (0 = second kernel does its work after the first ends, i.e. waits (PDL); -1 = real concurrent work), corr {cov/(vx**.5*(sum((b-my)**2 for b in B)/n)**.5):.2f}')
