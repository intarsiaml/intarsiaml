import os,statistics as st,sys,importlib
import shapes
D=sys.argv[1]; tag=sys.argv[2]
def ps(r):
    shapes.OURS=f'{D}/nsys-r{r}-{tag}_cuda_gpu_trace.csv'
    return shapes.ours3()
P0,P1=ps(0),ps(1)
print(len(P0),len(P1))
rows=[]
for a,b in zip(P0,P1):
    qa=sorted(a,key=lambda x:x['t']);qb=sorted(b,key=lambda x:x['t'])
    Ra=[x for x in qa if 'AllReduce' in x['n']];Rb=[x for x in qb if 'AllReduce' in x['n']]
    # offset: the ends of later all-reduces are close to simultaneous
    off=st.median((x['t']+x['d'])-(y['t']+y['d']) for x,y in zip(Ra[1:20],Rb[1:20]))
    ea=[x for x in qa if 'embedding_kernel' in x['n'] and x['gx']=='3'][0];eb=[x for x in qb if 'embedding_kernel' in x['n'] and x['gx']=='3'][0]
    rows.append(((eb['t']+off-ea['t'])/1e3,(Rb[0]['t']+off-Ra[0]['t'])/1e3,Ra[0]['d']/1e3,Rb[0]['d']/1e3))
for name,i in (('r1 embed start - r0 embed start',0),('r1 AR0 start - r0 AR0 start',1),('r0 AR0 dur',2),('r1 AR0 dur',3)):
    v=[r[i] for r in rows]; print(f'{name:34s} median {st.median(v):8.1f} min {min(v):8.1f} max {max(v):8.1f}')
