import statistics as st,sys
import shapes
from shapes import *
O=shapes.ours3()
sp=[]
for p in O:
    v=[x for x in p if 'embedding_kernel' in x['n'] and x['gx']=='3']
    sp.append((max(x['t']+x['d'] for x in p)-v[0]['t'])/1e6)
print('passes',len(O),'verify-only span median',round(st.median(sp),2))
T=[]
for p in O:
    q=sorted(p,key=lambda x:x['t']);e=q[0]['t'];g=0
    for x in q[1:]:
        if x['t']>e and 'nccl' in x['n'].lower(): g+=(x['t']-e)/1e6
        e=max(e,x['t']+x['d'])
    T.append(g)
print('idle before NCCL kernels ms/pass median',round(st.median(T),2))
AR=[x['d']/1e3 for p in O for x in p if 'Sum_bf16' in x['n']]; print('Engram AR us median',st.median(AR),'mean',round(sum(AR)/len(AR)),'max',round(max(AR)))
F=[x['d']/1e3 for p in O for x in p if 'Sum_f32' in x['n']]; print('f32 AR us median',st.median(F),'mean',round(sum(F)/len(F),1))
