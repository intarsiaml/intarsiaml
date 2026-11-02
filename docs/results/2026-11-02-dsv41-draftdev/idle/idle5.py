import statistics as st
import shapes
from shapes import *
O=shapes.ours3()
for i in (0,1,2,3):
    v=[[x for x in sorted(p,key=lambda x:x['t']) if 'AllReduce' in x['n']][i]['d']/1e3 for p in O]
    print('AR',i,'dur us median',round(st.median(v),1),'max',round(max(v)))
v=[]
for p in O:
    q=sorted(p,key=lambda x:x['t'])
    e=[x for x in q if 'embedding_kernel' in x['n'] and x['gx']=='3'][0]; a=[x for x in q if 'AllReduce' in x['n']][0]
    v.append((a['t']-e['t'])/1e3)
print('verify start to AR0 start us median',round(st.median(v),1))
