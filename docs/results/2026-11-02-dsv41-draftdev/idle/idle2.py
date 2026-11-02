import os,sys,statistics as st,collections,csv
import shapes
from shapes import *
_s=short
def short(n):
    return n[:30] if n.startswith('[') else _s(n)
O=shapes.ours3()
ALL=[]
for d in csv.DictReader(open(shapes.OURS)):
    ALL.append(dict(t=int(d['Start (ns)']),d=int(d['Duration (ns)']),n=d['Name']))
ALL.sort(key=lambda x:x['t'])
# for one pass: list all gaps > 15 us with neighbours and copy/mem events too
p=O[2]; a=p[0]['t']; b=p[-1]['t']+p[-1]['d']
q=[x for x in ALL if a<=x['t']<=b]
e=q[0]['t']; prev=q[0]
print('pass 2 span us',(b-a)/1e3,'events',len(q))
nccl_i=0
for x in q[1:]:
    g=(x['t']-e)/1e3
    if 'nccl' in x['n'].lower(): nccl_i+=1
    if g>15: print(f"gap {g:8.1f}us at {(x['t']-a)/1e3:9.1f}  {short(prev['n'])[:30]:30s} -> {short(x['n'])[:30]:30s} d={x['d']/1e3:.0f} ncclidx={nccl_i}")
    if x['t']+x['d']>e: e=x['t']+x['d'];prev=x
# cuda names in the pass
c=collections.Counter(x['n'][:40] for x in q if x['n'].startswith('[CUDA'))
print(c)
# first gap to AR by index across passes
idx=collections.defaultdict(list)
for pp in O:
    r=sorted(pp,key=lambda x:x['t']);e=r[0]['t'];i=0
    for x in r[1:]:
        if 'AllReduce' in x['n']:
            idx[(i,'bf16' if 'bf16' in x['n'] else 'f32')].append((x['t']-e)/1e3); i+=1
        e=max(e,x['t']+x['d'])
print('AR index: median gap (top 12)')
for k,v in sorted(idx.items(),key=lambda kv:-st.median(kv[1]))[:12]: print(k,round(st.median(v),1),'n',len(v))
print('sum of medians',round(sum(st.median(v) for v in idx.values())),'over',len(idx),'ARs; median of others',round(st.median([st.median(v) for k,v in idx.items()]),1))
