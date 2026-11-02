import csv,re,os,statistics as st,collections
from shapes import short
def loadall(p):
    R=[]
    for d in csv.DictReader(open(p)):
        R.append(dict(t=int(d['Start (ns)']),d=int(d['Duration (ns)']),n=d['Name'],gx=d['GrdX'],s=d['Strm'],b=d['Bytes (MB)']))
    R.sort(key=lambda x:x['t']); return R
R=loadall(os.environ['OURS_CSV'])
# print one round from a draft's last kernels through the next verify's first kernels
idx=[i for i,x in enumerate(R) if 'AllGather' in x['n']]
i=idx[12]
t0=R[i]['t']
print('OURS around draft end'); 
for x in R[i-3:i+36]:
    if 'hc_' in x['n'] : continue
    print(f"{(x['t']-t0)/1e3:9.1f} {x['d']/1e3:8.1f} {short(x['n'])[:34] if not x['n'].startswith('[') else x['n'][:34]:34s} {x['gx']:>6s} {x['b']:>8s}")
    if 'split_kernel' in x['n']: break
# round partition ours
K=[x for x in R if not x['n'].startswith('[CUDA')]
ps=[];c=[]
for x in K:
    c.append(x)
    if 'AllGather' in x['n']: ps.append(c);c=[]
rows=[]
for a,b,cn in zip(ps,ps[1:],ps[2:]):
    if len(a)<400 and len(b)>1000:   # a: draft (ends with gather), b: markov+verify
        vs=[x for x in b if 'embedding_kernel' in x['n'] and x['gx'] not in('1',)]
        if not vs: continue
        v0=vs[0]['t']
        mk=[x for x in b if x['t']<v0]
        rowsn=vs[0]['gx']
        dstart=a[0]['t']; dend=a[-1]['t']+a[-1]['d']
        mend=mk[-1]['t']+mk[-1]['d'] if mk else dend
        vend=b[-1]['t']+b[-1]['d']
        dn=[x for x in cn if 'embedding' in x['n'] or 'hc_' in x['n']]
        nxt=cn[0]['t']
        rows.append(dict(rows=rowsn,draft=(dend-dstart)/1e6,markov=(mend-dend)/1e6,gap=(v0-mend)/1e6,verify=(vend-v0)/1e6,after=(nxt-vend)/1e6,period=(nxt-dstart)/1e6, dfirst=short(a[0]['n'])))
print(collections.Counter(r['rows'] for r in rows), 'draft starts with', rows[0]['dfirst'])
for rw in ('3','6','9','12'):
    z=[r for r in rows if r['rows']==rw]
    if z: print('grid',rw,len(z),{k:round(st.median(r[k] for r in z),2) for k in ('draft','markov','gap','verify','after','period')})
print('all',len(rows),{k:round(st.median(r[k] for r in rows),2) for k in ('draft','markov','gap','verify','after','period')})
# what's in the draft segment beyond the draft: verify's tail
a=[p for p in ps if len(p)<400][5]
print('draft segment kernels:',[short(x['n'])[:18]+'/'+x['gx'] for x in a[:8]],'...',[short(x['n'])[:18]+'/'+x['gx'] for x in a[-4:]])
# TF
T=loadall('out/tf-r0_cuda_gpu_trace.csv')
print('TF T=5 kernels tail'); 
s=[i for i,x in enumerate(T) if x['n'].startswith('_embed_init')]
for ii,(a,b) in enumerate(zip(s,s[1:])):
    if T[a]['gx']=='5' and ii>10:
        for x in T[b-14:b+3]:
            print(f"{(x['t']-T[a]['t'])/1e3:9.1f} {x['d']/1e3:8.1f} {(short(x['n']) if not x['n'].startswith('[') else x['n'])[:34]:34s} {x['gx']:>6s} {x['b']:>8s}")
        break
