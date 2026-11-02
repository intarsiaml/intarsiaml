from shapes import *
from dense_table import tf as TFS, ours as OS, row
import statistics as st
TF=tf3();O=ours3()
BY={'wo_b':(518.7,518.7),'wq_b':(500,500),'wo_a':(400,400),'wq_a+wkv(+compr)':(230.7,230.7),'head':(236.7,236.7),'engram wkv':(84.4,168.8),'indexer wq_b':(25,25),'indexer wk':(.2,.2),'gate (f16)':(150,150),'indexer proj':(25,25)}
print('| shape | MiB TF / ours | TF calls | TF us/call | TF ms | TF GB/s | ours calls | ours us/call | ours ms | ours GB/s | ours-TF ms |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for k,(a,b) in BY.items():
    x=row(TF,TFS[k]);y=row(O,OS[k])
    print(f"| {k} | {a:g} / {b:g} | {x[0]:.0f} | {1000*x[1]/x[0]:.1f} | {x[1]:.2f} | {a*1.048576/x[1]:.0f} | {y[0]:.0f} | {1000*y[1]/y[0]:.1f} | {y[1]:.2f} | {b*1.048576/y[1]:.0f} | {y[1]-x[1]:+.2f} |")
x=row(TF,TFS['helpers (rot_many)']);y=row(O,OS['helpers (rot_in, finish, transpose)'])
print(f"| helpers (TF rot_many; ours rot_in, split_finish, transposes) | - | {x[0]:.0f} | {1000*x[1]/x[0]:.1f} | {x[1]:.2f} | - | {y[0]:.0f} | {1000*y[1]/y[0]:.1f} | {y[1]:.2f} | - | {y[1]-x[1]:+.2f} |")
# gaps, medians per pass
def gaps(ps,skip=lambda n:False):
    res=collections.defaultdict(list); tot=[]
    for p in ps:
        q=sorted([x for x in p if not skip(x['n'])],key=lambda x:x['t']); e=q[0]['t']+q[0]['d'];prev=q[0]; g=collections.Counter(); T=0
        for x in q[1:]:
            if x['t']>e:
                g[short(prev['n'])[:24]+' -> '+short(x['n'])[:22]]+=(x['t']-e)/1e3; T+=(x['t']-e)/1e6
            if x['t']+x['d']>e: e=x['t']+x['d'];prev=x
        tot.append(T)
        for k,v in g.items(): res[k].append(v)
    return res,tot
for label,ps,sk in (('TF',TF,lambda n:'prefetch' in n),('ours',O,lambda n:False)):
    r,t=gaps(ps,sk); print(label,'idle ms/pass median',round(st.median(t),2),'max',round(max(t),2))
    for k,v in sorted(r.items(),key=lambda kv:-st.median(kv[1])*len(kv[1])/len(ps))[:6]: print(f'   {k:50s} median {st.median(v):8.1f} us/pass n {len(v)}')
AR=[x['d']/1e3 for p in O for x in p if 'Sum_bf16' in x['n']]
print('Engram AR us median',st.median(AR),'mean',round(sum(AR)/len(AR)),'max',round(max(AR)),'n',len(AR))
F=[x['d']/1e3 for p in O for x in p if 'Sum_f32' in x['n']]; print('f32 AR us median',st.median(F),'mean',round(sum(F)/len(F),1))
# verify-only span, median
sp=[]
for p in O:
    v=[x for x in p if 'embedding_kernel' in x['n'] and x['gx']=='3']
    sp.append((max(x['t']+x['d'] for x in p)-v[0]['t'])/1e6)
print('ours verify-only span median',round(st.median(sp),2),'; TF',round(st.median((max(x['t']+x['d'] for x in p)-p[0]['t'])/1e6 for p in TF),2))
