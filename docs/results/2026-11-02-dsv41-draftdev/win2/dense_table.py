from shapes import *
BYTES={'wo_b':500.0,'wq_b':500.0,'wo_a':400.0,'wq_a+wkv+compressor':None,'head':236.72,'engram wkv':None,'indexer wq_b':25.0,'gate (f16)':150.0}
def row(ps,sel):
    n=len(ps);c=0;t=0
    for p in ps:
        for x in p:
            if sel(x): c+=1/n;t+=x['d']/n/1e6
    return c,t
def nm(x): return x['n']
TF=tf3();O=ours3()
g=lambda x,*gx:x['gx'] in gx
tf={
 'wo_b':lambda x:'glinear' in x['n'] and x['gx']=='320',
 'wq_b':lambda x:'glinear' in x['n'] and x['gx']=='128',
 'wo_a':lambda x:'glinear' in x['n'] and x['gx']=='256',
 'wq_a+wkv(+compr)':lambda x:'glinear' in x['n'] and x['gx'] in('112','144','176','96'),
 'head':lambda x:'glinear' in x['n'] and x['gx']=='505',
 'engram wkv':lambda x:'glinear' in x['n'] and x['gx']=='200',
 'indexer wq_b':lambda x:'glinear' in x['n'] and x['gx']=='64',
 'indexer wk':lambda x:'glinear' in x['n'] and x['gx']=='1',
 'gate (f16)':lambda x:'_rowmm_parts' in x['n'],
 'indexer proj':lambda x:'_rowmm2' in x['n'],
 'helpers (rot_many)':lambda x:'rot_many' in x['n'],
}
isl=lambda x:'linear_kernel' in x['n'] and 'split' not in x['n']
ours={
 'wo_b':lambda x:isl(x) and x['gx']=='40',
 'wq_b':lambda x:isl(x) and x['gx']=='128',
 'wo_a':lambda x:'split_kernel' in x['n'] and x['gx']=='8' and x['gy']=='8',
 'wq_a+wkv(+compr)':lambda x:'split_kernel' in x['n'] and ((x['gx']=='10' and x['gy']=='5') or (x['gx']=='4' and x['gy']=='10')),
 'head':lambda x:isl(x) and x['gx']=='505',
 'engram wkv':lambda x:isl(x) and x['gx']=='200',
 'indexer wq_b':lambda x:isl(x) and x['gx']=='32',
 'indexer wk':lambda x:'split_kernel' in x['n'] and x['gx']=='1',
 'gate (f16)':lambda x:'dense_f16w' in x['n'] and x['gx']=='96',
 'indexer proj':lambda x:'dense_f16w' in x['n'] and x['gx']=='32',
 'helpers (rot_in, finish, transpose)':lambda x:('rot_in_kernel' in x['n'] and x['gy']=='3') or 'split_finish' in x['n'] or 'transpose01_kernel<uint4>' in x['n'],
 'Markov head of the draft before (not verify)':lambda x:'dense_f16w' in x['n'] and x['gx']=='16160',
}
if __name__=='__main__':
    print(f"{'shape':38s}  TF: n   us/call   ms   | ours: n   us/call   ms  | ours-TF ms")
    tot=[0,0]
    keys=list(dict.fromkeys(list(tf)+list(ours)))
    for k in keys:
        a=row(TF,tf[k]) if k in tf else (0,0)
        b=row(O,ours[k]) if k in ours else (0,0)
        tot[0]+=a[1];tot[1]+=b[1]
        print(f"{k:38s} {a[0]:6.1f} {1000*a[1]/a[0] if a[0] else 0:8.1f} {a[1]:7.3f} | {b[0]:6.1f} {1000*b[1]/b[0] if b[0] else 0:8.1f} {b[1]:7.3f} | {b[1]-a[1]:7.3f}")
    print('total',tot)
    # tiny kernels
    for label,ps in (('TF',TF),('ours',O)):
        n=len(ps);c=collections.Counter();t=collections.Counter()
        for p in ps:
            for x in p:
                if x['d']<5000 and 'prefetch' not in x['n']:
                    k=short(x['n'])[:32];c[k]+=1/n;t[k]+=x['d']/n/1e6
        print(label,'kernels under 5 us: %.0f/pass, %.2f ms/pass; top:'%(sum(c.values()),sum(t.values())),[(k,round(c[k]),round(t[k],3)) for k,_ in t.most_common(7)])
        # gaps
        g=0;big=0
        for p in ps:
            q=sorted([x for x in p if 'prefetch' not in x['n']],key=lambda x:x['t']);e=q[0]['t']+q[0]['d']
            for x in q[1:]:
                if x['t']>e: g+=(x['t']-e)/n/1e6
                e=max(e,x['t']+x['d'])
        print(label,'idle gaps ms/pass %.2f'%g)
    
