import csv,collections,re,statistics as st,sys,itertools
def short(n):
    n=n.replace('void ','').replace('(anonymous namespace)::','').replace('<unnamed>::','')
    return re.split(r'[<(]',n)[0].split('::')[-1][:44]
CAT=[('coll',r'^nccl|rdma|tf_ds_add2'),('prefetch',r'l2_prefetch'),('moe',r'grouped_|down_combine|gateup'),
 ('dense',r'^_gemm|glinear|unpack|kernel2|rot_in|rot_many|rot_out|rowmm|_rmsnorm_rot'),
 ('attn',r'sparse_attn|index_score|topk|rope|kv_|_q_kv|compress|swa|attn|_finish$'),('hc',r'_hc_|hc_')]
def cat(n):
    s=short(n).lower()
    for c,p in CAT:
        if re.search(p,s): return c
    return 'other'
def load(path):
    rows=[]
    for d in csv.DictReader(open(path)):
        if d['Name'].startswith('[CUDA'): continue
        rows.append((int(d['Start (ns)']),int(d['Duration (ns)']),d['Name'],d['GrdX'],d['Strm']))
    rows.sort(); return rows
def ov(a,b):  # intervals lists sorted -> overlap ns
    tot=0;j=0
    for s,e in a:
        while j<len(b) and b[j][1]<=s: j+=1
        k=j
        while k<len(b) and b[k][0]<e:
            tot+=max(0,min(e,b[k][1])-max(s,b[k][0])); k+=1
    return tot
def merge(iv):
    iv=sorted(iv); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
def analyze(ps,label,catf=cat):
    n=len(ps)
    print(f'\n##### {label}: {n} passes')
    # streams
    sc=collections.defaultdict(lambda: collections.Counter())
    for p in ps:
        for x in p: sc[x[4]][catf(x[2])]+=1
    print('streams (kernels per pass by class):')
    for s,c in sorted(sc.items(),key=lambda kv:-sum(kv[1].values())):
        print(f'  stream {s:>4}: '+', '.join(f'{k} {v/n:.1f}' for k,v in c.most_common()))
    cl=sorted({catf(x[2]) for p in ps for x in p})
    # overlap matrix: ms per pass of time where class A and class B both running
    M=collections.defaultdict(float); busy=collections.defaultdict(float); union=0; span=0; conc=0; sumk=0
    for p in ps:
        t0=p[0][0]; 
        iv={c:merge([[x[0],x[0]+x[1]] for x in p if catf(x[2])==c]) for c in cl}
        for c in cl: busy[c]+=sum(e-s for s,e in iv[c])/n/1e6
        for a,b in itertools.combinations(cl,2):
            M[(a,b)]+=ov(iv[a],iv[b])/n/1e6
        allm=merge([[x[0],x[0]+x[1]] for x in p]); union+=sum(e-s for s,e in allm)/n/1e6
        span+=(max(x[0]+x[1] for x in p)-t0)/n/1e6; sumk+=sum(x[1] for x in p)/n/1e6
    print(f'span {span:.2f} ms, union busy {union:.2f}, kernel-sum {sumk:.2f} => concurrency saving {sumk-union:.2f} ms')
    print('class busy (union within class) ms/pass: '+', '.join(f'{c} {busy[c]:.2f}' for c in cl))
    print('pairwise overlap ms/pass:')
    for (a,b),v in sorted(M.items(),key=lambda kv:-kv[1]):
        if v>0.05: print(f'  {a:9s} x {b:9s} {v:6.2f}')
    return cl
if __name__=='__main__':
    r=int(sys.argv[1]); rows=load(f'out/tf-r{r}_cuda_gpu_trace.csv')
    s=[i for i,x in enumerate(rows) if x[2].startswith('_embed_init')]
    passes=[(int(rows[a][3]),rows[a:b]) for a,b in zip(s,s[1:]+[len(rows)])]
    v3=[p for t,p in passes if t==3]
    analyze(v3,f'TF rank {r} verify 3 rows')
