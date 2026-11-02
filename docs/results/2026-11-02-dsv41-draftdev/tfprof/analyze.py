import csv,collections,re,statistics as st,sys
r=int(sys.argv[1])
rows=[]
for d in csv.DictReader(open(f'tf-r{r}_cuda_gpu_trace.csv')):
    if d['Name'].startswith('[CUDA'): continue
    rows.append((int(d['Start (ns)']),int(d['Duration (ns)']),d['Name'],d['GrdX'],d['Strm']))
rows.sort()
starts=[i for i,x in enumerate(rows) if x[2].startswith('_embed_init')]
passes=[]
for a,b in zip(starts,starts[1:]+[len(rows)]):
    passes.append((int(rows[a][3]),rows[a:b]))
def short(n):
    n=n.replace('void ','').replace('(anonymous namespace)::','').replace('<unnamed>::','')
    n=re.split(r'[<(]',n)[0]
    return n.split('::')[-1][:44]
CAT=[('collectives (nccl head gather)',r'^nccl'),('collectives (rdma stage/collect)',r'rdma|tf_ds_add2'),('L2 prefetch (side stream)',r'l2_prefetch'),
 ('moe experts (grouped_cp/rows, down_combine)',r'grouped_|down_combine|gateup'),('dense linears (EXL3 glinear/gemm/unpack/rot)',r'^_gemm|glinear|unpack|kernel2|rot_in|rot_many|rot_out|rowmm|_rmsnorm_rot'),
 ('attention (sparse_attn, kv, index, rope)',r'sparse_attn|index_score|topk|rope|kv_|_q_kv|compress|swa|attn|_finish$'),('hyper-connection (hc_*)',r'_hc_|hc_'),('router/score/bias',r'route|_score|_bias|bias'),('norm',r'rmsnorm|norm'),('engram',r'engram'),('embed/head/sample',r'embed|head|argmax|sample|logit|bf16_to_f32|f32_to_bf16')]
def cat(n):
    s=short(n).lower()
    for c,p in CAT:
        if re.search(p,s): return c
    return 'other'
for label,sel in (('PREFILL T=1222',lambda t:t>100),('DRAFT T=5',lambda t:t==5),('VERIFY T=2..4',lambda t:2<=t<=4)):
    ps=[p for t,p in passes if sel(t)]
    if label=='DRAFT T=5': ps=ps[1:] if len(passes)>3 else ps
    if not ps: continue
    n=len(ps)
    spans=[(p[-1][0]+p[-1][1]-p[0][0])/1e6 for p in ps]
    # gpu-busy on main stream 13 and total kernel time
    ksum=[sum(x[1] for x in p)/1e6 for p in ps]
    s13=[sum(x[1] for x in p if x[4]=='13')/1e6 for p in ps]
    def union(p):
        iv=sorted((x[0],x[0]+x[1]) for x in p); tot=0; cur=None
        for a,b in iv:
            if cur is None: cur=[a,b]
            elif a<=cur[1]: cur[1]=max(cur[1],b)
            else: tot+=cur[1]-cur[0]; cur=[a,b]
        return (tot+(cur[1]-cur[0] if cur else 0))/1e6
    un=[union(p) for p in ps]
    print(f'\n  gpu busy (union of all streams) {st.median(un):.2f} ms of span {st.median(spans):.2f} ms -> idle {st.median(spans)-st.median(un):.2f} ms')
    print(f'\n== rank {r} {label}: {n} passes, span median {st.median(spans):.2f} ms, kernel-sum {st.median(ksum):.2f} ms, main-stream busy {st.median(s13):.2f} ms, kernels/pass {st.median(len(p) for p in ps)}')
    agg=collections.defaultdict(float); cnt=collections.defaultdict(float)
    for p in ps:
        for x in p:
            c=cat(x[2]); agg[c]+=x[1]/1e6/n; cnt[c]+=1/n
    for c,v in sorted(agg.items(),key=lambda kv:-kv[1]): print(f'  {c:34s} {v:7.3f} ms/pass  {cnt[c]:7.1f} kernels')
    if label!='PREFILL T=1222':
        a2=collections.defaultdict(float); c2=collections.defaultdict(float)
        for p in ps:
            for x in p:
                k=short(x[2]); a2[k]+=x[1]/1e6/n; c2[k]+=1/n
        print('  top kernels:')
        for k,v in sorted(a2.items(),key=lambda kv:-kv[1])[:14]: print(f'    {k:44s} {v:7.3f} ms/pass {c2[k]:6.1f}x')
# round period: embed_init(T=5) start to next T=5 start
d5=[rows[starts[i]][0] for i,(t,p) in enumerate(passes) if t==5]
if len(d5)>4: print('\nround period (draft start to next draft start) median ms', st.median((b-a)/1e6 for a,b in zip(d5[1:],d5[2:])), 'n',len(d5)-2)
v=[(rows[starts[i]][0],t) for i,(t,p) in enumerate(passes) if 2<=t<=4]
d=[rows[starts[i]][0] for i,(t,p) in enumerate(passes) if t==5]
# draft->verify gap and verify->draft gap
