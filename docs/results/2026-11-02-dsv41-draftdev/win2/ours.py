import sys,re,collections,statistics as st
exec(open('overlap.py').read().split("if __name__")[0])
OC=[('coll',r'^nccl|f32_to_bf16'),('moe',r'grouped_cp|group_kernel|epilogue|combine|moe_'),
    ('dense',r'linear_kernel|split_kernel|split_finish|dense_f16|rot_in|rot_out|rot_many|qdq|transpose01|add_f32|^add_|cast'),
    ('attn',r'attn_|rope|index_|window_build|ring_write|kv_|compress|topk'),('hc',r'^hc_')]
def ocat(n):
    s=short(n).lower()
    for c,p in OC:
        if re.search(p,s): return c
    return 'other'
path=sys.argv[1]; rows=load(path)
ps=[];cur=[]
for x in rows:
    cur.append(x)
    if 'AllGather' in x[2]: ps.append(cur);cur=[]
v3=[]
for i in range(len(ps)-1):
    if len(ps[i])>1000 and len(ps[i+1])<400:
        nxt=ps[i+1]
        am=[x for x in nxt[:4] if short(x[2])=='argmax_kernel']
        if am and am[0][3]=='3': v3.append(ps[i])
print('verify 3-row passes',len(v3))
cl=analyze(v3,'OURS verify 3 rows',ocat)
cs=collections.defaultdict(float)
for p in v3:
    for x in p: cs[ocat(x[2])]+=x[1]/len(v3)/1e6
print('class kernel-sum ms/pass',{k:round(v,2) for k,v in sorted(cs.items(),key=lambda kv:-kv[1])})
# top kernels
ks=collections.defaultdict(float);kc=collections.defaultdict(float)
for p in v3:
    for x in p: ks[short(x[2])]+=x[1]/len(v3)/1e6; kc[short(x[2])]+=1/len(v3)
for k,v in sorted(ks.items(),key=lambda kv:-kv[1])[:12]: print(f'  {k:40s} {v:7.2f} ms {kc[k]:6.0f}x')
# conc hist
H=collections.defaultdict(float)
for p in v3:
    ev=[]
    for x in p: ev+= [(x[0],1),(x[0]+x[1],-1)]
    ev.sort(); c=0; last=ev[0][0]
    for t,d in ev: H[c]+=(t-last)/len(v3)/1e6; last=t; c+=d
print('concurrency histogram ms/pass',{k:round(v,2) for k,v in sorted(H.items())})
# draft pass
d=[ps[i+1] for i in range(len(ps)-1) if len(ps[i])>1000 and len(ps[i+1])<400]
print('draft passes',len(d),'span median',st.median((p[-1][0]+p[-1][1]-p[0][0])/1e6 for p in d))
print('verify span by rows:')
byr=collections.defaultdict(list)
for i in range(len(ps)-1):
    if len(ps[i])>1000 and len(ps[i+1])<400:
        am=[x for x in ps[i+1][:4] if short(x[2])=='argmax_kernel']
        if am: byr[am[0][3]].append((ps[i][-1][0]+ps[i][-1][1]-ps[i][0][0])/1e6)
for k,v in sorted(byr.items()): print(' rows',k,len(v),round(st.median(v),2))
