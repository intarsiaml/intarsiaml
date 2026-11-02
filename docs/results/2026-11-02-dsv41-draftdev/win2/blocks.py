import sys,collections,statistics as st,re
exec(open('overlap.py').read().split("if __name__")[0])
def tf_passes():
    rows=load('out/tf-r0_cuda_gpu_trace.csv')
    s=[i for i,x in enumerate(rows) if x[2].startswith('_embed_init')]
    P=[(int(rows[a][3]),rows[a:b]) for a,b in zip(s,s[1:]+[len(rows)])]
    return [p for t,p in P if t==3], (lambda n:'rdma_stage' in n)
def ours_passes():
    rows=load(__import__('os').environ['OURS_CSV'])
    ps=[];cur=[]
    for x in rows:
        cur.append(x)
        if 'AllGather' in x[2]: ps.append(cur);cur=[]
    v3=[]
    for i in range(len(ps)-1):
        if len(ps[i])>1000 and len(ps[i+1])<400:
            am=[x for x in ps[i+1][:4] if short(x[2])=='argmax_kernel']
            if am and am[0][3]=='3': v3.append(ps[i])
    return v3,(lambda n:'AllReduce' in n and 'f32' in n)
for label,(ps,isc) in (('TF',tf_passes()),('ours',ours_passes())):
    att=[];moe=[];coll=[];first=[];lay=[]
    for p in ps:
        cs=[x for x in p if isc(x[2])]
        cs.sort()
        # collectives alternate attn,moe per layer (2/layer); the first segment before the first collective
        ends=[x[0]+x[1] for x in cs]; starts=[x[0] for x in cs]
        for i in range(len(cs)):
            coll.append(cs[i][1]/1e3)
        for i in range(1,len(cs)-1):
            seg=(starts[i]-ends[i-1])/1e3
            (moe if i%2==1 else att).append(seg)   # i odd: after attn collective (index even) -> moe block
        for i in range(2,len(cs),2):
            lay.append((ends[i]-ends[i-2])/1e3)
    n=len(ps)
    print(f'{label}: {n} passes; collective dur median {st.median(coll):.1f} us; attention-block wall (prev coll end -> this coll start) median {st.median(att):.0f} us; moe-block wall median {st.median(moe):.0f} us; layer (coll end to coll end) median {st.median(lay):.0f} us mean {st.mean(lay):.0f}')
