import json,sys,glob,os,numpy as np
d=sys.argv[1]
def bf(path,shape):
    a=np.fromfile(path,dtype=np.uint16).astype(np.uint32)<<16
    return a.view(np.float32).reshape(shape).astype(np.float64)
rows=[]
for jf in sorted(glob.glob(d+'/attn_*.json'),key=lambda p:int(os.path.basename(p)[5:-5])):
    i=jf[:-5]; m=json.load(open(jf)); T,N,D=m['q']; NKV,cap,_=m['k']; p0=m['pos0']; sc=m['scale']
    q=bf(i+'.q',(T,N,D)); k=bf(i+'.k',(NKV,cap,D)); v=bf(i+'.v',(NKV,cap,D)); out=bf(i+'.out',(T,N,D))
    G=N//NKV; ref=np.zeros((T,N,D))
    smax=0
    for h in range(N):
        kv=h//G
        L=p0+T
        s=(q[:,h,:]@k[kv,:L,:].T)*sc   # T x L
        mask=np.arange(L)[None,:]>(p0+np.arange(T))[:,None]
        s=np.where(mask,-np.inf,s); smax=max(smax,np.abs(s[np.isfinite(s)]).max())
        s=s-s.max(1,keepdims=True); e=np.exp(s); pr=e/e.sum(1,keepdims=True)
        ref[:,h,:]=pr@v[kv,:L,:]
    err=np.abs(out-ref); rel=np.linalg.norm(out-ref)/np.linalg.norm(ref)
    # per-row rel
    rr=np.linalg.norm((out-ref).reshape(T,-1),axis=1)/np.linalg.norm(ref.reshape(T,-1),axis=1)
    print(f"{os.path.basename(i)} {m['kernel']:28s} T={T} pos0={p0} max|s|={smax:.1f} max|q|={np.abs(q).max():.1f} max|k|={np.abs(k[:,:p0+T]).max():.1f} max|v|={np.abs(v[:,:p0+T]).max():.1f} maxabs={err.max():.3e} rel_l2={rel:.3e} worst_row={rr.argmax()} {rr.max():.3e}")
