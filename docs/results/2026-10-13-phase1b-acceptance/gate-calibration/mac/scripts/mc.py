import numpy as np
rng=np.random.default_rng(1)
N=400000
def sample(kind,size):
    if kind=="normal": return rng.standard_normal(size)
    if kind=="t3": return rng.standard_t(3,size)
    if kind=="laplace": return rng.laplace(size=size)
print("false-fail rate of ONE per-prompt check (upper limit b + max(fixed=0, k*max_i |e1_i-e0|)), equally correct new build")
print("kind     m   k=1    k=2    k=3    k=5    k=10")
for kind in ["normal","laplace","t3"]:
    for m in [1,2,3,4]:
        e0=sample(kind,N); e2=sample(kind,N); e1=sample(kind,(m,N))
        D=np.abs(e1-e0).max(axis=0); X=e2-e0
        row=[(X>k*D).mean() for k in (1,2,3,5,10)]
        print(f"{kind:8s} {m}  "+"  ".join(f"{r:5.3f}" for r in row))
print()
print("expected failing checks in a gate with n independent checks at k=2: 1-(1-p)^n")
for p in (0.05,0.10,0.15):
    print(p,[round(1-(1-p)**n,2) for n in (20,60)])
