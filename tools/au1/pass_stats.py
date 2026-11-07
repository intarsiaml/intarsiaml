import sys, os
sys.argv=['x','docs/results/2026-11-06-spec-auto-tp1/raw/w1']
sys.path.insert(0,'tools/asa')
os.environ['REPLAY_ARMS']='auto,autotrmtp,autotrdfl'
import replay as R
raw='docs/results/2026-11-06-spec-auto-tp1/raw/w1'
A=R.load(f'{raw}/pass-auto-r1.log'); M=R.load(f'{raw}/pass-autotrmtp-r1.log'); D=R.load(f'{raw}/pass-autotrdfl-r1.log')
import statistics as st
def stats(reqs, mode, skipfresh=True):
    t=ms=n=0
    for r in reqs:
        for j,(m,tk,mm,fr) in enumerate(r):
            if m!=mode: continue
            n+=1; t+=tk; ms+=mm
    return n, (t/n if n else 0), (ms/n if n else 0)
i=0
for c,k in R.CATS:
    a=A[i:i+k]; m=M[i:i+k]; d=D[i:i+k]; i+=k
    an_m=stats(a,'M'); an_d=stats(a,'D'); pm=stats(m,'M'); pd=stats(d,'D')
    tot=sum(len(r) for r in a)
    print(f"{c:10} auto: M n{an_m[0]:4d} t/p {an_m[1]:.2f} ms {an_m[2]:.1f} | D n{an_d[0]:4d} t/p {an_d[1]:.2f} ms {an_d[2]:.1f} | pureM t/p {pm[1]:.2f} ms {pm[2]:.1f} | pureD t/p {pd[1]:.2f} ms {pd[2]:.1f} | firsts {sum(fr for r in a for (_,_,_,fr) in r)}")
