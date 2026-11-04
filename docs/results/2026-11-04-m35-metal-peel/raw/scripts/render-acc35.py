import collections,sys,math
def load(p):
    d=collections.OrderedDict()
    for l in open(p):
        f=l.rstrip("\n").split("\t")
        if len(f)>=8: d.setdefault((f[0][:7],f[1]),[]).append(float(f[5]))
    return d
sets=[("d","d"),("d2","d"),("d3","d")]
print("| prompt (tool length n; prefilled L) | serving prompt length of that prefill | class | rows moved | KL before (main) | KL after (peel) | after / before | largest single row before -> after |")
print("|---|---|---|---|---|---|---|---|")
tot=[]
for s,_ in sets:
    try: B=load(f".scratch/acc35/{s}/main.pos.tsv"); A=load(f".scratch/acc35/{s}/peel.pos.tsv")
    except Exception: continue
    for k in B:
        if B[k]==A[k]: continue
        n=int(k[1][1:]); L=n-32; p=L+1
        c="exposed" if 2050<=p%4096<=2057 else "peeled, not exposed"
        kb=sum(B[k])/len(B[k]); ka=sum(A[k])/len(A[k]); mv=sum(x!=y for x,y in zip(B[k],A[k]))
        print(f"| {k[1]} (L = {L}) | {p} | {c} | {mv} of {len(B[k])} | {kb:.3e} | {ka:.3e} | {ka/kb:.2f} | {max(B[k]):.2f} -> {max(A[k]):.2f} |")
        tot.append((c,kb,ka))
for cl in ("exposed","peeled, not exposed",None):
    R=[t for t in tot if cl is None or t[0]==cl]
    if not R: continue
    n=len(R); rat=sorted(ka/kb for _,kb,ka in R); med=rat[n//2] if n%2 else (rat[n//2-1]+rat[n//2])/2
    print(f"\n{cl or 'all'}: {n} windows, mean KL before {sum(t[1] for t in R)/n:.3e} after {sum(t[2] for t in R)/n:.3e}, median ratio {med:.2f}, after closer on {sum(ka<kb for _,kb,ka in R)}, farther on {sum(ka>kb for _,kb,ka in R)}", file=sys.stderr)
