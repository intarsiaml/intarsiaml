import re,sys,statistics,os
raw=sys.argv[1]
arms=[("main","2048 (main, PF on)"),("main0","2048, PF=0"),("c4096","4096"),("c8192","8192")]
def load(a,r):
    p=f"{raw}/df-{a}-r{r}.out"
    if not os.path.exists(p): return None
    d={}
    for m in re.finditer(r"(\d+) tok rep \d/\d: prompt=\d+ ttft=[\d.]+s prefill=(\d+) tok/s decode=([\d.]+)",open(p).read()):
        d.setdefault(int(m.group(1)),([],[]))
        d[int(m.group(1))][0].append(int(m.group(2))); d[int(m.group(1))][1].append(float(m.group(3)))
    return d
res={}
for a,_ in arms:
    for r in (1,2,3):
        d=load(a,r)
        if d:
            for s,(pf,dc) in d.items():
                res.setdefault((a,s),[]).append((statistics.median(pf),statistics.median(dc)))
print("| arm | 1k prefill tok/s | 4k | 16k | vs 2048 PF on (1k / 4k / 16k) |\n|---|---|---|---|---|")
base={s:statistics.median(x[0] for x in res[("main",s)]) for s in (1024,4096,16384) if ("main",s) in res}
for a,n in arms:
    row=[];rel=[]
    for s in (1024,4096,16384):
        v=res.get((a,s))
        if not v: row.append("-");rel.append("-");continue
        m=statistics.median(x[0] for x in v)
        row.append(f"{m:.0f} ({', '.join(f'{x[0]:.0f}' for x in v)})")
        rel.append(f"{(m/base[s]-1)*100:+.1f} %" if s in base else "-")
    print(f"| {n} | {row[0]} | {row[1]} | {row[2]} | {' / '.join(rel)} |")
print("\ndecode tok/s (median over rounds): "+"; ".join(f"{n}: "+" / ".join(f"{statistics.median(x[1] for x in res[(a,s)]):.1f}" for s in (1024,4096,16384) if (a,s) in res) for a,n in arms))
