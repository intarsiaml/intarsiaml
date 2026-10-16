import json,sys,glob,statistics as st,collections
# per timed request (idle 3/10/40 s): GPU busy, span, idle-in-span of the prefill command buffers (records with t == chunk), lead = first record start
# relative to the previous request's last end is not usable; here: busy / span / idle-in-span / count, median [min..max] over rounds
d=sys.argv[1]; R=collections.defaultdict(list)
for f in sorted(glob.glob(f"{d}/*-r*.trace.jsonl")):
    arm=f.split('/')[-1].split('-r')[0]
    rec=[json.loads(l) for l in open(f)]
    if not rec: continue
    reqs=[[rec[0]]]
    for a,b in zip(rec,rec[1:]):
        (reqs[-1].append(b) if b["s"]-a["e"]<1.5 else reqs.append([b]))
    for k,rq in enumerate(reqs):
        if k==0: continue
        pre=[r for r in rq if r["t"]>1]   # prefill records (t = rows), decode is t == 1
        if len(pre)<3: continue
        busy=sum(r["e"]-r["s"] for r in pre)*1000; span=(pre[-1]["e"]-pre[0]["s"])*1000
        R[(arm,k)].append((busy,span,span-busy,len(pre)))
def med(x): return f"{st.median(x):7.1f} [{min(x):.1f}..{max(x):.1f}]"
print("prefill command buffers of the timed request (request 1/2/3 = idle 3/10/40 s): GPU busy, span first start..last end, idle inside the span, n cbs")
for (arm,k),v in sorted(R.items()):
    print(f" {arm:5s} req {k} (n={len(v)}): busy {med([a for a,_,_,_ in v])}  span {med([b for _,b,_,_ in v])}  idle-in-span {med([c for _,_,c,_ in v])}  cbs {v[0][3]}")
