import json,sys,glob,statistics as st,collections
# per timed request (idle 3/10/40 s): ms from commit() returning to GPU start for the first 3 command buffers, and the 48-cb prefill idle
d=sys.argv[1]; R=collections.defaultdict(list)
for f in sorted(glob.glob(f"{d}/*-r*.trace.jsonl")):
    arm=f.split('/')[-1].split('-r')[0]
    rec=[json.loads(l) for l in open(f)]
    rec=[r for r in rec if "hr" in r]
    if not rec: continue
    reqs=[[rec[0]]]
    for a,b in zip(rec,rec[1:]):
        (reqs[-1].append(b) if b["s"]-a["e"]<1.5 else reqs.append([b]))
    for k,rq in enumerate(reqs):
        if k==0 or len(rq)<3: continue
        lat=[(r["s"]-r["hr"])*1000 for r in rq[:3]]
        R[(arm,k)].append((lat[0],sum(lat)))
def med(x): return f"{st.median(x):6.1f} [{min(x):.1f}..{max(x):.1f}]"
print("first command buffer: commit -> GPU start (ms), and first three summed; median [min..max]; request 1/2/3 = idle 3/10/40 s")
for (arm,k),v in sorted(R.items()):
    print(f" {arm:5s} req {k} (n={len(v)}): first {med([a for a,_ in v])}   first3 {med([b for _,b in v])}")
