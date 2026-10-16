import json,sys,glob,statistics as st,collections
d=sys.argv[1]
S=collections.defaultdict(dict)
for f in glob.glob(f"{d}/*-r*.idle*.json"):
    base=f.split('/')[-1]; arm=base.split('-r')[0]; rest=base[len(arm)+2:]; r=int(rest.split('.')[0]); idle=int(rest.split('idle')[1].split('.')[0])
    try: S[(arm,idle)][r]=json.loads(open(f).read().strip().splitlines()[-1])
    except Exception as e: print("bad",f,e)
arms=sorted({a for a,_ in S}); idles=sorted({i for _,i in S})
def med(x): return f"{st.median(x):.1f} [{min(x):.1f}..{max(x):.1f}]" if x else "-"
print("TTFT ms, median [min..max] over rounds")
for i in idles:
    print(f" idle {i:>2} s: "+"   ".join(f"{a}: {med([v['ttft_ms'] for v in S[(a,i)].values() if v.get('ttft_ms')])} (n={len(S[(a,i)])})" for a in arms))
print("paired TTFT difference vs base (ms), median [min..max]")
for i in idles:
    for a in arms:
        if a=="base" or ("base",i) not in S: continue
        ds=[S[(a,i)][r]['ttft_ms']-S[("base",i)][r]['ttft_ms'] for r in S[(a,i)] if r in S[("base",i)]]
        print(f" idle {i:>2} s {a}-base: {med(ds)} n={len(ds)}")
print("ITL ms (decode after idle): mean of chunks 2..33, and first 3 gaps median")
for i in idles:
    for a in arms:
        v=[x for x in S[(a,i)].values() if x.get('itl_ms')]
        if v: print(f" idle {i:>2} s {a}: mean-ITL {med([st.mean(x['itl_ms']) for x in v])}  first-3 {med([st.mean(x['itl_ms'][:3]) for x in v])}")
print("text identical across arms for the same prompt:")
bad=0
for i in idles:
    rounds={r for a in arms for r in S[(a,i)]}
    for r in rounds:
        t={a:S[(a,i)][r]['text'] for a in arms if r in S[(a,i)]}
        if len(set(t.values()))>1: bad+=1; print("  DIFF",i,r)
print(f"  {bad} differing groups")
