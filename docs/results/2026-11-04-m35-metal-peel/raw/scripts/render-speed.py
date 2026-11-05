import re,sys,glob,statistics as st
# rows for the doc: per size and class, prompt token range of the reps in that class
def rows(d):
    out=[]
    for f in sorted(glob.glob(f"{d}/*-localeval.out")):
        m0=re.search(r"/(main|peel)-(\d)-(\d+)-localeval",f)
        if not m0: continue
        for l in open(f):
            m=re.search(r"(\d+) tok rep (\d)/\d: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)",l)
            if m: out.append((m0[1],int(m0[2]),int(m[1]),int(m[3]),float(m[4]),float(m[6])))
    return list(dict.fromkeys(out))
def cls(p):
    if 2050<=p%4096<=2057: return "E"
    if 2<=p%256<=9 and p>256 and p%4096>256: return "C"
    return "N"
name={"E":"exposed","C":"peeled, not exposed","N":"control (no peel effect)"}
for d in sys.argv[1:]:
    R=rows(d)
    for size in sorted({r[2] for r in R}):
        for c in "ECN":
            a=[r for r in R if r[2]==size and r[0]=="main" and cls(r[3])==c]; b=[r for r in R if r[2]==size and r[0]=="peel" and cls(r[3])==c]
            if not(a and b): continue
            ta,tb=st.median(r[4] for r in a),st.median(r[4] for r in b); pa,pb=st.median(r[3]/r[4] for r in a),st.median(r[3]/r[4] for r in b)
            lo=min(r[3] for r in a+b); hi=max(r[3] for r in a+b)
            print(f"| size {size}, {name[c]} | {lo}..{hi} | {len(a)} / {len(b)} | {ta:.3f} -> {tb:.3f} s ({tb-ta:+.3f} s, {tb/ta-1:+.1%}) | {pa:.0f} -> {pb:.0f} ({pb/pa-1:+.1%}) |")
