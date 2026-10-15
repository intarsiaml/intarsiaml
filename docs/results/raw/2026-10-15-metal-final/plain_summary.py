import re,glob,statistics as st,sys
d=sys.argv[1]
print("| prompt | build | chunk | n | prefill ms median (min-max) | tok/s median |\n|---|---|---|---|---|---|")
for p in ["1k","4k","16k"]:
    row={}
    for b in ["safe","cand"]:
        ms=[];tk=[];ch=None
        for f in sorted(glob.glob(f"{d}/{b}-{p}-*.err")):
            t=open(f).read(); m=re.findall(r"prefill (\d+) ms \(([\d.]+) tok/s\)",t)[-1]
            ms.append(int(m[0]));tk.append(float(m[1])); ch=re.search(r"prefill chunk (\d+)",t).group(1)
        row[b]=st.median(tk)
        print(f"| {p} | {b.upper()} | {ch} | {len(ms)} | {st.median(ms):.0f} ({min(ms)}-{max(ms)}) | {st.median(tk):.0f} |")
    print(f"| {p} | CAND/SAFE | | | | {row['cand']/row['safe']:.3f}x |")
