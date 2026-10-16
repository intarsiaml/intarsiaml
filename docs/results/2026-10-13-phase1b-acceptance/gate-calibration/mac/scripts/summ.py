import re,sys,glob,os
S="/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/mac"
for f in sorted(glob.glob(S+"/gate-*.log")):
    t=open(f).read().splitlines()
    mode=None; agg={}; pp={}; ks={}
    for l in t:
        m=re.match(r"baseline gate, (\w+)",l)
        if m: mode=m.group(1)
        if mode and re.match(r"  (top-1 %|mean KL|tail-bucket KL|top-5 %|\|tail excess\||mean \|tail diff\||max KL\(21-way\)|non-finite rows) ",l):
            if l.rstrip().endswith("FAIL"): agg.setdefault(mode,[]).append(l.split("  ")[1].strip())
        m=re.match(r"  per-prompt gate: (.*)",l)
        if m and mode: pp[mode]=m.group(1)[:80]
        m=re.match(r"  failing per-prompt checks by k.*: (.*)",l)
        if m and mode:
            ks[mode]=" ".join(re.sub(r"k=([\d.]+): (\d+) prompt\(s\) / (\d+) check\(s\)(?: <- used)?",r"k\1:\2p/\3c",x.strip()) for x in m.group(1).split(", "))
    print(os.path.basename(f))
    for mo in ("prefill","decode"):
        print(f"   {mo:8s} aggregate FAIL: {agg.get(mo,[])}  per-prompt: {pp.get(mo)}  | {ks.get(mo,'')}")
