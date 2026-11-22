"""Split the ConvMix row-kernel dispatches of a per-dispatch trace (disp.py's input) by call type: followed by the blend
stages (#1-#3), or alone (with the ops around it). usage: calltype.py TRACE"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:])][2:]
n = len(segs)
pop = collections.defaultdict(list)
for s in segs:
    for k, r in enumerate(s):
        if r["op"] == "conv_mix_bf16":
            nxt = s[k + 1]["op"] if k + 1 < len(s) else ""
            prv = s[k - 1]["op"] if k else ""
            kind = "with blend" if nxt == "conv_mix_bf16#1" else f"alone, prev {prv.split('#')[0]}, next {nxt.split('#')[0]}"
            pop[kind].append((r["e"] - r["s"]) * 1e6)
for k, v in sorted(pop.items(), key=lambda kv: -sum(kv[1])):
    print(f"{k}: {len(v)/n:.1f} per token, median {st.median(v):.2f} us, {sum(v)/n:.0f} us/token")
