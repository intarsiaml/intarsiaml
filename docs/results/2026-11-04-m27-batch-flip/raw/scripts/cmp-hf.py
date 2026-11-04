#!/usr/bin/env python3
"""HF f32 (activations f32) as reference on the solo stream, against every path on rows sharing context with that stream."""
import json, math, statistics as st, os
from safetensors.numpy import load_file
def rm(r): return {int(x[0]): x[1] for x in r}
O = ".scratch/logs/oracle/"
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
A = json.load(open(".scratch/logs/sweep/all.json")); LEN["772"] = {"solo": A["q4-solo"], "behind": A["q4-behind"]}
P = json.load(open(O + "perturb.json")); F = json.load(open(O + "ref-free.json")); C = json.load(open(O + "f32-free.json"))
H32 = load_file(O + "hf-float32.safetensors"); H16 = load_file(O + "hf-bfloat16.safetensors")
order = list(json.load(open(O + "hf-seqs.json")))
def hrow(H, i, plen, t):
    return {int(a): float(b) for a, b in zip(H[f"topk_ids_{i}"][plen - 1 + t], H[f"topk_logprobs_{i}"][plen - 1 + t])}
def kl(x, y):
    return sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys())
agg = {}
for i, k in enumerate(order):
    plen = int(k); S = LEN[k]["solo"]; n = 200
    print(f"plen {k}")
    paths = {"solo-NAX": S, "busy": LEN[k]["behind"], "allGEMV": P.get(f"nax0/{k}"), "tail3": P.get(f"tail3/{k}"), "tail5": P.get(f"tail5/{k}"), "tail7": P.get(f"tail7/{k}"), "f64-ref": F.get(k), "cpu-f32": C.get(k)}
    R = [hrow(H32, i, plen, t) for t in range(n)]; B = [hrow(H16, i, plen, t) for t in range(n)]
    def score(v):
        K, D = [], []
        for t in range(min(n, len(v["top"]))):
            if v["ids"][:t] != S["ids"][:t]: break
            y = rm(v["top"][t]); K.append(kl(R[t], y)); top = sorted(R[t], key=R[t].get, reverse=True)[:3]; D.append(max(abs(R[t][j] - y[j]) for j in top if j in y))
        return K, D
    K = [kl(R[t], B[t]) for t in range(n)]; D = [max(abs(R[t][j] - B[t][j]) for j in sorted(R[t], key=R[t].get, reverse=True)[:3] if j in B[t]) for t in range(n)]
    print(f"   {'HF-bf16':9} rows {n:3d} KL median {st.median(K):.1e} mean {st.mean(K):.1e} max {max(K):.3f}@{K.index(max(K))} dlp>1 rows {sum(d > 1 for d in D):3d} row0 dlp {D[0]:.4f} 50-row medians {[round(st.median(D[l:l+50]), 3) for l in range(0, n, 50)]}")
    agg.setdefault("HF-bf16", []).append((st.median(K), st.mean(K)))
    for name, v in paths.items():
        if not v: continue
        K, D = score(v)
        if not K: continue
        print(f"   {name:9} rows {len(K):3d} KL median {st.median(K):.1e} mean {st.mean(K):.1e} max {max(K):.3f}@{K.index(max(K))} dlp>1 rows {sum(d > 1 for d in D):3d} row0 dlp {D[0]:.4f} 50-row medians {[round(st.median(D[l:l+50]), 3) for l in range(0, len(D), 50)]}")
        agg.setdefault(name, []).append((st.median(K), st.mean(K)))
    rows = {"258": [100, 104, 105], "260": [138, 197], "262": [188], "772": [184], "1028": [63, 129]}[k]
    for r in rows:
        ids = sorted(rm(S["top"][r]), key=rm(S["top"][r]).get, reverse=True)[:2]
        if k == "772": ids = [14663, 557]
        print(f"   row {r}: HF-f32 top {max(R[r], key=R[r].get)} " + " | ".join(f"[{j}] f32 {R[r].get(j, float('nan')):.4f} hf-bf16 {B[r].get(j, float('nan')):.4f} solo {rm(S['top'][r]).get(j, float('nan')):.4f} busy {rm(LEN[k]['behind']['top'][r]).get(j, float('nan')):.4f}" for j in ids))
print("\nmedian over prompts of per-prompt median and mean KL(HF f32 || path)")
for n, v in agg.items(): print(f"   {n:9} prompts {len(v)} median-of-medians {st.median(x[0] for x in v):.1e} median-of-means {st.median(x[1] for x in v):.1e}")
