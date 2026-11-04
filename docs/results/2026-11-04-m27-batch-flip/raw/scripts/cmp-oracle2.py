#!/usr/bin/env python3
import json, math, statistics as st
def rm(r): return {int(x[0]): x[1] for x in r}
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
F = json.load(open(".scratch/logs/oracle/ref-free.json")); T = json.load(open(".scratch/logs/oracle/ref-forced.json")); C = json.load(open(".scratch/logs/oracle/f32-free.json")); P = json.load(open(".scratch/logs/oracle/perturb.json"))
def kl(x, y): return sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys())
print("== f64 teacher-forced one pass vs f64 free-run at the same row (reference self-consistency)")
a = rm(T["260@solo197"]["top"][0]); b = rm(F["260"]["top"][197]); print("260 row 197 forced top", max(a, key=a.get), round(max(a.values()), 4), "free top", max(b, key=b.get), round(max(b.values()), 4), "KL", f"{kl(a, b):.2e}", "max |dlp| top5", round(max(abs(a[i] - b[i]) for i in sorted(a, key=a.get, reverse=True)[:5] if i in b), 4))
print("== reference at swing rows, same context as the solo stream")
for key, k, r in [("262@solo188", "262", 188), ("1028@solo63", "1028", 63), ("260@solo197", "260", 197)]:
    o = rm(T[key]["top"][0]); s = LEN[k]["solo"]; b = LEN[k]["behind"]; top = sorted(rm(s["top"][r]), key=rm(s["top"][r]).get, reverse=True)[:2]
    print(k, "row", r, "ref top", max(o, key=o.get), " ".join(f"[{i}] ref {o.get(i, float('nan')):.4f} solo {rm(s['top'][r]).get(i, float('nan')):.4f} busy {rm(b['top'][r]).get(i, float('nan')):.4f}" for i in top))
print("== CPU f32 kernels vs f64 reference: a pure precision perturbation, no Metal, no GEMV / NAX split")
for k in ["258", "260", "1028"]:
    c, f = C[k], F[k]; d = next((i for i, (x, y) in enumerate(zip(c["ids"], f["ids"])) if x != y), None); m = (d + 1) if d is not None else len(f["ids"])
    K = [kl(rm(f["top"][t]), rm(c["top"][t])) for t in range(m)]
    D = []
    for t in range(m):
        x, y = rm(f["top"][t]), rm(c["top"][t]); top = sorted(x, key=x.get, reverse=True)[:3]; D.append(max(abs(x[i] - y[i]) for i in top if i in y))
    S = LEN[k]["solo"]; ds = next((i for i, (x, y) in enumerate(zip(c["ids"], S["ids"])) if x != y), None)
    print(f"{k}: wall {c['wall']:.0f}s first id diff vs f64 {d}, vs Metal solo {ds}; rows {m}; KL median {st.median(K):.2e} mean {st.mean(K):.2e} max {max(K):.3f}@{K.index(max(K))}; top-3 dlp row0 {D[0]:.4f}, 50-row medians", [round(st.median(D[l:l + 50]), 3) for l in range(0, m, 50)], "rows with dlp>1:", sum(1 for x in D if x > 1))
print("== median KL(ref||path) per prompt, rows sharing context with the reference, and how each path ranks")
paths = lambda k: {"solo-NAX": LEN[k]["solo"], "busy": LEN[k]["behind"], "allGEMV": P.get(f"nax0/{k}"), "tail3": P.get(f"tail3/{k}"), "tail5": P.get(f"tail5/{k}"), "tail7": P.get(f"tail7/{k}"), "cpu-f32": C.get(k)}
for k in ["258", "260", "262", "1028"]:
    row = {}
    for n, v in paths(k).items():
        if not v: continue
        f = F[k]; d = next((i for i, (x, y) in enumerate(zip(f["ids"], v["ids"])) if x != y), None); m = min(d if d is not None else len(f["ids"]), len(v["top"]))
        if m == 0: continue
        K = [kl(rm(f["top"][t]), rm(v["top"][t])) for t in range(m)]; row[n] = (m, st.median(K), st.mean(K))
    print(k, "  ".join(f"{n}: rows {r[0]} med {r[1]:.1e} mean {r[2]:.1e}" for n, r in sorted(row.items(), key=lambda kv: kv[1][2])))
