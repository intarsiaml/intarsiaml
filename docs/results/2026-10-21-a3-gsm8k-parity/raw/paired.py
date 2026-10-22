import json, re, sys, math
def last_num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t); return m[-1].replace(",", "").rstrip(".") if m else None
a = {r["i"]: r for r in map(json.loads, open(sys.argv[1]))}; b = {r["i"]: r for r in map(json.loads, open(sys.argv[2]))}
ks = sorted(set(a) & set(b)); n = len(ks)
for name, f in [("exact (deepeval)", lambda r: r["score"]), ("last number", lambda r: int(last_num(r["text"]) == r["expected"].replace(",", "")))]:
    x = [f(a[k]) for k in ks]; y = [f(b[k]) for k in ks]
    d = [p - q for p, q in zip(x, y)]; md = sum(d) / n; sd = math.sqrt(sum((v - md) ** 2 for v in d) / (n - 1)); se = sd / math.sqrt(n)
    b01 = sum(1 for p, q in zip(x, y) if p < q); b10 = sum(1 for p, q in zip(x, y) if p > q)
    z = (b10 - b01) / math.sqrt(b10 + b01) if b10 + b01 else 0; pval = math.erfc(abs(z) / math.sqrt(2))
    print(f"{name}: A {sum(x)/n:.4f} B {sum(y)/n:.4f} diff {md:+.4f} [{md-1.96*se:+.4f}, {md+1.96*se:+.4f}] McNemar A-only {b10} B-only {b01} z {z:+.2f} p {pval:.3g}")
long = lambda r: r["ct"] > 20
print("long outputs A", sum(long(a[k]) for k in ks), "B", sum(long(b[k]) for k in ks), "| A long&B short", sum(long(a[k]) and not long(b[k]) for k in ks), "B long&A short", sum(long(b[k]) and not long(a[k]) for k in ks))
for nm, d in [("A", a), ("B", b)]:
    L = [k for k in ks if long(d[k])]; S_ = [k for k in ks if not long(d[k])]
    print(nm, "long n", len(L), "last-number correct", sum(last_num(d[k]["text"]) == d[k]["expected"].replace(",", "") for k in L), "| short n", len(S_), "exact correct", sum(d[k]["score"] for k in S_), "| finish", {f: sum(d[k]["finish"] == f for k in ks) for f in set(d[k]["finish"] for k in ks)})
