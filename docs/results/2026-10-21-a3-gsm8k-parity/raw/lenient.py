import json, re, sys
def last_num(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return m[-1].replace(",", "").rstrip(".") if m else None
for f in sys.argv[1:]:
    rs = list(map(json.loads, open(f)))
    ex = sum(r["score"] for r in rs)
    len_ = [r for r in rs if r["ct"] > 20]
    ln = sum(last_num(r["text"]) == r["expected"].replace(",", "") for r in rs)
    lnl = sum(last_num(r["text"]) == r["expected"].replace(",", "") for r in len_)
    sh = [r for r in rs if r["ct"] <= 20]
    print(f, "exact", ex, "lenient", ln, "| long n", len(len_), "long lenient-correct", lnl, "| short n", len(sh), "short exact", sum(r['score'] for r in sh))
