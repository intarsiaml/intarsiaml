#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Same-slot A/B regression guard for `intarsia serve --max-batch 1` (docs/results/2026-10-15-metal-cb-soak-harness.md, SLO 1).

  uv run tools/cbsoak_ab.py --base DIR1 DIR2 --cand DIR3 DIR4 [--max-ratio 1.1] [--out report.md]

Each DIR is a cbsoak output dir (raw.json inside, or in DIR/soak/). Runs of one variant are pooled (all ITL gaps of all their
requests). Prints per-run and pooled ITL p50/p95/p99, TTFT, decode tok/s, and the candidate/baseline ratio of ITL p50 and p95;
exit 1 when a ratio exceeds --max-ratio. Per-run values are shown so slot drift between interleaved runs is visible.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cbsoak import percentile  # noqa: E402


def load(d):
    d = pathlib.Path(d)
    for c in (d / "raw.json", d / "soak" / "raw.json"):
        if c.exists():
            return json.loads(c.read_text())
    raise SystemExit(f"no raw.json under {d}")


def stats(raws):
    itl = [g for raw in raws for r in raw["requests"] for g in (r.get("itl_ms") or [])]
    ttft = [r["ttft_ms"] for raw in raws for r in raw["requests"] if r.get("ttft_ms") is not None]
    toks = []
    for raw in raws:
        for r in raw["requests"]:
            if r.get("status") == "ok" and r.get("itl_ms"):
                toks.append(len(r["itl_ms"]) / (sum(r["itl_ms"]) / 1000.0))
    return {"n_itl": len(itl), "itl_p50": percentile(itl, 50), "itl_p95": percentile(itl, 95), "itl_p99": percentile(itl, 99),
            "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95), "decode_tok_s": sum(toks) / len(toks) if toks else None,
            "n_req": sum(len(raw["requests"]) for raw in raws)}


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--max-ratio", type=float, default=1.1)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    L = ["| run | variant | requests | ITL n | ITL p50 | ITL p95 | ITL p99 | TTFT p50 | TTFT p95 | decode tok/s |", "|---|---|---|---|---|---|---|---|---|---|"]
    pools = {}
    for var, dirs in (("base", a.base), ("cand", a.cand)):
        raws = [load(d) for d in dirs]
        pools[var] = raws
        for d, raw in zip(dirs, raws):
            s = stats([raw])
            L.append(f"| {pathlib.Path(d).name} | {var} | {s['n_req']} | {s['n_itl']} | {fmt(s['itl_p50'])} | {fmt(s['itl_p95'])} | {fmt(s['itl_p99'])} | {fmt(s['ttft_p50'], 0)} | {fmt(s['ttft_p95'], 0)} | {fmt(s['decode_tok_s'])} |")
    sb, sc = stats(pools["base"]), stats(pools["cand"])
    for var, s in (("base pooled", sb), ("cand pooled", sc)):
        L.append(f"| **{var}** | | {s['n_req']} | {s['n_itl']} | {fmt(s['itl_p50'])} | {fmt(s['itl_p95'])} | {fmt(s['itl_p99'])} | {fmt(s['ttft_p50'], 0)} | {fmt(s['ttft_p95'], 0)} | {fmt(s['decode_tok_s'])} |")
    viol = []
    L += [""]
    for k in ("itl_p50", "itl_p95"):
        r = sc[k] / sb[k] if sb[k] and sc[k] else None
        L.append(f"{k}: cand/base = {fmt(r, 3)} (limit {a.max_ratio})")
        if r is None or r > a.max_ratio:
            viol.append(k)
    L.append("RESULT: " + ("PASS" if not viol else "FAIL " + ",".join(viol)))
    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        pathlib.Path(a.out).write_text(out)
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
