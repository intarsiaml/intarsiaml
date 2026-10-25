#!/usr/bin/env python3
"""The tp>1 accuracy gate, check 3 (external reference), decided by the coordinator 2026-10-25 (docs/decisions/LOG.md).

    tp-gate.py --ref acc-tp1.out --cand acc-tp2.out [--control acc-tp1-variant.out ...] [--mutant NAME=acc-bug.out ...]

Inputs are the stdout of `intarsia accuracy --golden <vLLM top-20 golden> --mode both` (a prefill table, then a decode table).
  --ref   the same build at tp=1 (reference = KL(tp1 vs vLLM))
  --cand  the build at tp>1       (candidate = KL(tp2 vs vLLM))
Check 3, per mode (prefill, decode), on the "KL21 scored" column (mean 21-way KL against vLLM):
  * mean:        cand_overall <= ref_overall * MEAN_TOL      (1.02)
  * per prompt:  cand_prompt  <= ref_prompt  * cap(cell), cap(cell) = max(PROMPT_CAP, CONTROL_MARGIN * the worst control/ref ratio
                 in that same cell), cell = (mode, prompt), prefill and decode separately.   PROMPT_CAP 1.10, CONTROL_MARGIN 1.1
                 (coordinator, 2026-10-25, fixed before window w8 / before any control data existed; docs/decisions/LOG.md)
Exit 0 only if both hold for both modes. --control files are the two equally valid tp=1 variants (no fusion; chunk 233): each is a
full accuracy table of the same build at tp=1; their per-cell ratios to --ref set the cap above. They are printed, not gated.
With no control the cap is the fixed 1.10 (and the output says so). --mutant files
(deliberate partial-scale bugs at tp>1) are scored the same way and printed with their verdict, to answer "does check 3 kill it".
KL(tp vs tp1) is reported by `accuracy --self-golden --self-vs-ref-noise`, not here, and does not gate.
"""
import argparse, re, sys

MEAN_TOL, PROMPT_CAP, CONTROL_MARGIN = 1.02, 1.10, 1.1


def table(path):
    """{(mode, prompt): KL21 scored}; the first 'prompt n pos ...' header starts prefill, the second decode."""
    out, mode, seen = {}, None, 0
    for line in open(path, errors="replace"):
        if line.startswith("prompt "):
            seen += 1
            mode = "prefill" if seen == 1 else "decode"
            continue
        m = re.match(r"^(OVERALL|\S+)\s+(\d+)\s+(?:(\d+)\s+)?([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if m and mode:
            # prompt rows: name n pos top1 klsupp kl21way kl21scored ... ; OVERALL rows: OVERALL n top1 klsupp kl21way kl21scored
            parts = line.split()
            try:
                v = float(parts[6]) if parts[0] != "OVERALL" else float(parts[5])
            except ValueError:
                continue
            out[(mode, parts[0])] = v
    return out


def caps(ref, controls):
    """cap per cell: MEAN_TOL for OVERALL, else max(PROMPT_CAP, CONTROL_MARGIN * worst control/ref ratio in the cell)."""
    out = {}
    for (mode, p), r in ref.items():
        if p == "OVERALL":
            out[(mode, p)] = MEAN_TOL
            continue
        worst = max([c[(mode, p)] / r for c in controls if (mode, p) in c and r > 0] or [0.0])
        out[(mode, p)] = max(PROMPT_CAP, CONTROL_MARGIN * worst)
    return out


def score(ref, cand, cap):
    rows, ok = [], True
    for (mode, p), r in ref.items():
        c = cand.get((mode, p))
        if c is None:
            continue
        lim = cap[(mode, p)]
        ratio = c / r if r > 0 else float("inf")
        rows.append((mode, p, r, c, ratio, lim, ratio <= lim))
        ok &= ratio <= lim
    return rows, ok


def show(title, rows, ok):
    print(f"\n== {title}: {'PASS' if ok else 'FAIL'}")
    print(f"{'mode':8}{'prompt':18}{'ref':>10}{'cand':>10}{'ratio':>8}{'cap':>7}")
    for mode, p, r, c, ratio, lim, good in rows:
        print(f"{mode:8}{p:18}{r:10.3e}{c:10.3e}{ratio:8.3f}{lim:7.2f}{'' if good else '  FAIL'}")
    bad = [f"{m}/{p} ({ra:.2f} > {lim:.2f})" for m, p, _, _, ra, lim, g in rows if not g]
    print(f"failing: {', '.join(bad) if bad else 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--control", action="append", default=[])
    ap.add_argument("--mutant", action="append", default=[], help="NAME=path")
    a = ap.parse_args()
    ref = table(a.ref)
    if not ref:
        sys.exit(f"no table in {a.ref}")
    ctl = [table(c) for c in a.control]
    if len(ctl) < 2:
        print(f"NOTE: {len(ctl)} control table(s) given, the rule needs two; per-prompt cap falls back to {'the worst of those' if ctl else 'the fixed 1.10'}")
    cap = caps(ref, ctl)
    rows, ok = score(ref, table(a.cand), cap)
    show(f"check 3: {a.cand} against {a.ref}", rows, ok)
    for c, t in zip(a.control, ctl):
        r, _ = score(ref, t, cap)
        show(f"control (equally valid tp=1 variant; ratios here set the caps, never gating): {c}", r, True)
    for m in a.mutant:
        name, path = m.split("=", 1)
        r, g = score(ref, table(path), cap)
        show(f"mutant {name}: {'KILLED by check 3' if not g else 'NOT killed by check 3'}", r, g)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
