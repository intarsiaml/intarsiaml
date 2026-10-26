#!/usr/bin/env python3
"""The tp>1 accuracy gate, check 3 (external reference). Rule as decided by the coordinator, 2026-10-25 (docs/decisions/LOG.md):

    tp-gate.py --ref acc-tp1.out --cand acc-tp2.out [--noise ref_noise.json] [--mutant NAME=acc-bug.out ...] [--control acc-variant.out ...]

Inputs are the stdout of `intarsia accuracy --golden <vLLM top-20 golden> --mode both` (a prefill table, then a decode table).
  --ref   the same build at tp=1 (reference = KL(tp1 vs vLLM))
  --cand  the build at tp>1       (candidate = KL(tp2 vs vLLM))
GATE (exit code): the mean 21-way KL against vLLM, per mode (prefill, decode):  cand_overall <= ref_overall * MEAN_TOL (1.02).
REPORT ONLY (never gating): per prompt, the ratio cand/ref next to the vLLM re-run noise of that prompt (--noise, the golden's
reference-noise record; prefill, all-rows values; no per-prompt noise exists for decode) and (cand - ref) / noise. Per-prompt noise of
the tp=1 build itself is unmeasured (the two tp=1 control variants, no fusion and chunk 233, were bitwise identical to the reference),
so no per-prompt cap is set. --control files are printed the same way. --mutant files (deliberate partial-scale bugs at tp>1) are
scored on the mean only, to show whether check 3 detects them (it does not reliably: LOG.md).
KL(tp vs tp1) is reported by `accuracy --self-golden --self-vs-ref-noise`, not here.
"""
import argparse, json, re, sys

MEAN_TOL = 1.02


def table(path):
    """{(mode, prompt): KL21 scored}; the first 'prompt n pos ...' header starts prefill, the second decode."""
    out, mode, seen = {}, None, 0
    for line in open(path, errors="replace"):
        if line.startswith("prompt "):
            seen += 1
            mode = "prefill" if seen == 1 else "decode"
            continue
        parts = line.split()
        if not parts or not mode:
            continue
        try:
            # prompt rows: name n pos top1 klsupp kl21way kl21scored ... ; OVERALL rows: OVERALL n top1 klsupp kl21way kl21scored
            v = float(parts[5]) if parts[0] == "OVERALL" else float(parts[6])
            int(parts[1])
        except (ValueError, IndexError):
            continue
        out[(mode, parts[0])] = v
    return out


def noise_of(path):
    if not path:
        return {}
    j = json.load(open(path))
    return {p: v["kl21_mean"] for p, v in (j.get("prompts") or {}).items()}


def mean_gate(ref, cand):
    rows, ok = [], True
    for mode in ("prefill", "decode"):
        r, c = ref.get((mode, "OVERALL")), cand.get((mode, "OVERALL"))
        if r is None or c is None:
            sys.exit(f"no OVERALL row for {mode}")
        ratio = c / r
        rows.append((mode, r, c, ratio, ratio <= MEAN_TOL))
        ok &= ratio <= MEAN_TOL
    return rows, ok


def show_mean(title, rows, ok):
    print(f"\n== {title}: {'PASS' if ok else 'FAIL'}   (gate: mean KL ratio <= {MEAN_TOL})")
    print(f"{'mode':8}{'ref':>11}{'cand':>11}{'ratio':>8}")
    for mode, r, c, ratio, good in rows:
        print(f"{mode:8}{r:11.3e}{c:11.3e}{ratio:8.3f}{'' if good else '  FAIL'}")


def show_prompts(title, ref, cand, noise):
    print(f"\n-- per prompt, REPORT ONLY (no gate): {title}")
    print(f"{'mode':8}{'prompt':18}{'ref':>10}{'cand':>10}{'ratio':>8}{'vLLM noise':>12}{'(c-r)/noise':>13}")
    for (mode, p), r in ref.items():
        if p == "OVERALL" or (mode, p) not in cand:
            continue
        c = cand[(mode, p)]
        n = noise.get(p) if mode == "prefill" else None
        print(f"{mode:8}{p:18}{r:10.3e}{c:10.3e}{c / r if r > 0 else float('inf'):8.3f}"
              f"{(f'{n:12.3e}' if n else '           -')}{(f'{(c - r) / n:13.2f}' if n else '            -')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--noise", help="the golden's reference-noise json (golden/ref_noise/*.json)")
    ap.add_argument("--control", action="append", default=[], help="further tp=1 variants, printed only")
    ap.add_argument("--mutant", action="append", default=[], help="NAME=path, scored on the mean only")
    a = ap.parse_args()
    ref, noise = table(a.ref), noise_of(a.noise)
    if not ref:
        sys.exit(f"no table in {a.ref}")
    cand = table(a.cand)
    rows, ok = mean_gate(ref, cand)
    show_mean(f"check 3: {a.cand} against {a.ref}", rows, ok)
    show_prompts(a.cand, ref, cand, noise)
    for c in a.control:
        t = table(c)
        r, _ = mean_gate(ref, t)
        show_mean(f"control (not gating): {c}", r, True)
        show_prompts(c, ref, t, noise)
    for m in a.mutant:
        name, path = m.split("=", 1)
        r, g = mean_gate(ref, table(path))
        show_mean(f"mutant {name}: {'detected' if not g else 'NOT detected'} by the mean gate", r, g)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
