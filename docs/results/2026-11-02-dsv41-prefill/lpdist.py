#!/usr/bin/env python3
"""Distances between runs of the same prompts, from completions with logprobs=5 (16 greedy tokens).

Usage: lpdist.py REF=DIR[:PREFIX] LABEL=DIR[:PREFIX] ...   (the first is the reference; PREFIX is the file prefix: longp- (run-full.sh), lp- (run-tflp.sh))

Per prompt and run: the first token's top-5 {token: logprob}; against the reference, over the tokens in both lists, the mean absolute logprob
difference, whether the argmax is the same, how many of the 16 greedy tokens agree from the start. Printed per prompt and as a mean over the
prompts. Token strings are compared as strings (TensorFold and ours tokenise alike for the same text).
"""
import glob
import json
import os
import statistics as st
import sys


def load(spec):
    d, _, pre = spec.partition(":")
    pre = pre or "longp-"
    out = {}
    for f in sorted(glob.glob(os.path.join(d, pre + "*.json"))):
        name = os.path.basename(f)[len(pre):-5]
        try:
            c = json.load(open(f))["choices"][0]
            lp = c["logprobs"]
            out[name] = (lp["tokens"], lp["token_logprobs"], lp["top_logprobs"][0], c["text"])
        except Exception:
            pass
    return out


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    runs = [(l, load(s)) for l, s in specs]
    rl, ref = runs[0]
    print(f"reference {rl}: {len(ref)} prompts")
    for label, run in runs[1:]:
        diffs, agree, lead, d0 = [], 0, [], []
        names = sorted(set(ref) & set(run))
        for n in names:
            rt, rlp, rtop, _ = ref[n]
            t, lp, top, _ = run[n]
            common = set(rtop) & set(top)
            d = st.mean(abs(rtop[k] - top[k]) for k in common) if common else float("nan")
            diffs.append(d)
            agree += max(rtop, key=rtop.get) == max(top, key=top.get)
            m = 0
            for x, y in zip(rt, t):
                if x != y:
                    break
                m += 1
            lead.append(m)
            d0.append(abs(rlp[0] - lp[0]))
        if not names:
            print(f"{label}: no common prompts")
            continue
        print(f"{label:<28} n={len(names)}  mean |dlogprob| of top-5 {st.mean(diffs):.4f} (max {max(diffs):.3f})  first-token |dlogprob| {st.mean(d0):.4f}  "
              f"argmax equal {agree}/{len(names)}  greedy tokens agreeing from the start: mean {st.mean(lead):.1f} of 16, all 16: {sum(1 for x in lead if x == 16)}/{len(names)}")


if __name__ == "__main__":
    main()
