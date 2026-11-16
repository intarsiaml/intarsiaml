#!/usr/bin/env python3
"""Lane dsubset: the acceptance-per-script table from accept-<arm>.json files (accept.py).

accept_table.py <raw dir> <subset dir> [arms...]   (default arms: F P65536 P131072 S32768 S65536 Q65536 when present)

Per set and arm: acceptance (accepted / proposed drafts), tokens per verify pass, tok/s; and the reachable share: of the tokens
the full-head arm F generated for the set (greedy, so every arm generates the same ids), the share each arm's draft head has a row
for at all (a prefix: id < N; a subset: id in the file). Also checks that every arm's replies equal F's (drafting never changes ids).
"""
from __future__ import annotations

import json
import pathlib
import sys


def load_ids(path: pathlib.Path) -> set[int]:
    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, _, b = line.partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return ids


def main():
    raw, subdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    arms = sys.argv[3:] or [a for a in ["F", "P65536", "P131072", "S32768", "S65536", "Q65536"] if (raw / f"accept-{a}.json").exists()]
    res = {a: json.loads((raw / f"accept-{a}.json").read_text()) for a in arms}
    subsets = {int(f.stem.rsplit("-", 1)[1]): load_ids(f) for f in subdir.glob("*.txt")}

    def holds(arm: str):
        if arm == "F":
            return lambda i: True
        n = int(arm[1:])
        if arm[0] in "PQ":
            return lambda i: i < n
        s = subsets[n]
        return lambda i: i in s

    sets = list(res[arms[0]]["sets"])
    ref = res.get("F")
    print("| set | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for s in sets:
        cells = []
        for a in arms:
            r = res[a]["sets"].get(s)
            if r is None:
                cells.append("-")
                continue
            reach = ""
            if ref:
                toks = [i for rep in ref["sets"][s]["replies"] for i in rep["ids"]]
                h = holds(a)
                reach = f" / {100 * sum(1 for i in toks if h(i)) / max(1, len(toks)):.1f}%"
            cells.append(f"{r['accept_rate']:.3f} / {r['tokens_per_pass']:.2f} / {r['tok_s']:.1f}{reach}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print("\ncell: acceptance / tokens per verify pass / tok/s (one stream, greedy, incl. prefill) / share of the set's tokens the head can draft")
    if ref:
        for a in arms:
            same = all(res[a]["sets"][s]["replies"][k]["ids"] == ref["sets"][s]["replies"][k]["ids"] for s in sets if s in res[a]["sets"] for k in range(len(ref["sets"][s]["replies"])))
            print(f"replies of {a} identical to F's: {same}")
    if "Q65536" in res and "P65536" in res:
        keys = ("proposed", "accepted", "verify_passes")
        eq = all(res["Q65536"]["sets"][s][k] == res["P65536"]["sets"][s][k] for s in sets for k in keys)
        print(f"Q65536 (prefix as a subset file) counters == P65536's on every set: {eq}")
        for s in sets:
            print(f"  {s}: Q {[res['Q65536']['sets'][s][k] for k in keys]}  P {[res['P65536']['sets'][s][k] for k in keys]}")


if __name__ == "__main__":
    main()
