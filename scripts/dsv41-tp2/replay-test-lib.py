#!/usr/bin/env python3
"""Helpers of scripts/dsv41-tp2/replay-test.sh (bounded-replay prefill tests). Run with
   uv run --no-project --with numpy --with safetensors python replay-test-lib.py <cmd> ...
Commands (each prints a table; the exit code is 1 when a stated requirement fails):
  tokcmp A.st B.st [--name-a X --name-b Y] [--require-identical] [--require-total N] [--max-prompt N]
      two intarsia-golden-tokens-v1 files (tools/dsv41/golden_tf.py tokens): per prompt the leading-match length and the
      count of equal positions; --require-identical: every prompt same prompt ids and same continuation; --require-total:
      A holds exactly N continuation tokens in all; --max-prompt: every prompt of both files has at most N prompt ids.
  speedtab OUT.md cell... (cell = chunk:cfg:order:result.json of `localeval speed`)
      prefill seconds (TTFT median) per cell and the on/off ratios of the ABAB passes.
  sha256 FILE...                         print sha256 and name
  checksha SHA.json DIR PREFIX KEY...    each DIR/PREFIX+KEY+.jsonl against SHA.json[KEY]; exit 1 on any mismatch
  quality MANIFEST.json PAIRS.txt OUT.md paired off/on accuracy; PAIRS.txt lines: `shard off_result.json on_result.json`
"""
import hashlib
import json
import math
import os
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tokens(path):
    from safetensors import safe_open
    f = safe_open(path, "np")
    m = f.metadata() or {}
    if not str(m.get("format", "")).startswith("intarsia-golden-tokens"):
        raise SystemExit(f"{path}: format {m.get('format')!r}, want intarsia-golden-tokens-v1")
    names = [p["name"] for p in json.loads(m["prompts"])]
    return {nm: (f.get_tensor(f"ids_{i}"), f.get_tensor(f"cont_{i}")) for i, nm in enumerate(names)}


def tokcmp(argv):
    import numpy as np
    opts = {"--name-a": "A", "--name-b": "B", "--require-total": None, "--max-prompt": None}
    flags = {"--require-identical": False}
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        elif argv[i] in flags:
            flags[argv[i]] = True
            i += 1
        else:
            pos.append(argv[i])
            i += 1
    A, B = load_tokens(pos[0]), load_tokens(pos[1])
    na, nb = opts["--name-a"], opts["--name-b"]
    bad = []
    if set(A) != set(B):
        bad.append(f"prompt sets differ: only in {na}: {sorted(set(A) - set(B))}, only in {nb}: {sorted(set(B) - set(A))}")
    print(f"{'prompt':<18} {'n_prompt':>8} {'cont ' + na + '/' + nb:>12} {'leading':>8} {'equal':>8}  identical")
    tot_a = tot_lead = tot_eq = tot_n = 0
    same_prompts = 0
    for nm in A:
        if nm not in B:
            continue
        (ia, ca), (ib, cb) = A[nm], B[nm]
        if opts["--max-prompt"] is not None:
            for who, ids in ((na, ia), (nb, ib)):
                if len(ids) > int(opts["--max-prompt"]):
                    bad.append(f"{nm}: {who} prompt has {len(ids)} ids, more than {opts['--max-prompt']}")
        if not np.array_equal(ia, ib):
            print(f"{nm:<18} prompt ids differ ({len(ia)} vs {len(ib)}): not comparable")
            bad.append(f"{nm}: prompt ids differ")
            continue
        n = min(len(ca), len(cb))
        neq = np.nonzero(ca[:n] != cb[:n])[0]
        lead = int(neq[0]) if len(neq) else n
        eq = int((ca[:n] == cb[:n]).sum())
        ident = len(ca) == len(cb) and lead == n
        print(f"{nm:<18} {len(ia):>8} {len(ca):>5}/{len(cb):<6} {lead:>4}/{n:<3} {eq:>4}/{n:<3}  {'yes' if ident else 'NO'}")
        tot_a += len(ca)
        tot_lead += lead
        tot_eq += eq
        tot_n += n
        same_prompts += ident
        if not ident:
            bad.append(f"{nm}: continuations differ (first difference at token {lead})")
    print(f"TOTAL {na} vs {nb}: leading-match {tot_lead}/{tot_n} ({100 * tot_lead / max(tot_n, 1):.2f}%), equal positions {tot_eq}/{tot_n} "
          f"({100 * tot_eq / max(tot_n, 1):.2f}%), identical prompts {same_prompts}/{len(A)}; {na} holds {tot_a} continuation tokens")
    if opts["--require-total"] is not None and tot_a != int(opts["--require-total"]):
        bad.append(f"{na} holds {tot_a} continuation tokens, wanted {opts['--require-total']}")
    if opts["--require-total"] is not None and tot_eq != int(opts["--require-total"]):
        bad.append(f"only {tot_eq} of {opts['--require-total']} tokens identical")
    if flags["--require-identical"] or opts["--require-total"] is not None or opts["--max-prompt"] is not None:
        if bad:
            for b in bad:
                print("REQUIREMENT FAILED:", b)
            return 1
        print("REQUIREMENT MET")
    return 0


def speedtab(argv):
    out, cells = argv[0], argv[1:]
    rows = {}
    for c in cells:
        chunk, cfg, order, path = c.split(":", 3)
        try:
            sp = json.load(open(path))["speed"]
        except Exception as e:  # noqa: BLE001
            print(f"cannot read {path}: {e}")
            continue
        for r in sp["rows"]:
            rows[(int(chunk), int(r["target_prompt_tokens"]), cfg, int(order))] = (
                r["prompt_tokens"], r["ttft_s"]["median"], r["prefill_tok_s"]["median"], r.get("errors"))
    L = ["| chunk | prompt tokens | pass | off prefill s (tok/s) | on prefill s (tok/s) | on / off |", "|---:|---:|---:|---|---|---:|"]
    summ = []
    for chunk, size in sorted({(k[0], k[1]) for k in rows}):
        offs, ons = [], []
        for order in sorted({k[3] for k in rows if k[0] == chunk and k[1] == size}):
            o, n = rows.get((chunk, size, "off", order)), rows.get((chunk, size, "on", order))
            f = lambda x: f"{x[1]:.2f} ({x[2]:.0f})" if x else "-"  # noqa: E731
            ratio = f"{n[1] / o[1]:.3f}" if o and n else "-"
            L.append(f"| {chunk} | {(o or n)[0]} | {order} | {f(o)} | {f(n)} | {ratio} |")
            if o:
                offs.append(o[1])
            if n:
                ons.append(n[1])
        if offs and ons:
            mo, mn = sum(offs) / len(offs), sum(ons) / len(ons)
            summ.append(f"chunk {chunk}, {size} tokens: mean off {mo:.2f} s, mean on {mn:.2f} s, on/off {mn / mo:.3f} ({len(offs)} off, {len(ons)} on cells)")
    text = "\n".join(L) + "\n\n" + "\n".join(summ) + "\n"
    open(out, "w").write(text)
    print(text)
    return 0


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def quality(argv):
    man = {m["name"]: m for m in json.load(open(argv[0]))["items"]}
    pairs = [l.split() for l in open(argv[1]) if l.strip()]
    off, on = {}, {}
    errs = 0
    for sh, po, pn in pairs:
        for d, p in ((off, po), (on, pn)):
            for ds in json.load(open(p))["datasets"]:
                for c in ds["cases"]:
                    if c.get("error"):
                        errs += 1
                        continue
                    d[c["name"]] = c
    names = sorted(set(off) & set(on), key=lambda n: man[n]["idx"])
    n = len(names)
    L = [f"Paired off / on: {n} items scored in both arms ({len(pairs)} shards; {errs} cases with a call error left out; "
         f"{len(set(off) ^ set(on))} items scored in one arm only)."]
    if not n:
        open(argv[2], "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return 1

    def tab(sub):
        a = sum(bool(off[x]["success"]) and bool(on[x]["success"]) for x in sub)
        b = sum(bool(off[x]["success"]) and not on[x]["success"] for x in sub)
        c = sum(bool(on[x]["success"]) and not off[x]["success"] for x in sub)
        return a, b, c, len(sub) - a - b - c

    def row(label, sub):
        a, b, c, d = tab(sub)
        k = len(sub)
        return f"| {label} | {k} | {a + b} ({(a + b) / k:.3f}) | {a + c} ({(a + c) / k:.3f}) | {b} | {c} | {mcnemar_p(b, c):.3f} |"

    head = "| {} | items | off correct | on correct | off right, on wrong | on right, off wrong | McNemar exact p |\n|---|---:|---:|---:|---:|---:|---:|"
    L.append("\n**Overall**\n")
    L.append(head.format("set"))
    L.append(row("all", names))

    # The frozen set (chunk32k-2) puts facts across the whole document: every item has a fact in the first 0.5 and one near the end,
    # so there is no item whose facts are ALL early. Position is therefore split by where the EARLIEST fact sits (the one
    # furthest from the last 128 rows), where the LATEST fact sits, and by how many facts lie in the first third.
    splits = (
        ("depth of the EARLIEST fact (fraction of the document)", lambda m: min(m["depths"]),
         (("< 0.10", lambda v: v < 0.10), ("0.10 to 0.20", lambda v: 0.10 <= v < 0.20), (">= 0.20", lambda v: v >= 0.20))),
        ("depth of the LATEST fact", lambda m: max(m["depths"]),
         (("< 0.90 (answer needs nothing near the end)", lambda v: v < 0.90), (">= 0.90", lambda v: v >= 0.90))),
        ("number of facts in the first third of the document", lambda m: sum(d < 1 / 3 for d in m["depths"]),
         (("0 to 1", lambda v: v <= 1), ("2 to 3", lambda v: 2 <= v <= 3), ("4 or more", lambda v: v >= 4))),
    )
    for title, key, bins in splits:
        L.append(f"\n**By {title}**\n")
        L.append(head.format("bin"))
        for lab, f in bins:
            sub = [x for x in names if f(key(man[x]))]
            L.append(row(lab, sub) if sub else f"| {lab} | 0 | - | - | - | - | - |")
    L.append(f"\nNo item has all its facts early: smallest latest-fact depth in the scored items {min(max(man[x]['depths']) for x in names):.3f}.")
    L.append("\n**By kind**\n")
    L.append(head.format("kind"))
    for k in ("hop2", "hop3", "agg_sum", "agg_max", "repo_sum"):
        sub = [x for x in names if man[x]["kind"] == k]
        if sub:
            L.append(row(k, sub))
    L.append("\n**By size**\n")
    L.append(head.format("size"))
    for k in (32, 65):
        sub = [x for x in names if man[x]["k"] == k]
        if sub:
            L.append(row(f"{k}K", sub))
    disc = [x for x in names if bool(off[x]["success"]) != bool(on[x]["success"])]
    L.append("\nDiscordant items: " + (", ".join(f"`{x}` (depths {man[x]['depths']}; off {'right' if off[x]['success'] else 'wrong'}, "
                                                  f"on {'right' if on[x]['success'] else 'wrong'})" for x in disc) or "none"))
    same = sum((off[x].get("actual_output") or "").strip() == (on[x].get("actual_output") or "").strip() for x in names)
    L.append(f"\nReply text identical off vs on for {same} of {n} items.")
    open(argv[2], "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def checksha(argv):
    want = json.load(open(argv[0]))
    d, prefix, keys = argv[1], argv[2], argv[3:]
    bad = 0
    for k in keys:
        p = os.path.join(d, f"{prefix}{k}.jsonl")
        got = sha(p) if os.path.exists(p) else "MISSING"
        ok = got == want.get(k)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} {k} {got[:16]} {'==' if ok else '!='} {str(want.get(k))[:16]}  {p}")
    return 1 if bad else 0


def main():
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "sha256":
        for p in a:
            print(sha(p), os.path.basename(p))
        return 0
    return {"tokcmp": tokcmp, "speedtab": speedtab, "quality": quality, "checksha": checksha}[cmd](a)


if __name__ == "__main__":
    sys.exit(main())
