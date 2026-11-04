#!/usr/bin/env python3
"""Helpers of scripts/dsv41-tp2/replay-test.sh (bounded-replay prefill tests). Run with
   uv run --no-project --with numpy --with safetensors python replay-test-lib.py <cmd> ...
Commands (each prints a table; the exit code is 1 when a stated requirement fails):
  tokcmp A.st B.st [--name-a X --name-b Y] [--require-identical] [--require-total N] [--max-prompt N]
         [--prefix P] [--common] [--min-equal-pct P] [--same-lengths]
      two intarsia-golden-tokens-v1 files (tools/dsv41/golden_tf.py tokens): per prompt the leading-match length and the
      count of equal positions; --require-identical: every prompt same prompt ids and same continuation; --require-total:
      A holds exactly N continuation tokens in all; --max-prompt: every prompt of both files has at most N prompt ids.
      --prefix: only prompts whose name starts with P; --common: only prompts in both files; --min-equal-pct: the equal
      positions over all compared prompts are at least P percent (a tripwire against garbage, not a quality test);
      --same-lengths: every compared pair has the same number of continuation tokens (and at least one).
  tfagree TF.st OFF.st ON.st [--tol N]
      the step-4 criterion against TensorFold: on prompts of more than 128 ids (where the replay is active) ON's equal positions
      with TF are at least OFF's minus N (default 8 of 64); prompts of at most 128 ids must be identical off vs on.
  gatecmp OFF_DIR ON_DIR [--min-files N] [--absent P,P] [--present P,P] [--differ P,P]
      the encoder-half gate: dumps (INTARSIA_DSV41_DUMP) of layers 0-19 and layer 20's kv-source steps of an off run and an on run
      of the same request sequence must be byte-identical; the decoder half must be absent below the floors, present above,
      and different at the first replayed rows.
  speedtab OUT.md [--max-ratio R] [--expect-cells N] cell... (cell = chunk:cfg:order:result.json of `localeval speed`)
      prefill seconds (TTFT median) per cell and the on/off ratios of the ABAB passes; with --max-ratio the exit code is 1 when
      the mean on/off ratio of any (chunk, size) exceeds R, a cell has errors, or fewer than N cells were read.
  sha256 FILE...                         print sha256 and name
  checksha SHA.json DIR PREFIX KEY...    each DIR/PREFIX+KEY+.jsonl against SHA.json[KEY]; exit 1 on any mismatch
  quality MANIFEST.json PAIRS.txt OUT.md [--expect-items N] [--max-loss D] [--alpha A]
      paired off/on accuracy; PAIRS.txt lines: `shard off_result.json on_result.json`. Exit 1 when fewer than N items are scored in
      both arms, or on is worse than off by at least D (default 0.10) or by more than 0.05 with McNemar p < A (default 0.05).
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
    opts = {"--name-a": "A", "--name-b": "B", "--require-total": None, "--max-prompt": None, "--prefix": None, "--min-equal-pct": None}
    flags = {"--require-identical": False, "--common": False, "--same-lengths": False}
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
    if opts["--prefix"] is not None:
        A = {k: v for k, v in A.items() if k.startswith(opts["--prefix"])}
        B = {k: v for k, v in B.items() if k.startswith(opts["--prefix"])}
    if flags["--common"]:
        A = {k: v for k, v in A.items() if k in B}
        B = {k: v for k, v in B.items() if k in A}
    bad = []
    if not A or not B:
        bad.append("no prompts to compare")
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
        if flags["--same-lengths"] and (len(ca) != len(cb) or n == 0):
            bad.append(f"{nm}: continuation lengths {len(ca)} vs {len(cb)}")
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
        if not ident and (flags["--require-identical"] or opts["--require-total"] is not None):
            bad.append(f"{nm}: continuations differ (first difference at token {lead})")
    print(f"TOTAL {na} vs {nb}: leading-match {tot_lead}/{tot_n} ({100 * tot_lead / max(tot_n, 1):.2f}%), equal positions {tot_eq}/{tot_n} "
          f"({100 * tot_eq / max(tot_n, 1):.2f}%), identical prompts {same_prompts}/{len(A)}; {na} holds {tot_a} continuation tokens")
    if opts["--require-total"] is not None and tot_a != int(opts["--require-total"]):
        bad.append(f"{na} holds {tot_a} continuation tokens, wanted {opts['--require-total']}")
    if opts["--require-total"] is not None and tot_eq != int(opts["--require-total"]):
        bad.append(f"only {tot_eq} of {opts['--require-total']} tokens identical")
    if opts["--min-equal-pct"] is not None and 100 * tot_eq < float(opts["--min-equal-pct"]) * max(tot_n, 1):
        bad.append(f"equal positions {tot_eq}/{tot_n} ({100 * tot_eq / max(tot_n, 1):.1f}%) are below {opts['--min-equal-pct']}%")
    if (flags["--require-identical"] or opts["--require-total"] is not None or opts["--max-prompt"] is not None
            or opts["--min-equal-pct"] is not None or flags["--same-lengths"]):
        if bad:
            for b in bad:
                print("REQUIREMENT FAILED:", b)
            return 1
        print("REQUIREMENT MET")
    return 0


def speedtab(argv):
    out, rest = argv[0], argv[1:]
    max_ratio = expect = None
    cells = []
    i = 0
    while i < len(rest):
        if rest[i] == "--max-ratio":
            max_ratio, i = float(rest[i + 1]), i + 2
        elif rest[i] == "--expect-cells":
            expect, i = int(rest[i + 1]), i + 2
        else:
            cells.append(rest[i])
            i += 1
    rows = {}
    bad = []
    for c in cells:
        chunk, cfg, order, path = c.split(":", 3)
        try:
            sp = json.load(open(path))["speed"]
        except Exception as e:  # noqa: BLE001
            print(f"cannot read {path}: {e}")
            bad.append(f"cannot read {path}")
            continue
        for r in sp["rows"]:
            if r.get("errors"):
                bad.append(f"{path}: errors {r.get('errors')} at {r.get('target_prompt_tokens')} tokens")
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
            if max_ratio is not None and mn / mo > max_ratio:
                bad.append(f"chunk {chunk}, {size} tokens: on/off {mn / mo:.3f} is above {max_ratio}")
            if max(offs) > 1.15 * min(offs):
                summ.append(f"  NOTE chunk {chunk}, {size} tokens: the two off passes differ by {max(offs) / min(offs):.2f}x: the machine was not quiet")
        elif max_ratio is not None:
            bad.append(f"chunk {chunk}, {size} tokens: no off/on pair")
    read = len(cells) - sum(b.startswith("cannot read") for b in bad)
    if expect is not None and read < expect:
        bad.append(f"{read} cell files read, expected {expect}")
    if max_ratio is not None and not rows:
        bad.append("no speed rows")
    text = "\n".join(L) + "\n\n" + "\n".join(summ) + "\n"
    if max_ratio is not None:
        text += ("\nCriterion: mean on/off prefill time at most %.2f for every chunk and size, no errors: %s\n" % (max_ratio, "FAILED: " + "; ".join(bad) if bad else "met"))
    open(out, "w").write(text)
    print(text)
    return 1 if bad and max_ratio is not None else 0


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def quality(argv):
    expect_items, max_loss, alpha = None, 0.10, 0.05
    i = 3
    while i < len(argv):
        if argv[i] == "--expect-items":
            expect_items, i = int(argv[i + 1]), i + 2
        elif argv[i] == "--max-loss":
            max_loss, i = float(argv[i + 1]), i + 2
        elif argv[i] == "--alpha":
            alpha, i = float(argv[i + 1]), i + 2
        else:
            raise SystemExit(f"quality: unknown argument {argv[i]}")
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
    # the pass criterion (there is no oracle: off and on differ by design): enough items scored, and no detectable loss
    a, b, c, d = tab(names)
    acc_off, acc_on = (a + b) / n, (a + c) / n
    pval = mcnemar_p(b, c)
    bad = []
    if expect_items is not None and n < expect_items:
        bad.append(f"only {n} items scored in both arms, expected at least {expect_items}")
    if acc_off - acc_on >= max_loss:
        bad.append(f"on is worse than off by {acc_off - acc_on:.3f} (limit {max_loss})")
    if acc_off - acc_on > 0.05 and pval < alpha:
        bad.append(f"on is worse than off by {acc_off - acc_on:.3f} with McNemar p {pval:.3f} < {alpha}")
    L.append(f"\n**Criterion**: at least {expect_items} items scored in both arms; on not worse than off by {max_loss} or more, and not worse "
             f"by more than 0.05 with McNemar p < {alpha}. This detects a large loss only: with this many items a loss of a few points is "
             f"inside the noise, and the items depend on compressed KV far more than on the recent window (see the doc). "
             + ("FAILED: " + "; ".join(bad) if bad else "Met (off %.3f, on %.3f, p %.3f)." % (acc_off, acc_on, pval)))

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
    return 1 if bad else 0


def tfagree(argv):
    tol = 8
    if "--tol" in argv:
        k = argv.index("--tol")
        tol = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    TF, OFF, ON = load_tokens(argv[0]), load_tokens(argv[1]), load_tokens(argv[2])
    names = [x for x in TF if x in OFF and x in ON]
    bad = []
    if not names:
        bad.append("no prompt is in all three files")
    print(f"{'prompt':<18} {'n_prompt':>8} {'TF~off':>8} {'TF~on':>8} {'off~on':>8}  note")
    tot = {"off": 0, "on": 0}
    n_long = 0
    for nm in names:
        (it, ct), (io, co), (ix, cx) = TF[nm], OFF[nm], ON[nm]
        if not (len(it) == len(io) == len(ix) and (it == io).all() and (it == ix).all()):
            bad.append(f"{nm}: prompt ids differ between the files")
            continue
        n = min(len(ct), len(co), len(cx))
        eo, ex = int((ct[:n] == co[:n]).sum()), int((ct[:n] == cx[:n]).sum())
        oo = int((co[:n] == cx[:n]).sum())
        note = ""
        if len(it) > 128:
            n_long += 1
            note = "replay active"
            tot["off"] += eo
            tot["on"] += ex
            if ex < eo - tol:
                bad.append(f"{nm}: on agrees with TensorFold at {ex}/{n} positions, off at {eo}/{n} (tolerance {tol})")
        elif oo != n or len(co) != len(cx):
            bad.append(f"{nm}: a prompt of {len(it)} ids differs off vs on ({oo}/{n} equal): step 1 should have caught it")
        print(f"{nm:<18} {len(it):>8} {eo:>5}/{n:<2} {ex:>5}/{n:<2} {oo:>5}/{n:<2}  {note}")
    if not n_long:
        bad.append("no prompt of more than 128 ids: the replay was never active")
    print(f"prompts with the replay active: {n_long}; TF agreement there: off {tot['off']}, on {tot['on']} positions")
    if bad:
        for b in bad:
            print("REQUIREMENT FAILED:", b)
        return 1
    print("REQUIREMENT MET")
    return 0


def gatecmp(argv):
    import re
    opts = {"--min-files": "100", "--absent": "", "--present": "", "--differ": ""}
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    offd, ond = pos[0], pos[1]
    rx = re.compile(r"^(?P<what>.+)-o(?P<k>\d+)\.p(?P<pos>\d+)\.")

    def scan(d):
        out = {}
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            m = rx.match(f)
            if m:
                out[(m["what"], int(m["k"]), int(m["pos"]))] = os.path.join(d, f)
        return out

    A, B = scan(offd), scan(ond)
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    bad = []
    if not A or not B:
        bad.append(f"no dump files (off {len(A)}, on {len(B)}) in {offd} / {ond}")

    def enc(w):
        return w.startswith(("layers.0.hc_post", "layers.9.hc_post", "layers.19.hc_post", "layers.20.attn.rope", "layers.20.attn.act_quant"))

    def dec(w):
        return w.startswith("layers.39.hc_post")

    encA = {k: v for k, v in A.items() if enc(k[0])}
    same = diff = missing = 0
    for k, pa in encA.items():
        pb = B.get(k)
        if pb is None:
            missing += 1
            bad.append(f"encoder-half dump {os.path.basename(pa)} is missing from the on run")
        elif open(pa, "rb").read() == open(pb, "rb").read():
            same += 1
        else:
            diff += 1
            bad.append(f"encoder-half dump differs off vs on: {os.path.basename(pa)}")
    extra = [k for k in B if enc(k[0]) and k not in encA]
    for k in extra:
        bad.append(f"encoder-half dump only in the on run: {k}")
    print(f"encoder half (layers 0, 9, 19 outputs; layer 20 rope / act_quant of the kv source): {len(encA)} dumps, {same} identical, {diff} differ, {missing} missing from on")
    if len(encA) < int(opts["--min-files"]):
        bad.append(f"only {len(encA)} encoder-half dumps (wanted at least {opts['--min-files']}): the dump hook did not select what was expected")
    decA = {k[2]: v for k, v in A.items() if dec(k[0])}
    decB = {k[2]: v for k, v in B.items() if dec(k[0])}
    for p in ints(opts["--absent"]):
        print(f"layer 39 at {p}: off {'yes' if p in decA else 'no'}, on {'yes' if p in decB else 'no'}  (on must not have it: below every floor)")
        if p in decB:
            bad.append(f"the on run ran layer 39 at position {p}, below the replay floor")
        if p not in decA:
            bad.append(f"the off run has no layer 39 dump at {p}: the hook did not select it")
    for p in ints(opts["--present"]):
        print(f"layer 39 at {p}: off {'yes' if p in decA else 'no'}, on {'yes' if p in decB else 'no'}  (both must have it)")
        if p not in decB or p not in decA:
            bad.append(f"layer 39 dump at {p} missing (off {p in decA}, on {p in decB})")
    for p in ints(opts["--differ"]):
        if p in decA and p in decB:
            d = open(decA[p], "rb").read() != open(decB[p], "rb").read()
            print(f"layer 39 at {p}: off vs on {'differ' if d else 'IDENTICAL'}  (must differ: the first replayed rows lose their window keys below the floor)")
            if not d:
                bad.append(f"layer 39 at {p} is identical off vs on: the replay's window floor changed nothing")
        else:
            bad.append(f"layer 39 dump at {p} missing, cannot compare")
    if bad:
        for b in bad[:40]:
            print("REQUIREMENT FAILED:", b)
        return 1
    print("REQUIREMENT MET")
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
    return {"tokcmp": tokcmp, "speedtab": speedtab, "quality": quality, "checksha": checksha, "tfagree": tfagree, "gatecmp": gatecmp}[cmd](a)


if __name__ == "__main__":
    sys.exit(main())
