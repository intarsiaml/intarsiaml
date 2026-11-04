"""Self-test of replay-test-lib.py's pass criteria on synthetic inputs (no GPU, no server):
  uv run --no-project --with numpy --with safetensors python scripts/dsv41-tp2/replay-test-lib-selftest.py
Each criterion must PASS on good data and FAIL on the failure it exists for."""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lib", os.path.join(HERE, "replay-test-lib.py"))
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
ROOT = os.environ.get("SELFTEST_DIR") or os.path.join(HERE, "..", "..", ".scratch")
os.makedirs(ROOT, exist_ok=True)
T = tempfile.mkdtemp(dir=ROOT, prefix="selftest-")
FAILS = []


def run(fn, argv):
    with contextlib.redirect_stdout(io.StringIO()) as o:
        rc = fn(argv)
    return rc, o.getvalue()


def expect(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: exit {got}, wanted {want}")
    if not ok:
        FAILS.append(name)


def tokfile(name, prompts):
    t, mp = {}, []
    for i, (nm, n_prompt, cont) in enumerate(prompts):
        t[f"ids_{i}"] = np.arange(n_prompt, dtype=np.int32)
        t[f"cont_{i}"] = np.array(cont, np.int32)
        mp.append({"name": nm})
    p = os.path.join(T, name)
    save_file(t, p, metadata={"format": "intarsia-golden-tokens-v1", "prompts": json.dumps(mp)})
    return p


base = list(range(64))
shift = [x + 1 for x in base]
half = base[:32] + shift[32:]
garbage = [(x * 7919 + 13) % 129000 + 1000 for x in base]

# tokcmp --require-identical: identical passes, one different token fails
a = tokfile("a.st", [("p1", 40, base), ("p2", 90, base)])
b_same = tokfile("b_same.st", [("p1", 40, base), ("p2", 90, base)])
b_diff = tokfile("b_diff.st", [("p1", 40, base), ("p2", 90, base[:10] + [999] + base[11:])])
expect("tokcmp identical", run(lib.tokcmp, [a, b_same, "--require-identical", "--max-prompt", "128"])[0], 0)
expect("tokcmp one token differs", run(lib.tokcmp, [a, b_diff, "--require-identical"])[0], 1)
expect("tokcmp prompt longer than 128", run(lib.tokcmp, [tokfile("l.st", [("p1", 300, base)]), tokfile("l2.st", [("p1", 300, base)]), "--max-prompt", "128"])[0], 1)
# --require-total 640: 640 tokens identical passes, 639 fails
ten = [(f"p{i}", 50, base) for i in range(10)]
t640a, t640b = tokfile("t640a.st", ten), tokfile("t640b.st", ten)
ten_bad = [(f"p{i}", 50, base if i else base[:-1] + [5]) for i in range(10)]
expect("tokcmp 640 of 640", run(lib.tokcmp, [t640a, t640b, "--require-total", "640"])[0], 0)
expect("tokcmp 639 of 640", run(lib.tokcmp, [t640a, tokfile("t639.st", ten_bad), "--require-total", "640"])[0], 1)
# --min-equal-pct / --same-lengths / --prefix / --common: the long-decode tripwire
d_on = tokfile("d_on.st", [("d_1", 700, half), ("g_1", 50, base)])
d_off = tokfile("d_off.st", [("d_1", 700, base), ("g_1", 50, garbage)])
expect("tokcmp prefix + min-equal 30% (50% equal)", run(lib.tokcmp, [d_off, d_on, "--prefix", "d_", "--min-equal-pct", "30", "--same-lengths"])[0], 0)
expect("tokcmp prefix + min-equal 60% fails", run(lib.tokcmp, [d_off, d_on, "--prefix", "d_", "--min-equal-pct", "60"])[0], 1)
d_gar = tokfile("d_gar.st", [("d_1", 700, garbage)])
expect("tokcmp garbage ids fail the tripwire", run(lib.tokcmp, [tokfile("d_ref.st", [("d_1", 700, base)]), d_gar, "--min-equal-pct", "30"])[0], 1)
d_short = tokfile("d_short.st", [("d_1", 700, base[:10])])
expect("tokcmp short continuation fails --same-lengths", run(lib.tokcmp, [tokfile("d_ref2.st", [("d_1", 700, base)]), d_short, "--same-lengths"])[0], 1)
expect("tokcmp --common ignores unshared prompts", run(lib.tokcmp, [tokfile("c1.st", [("x", 40, base), ("y", 40, base)]), tokfile("c2.st", [("x", 40, base)]), "--common", "--require-identical"])[0], 0)
expect("tokcmp unshared prompts fail without --common", run(lib.tokcmp, [tokfile("c3.st", [("x", 40, base), ("y", 40, base)]), tokfile("c4.st", [("x", 40, base)]), "--require-identical"])[0], 1)

# tfagree: on as good as off passes; on much worse fails; only short prompts fails; short prompt off != on fails
tf = tokfile("tf.st", [("s", 40, base), ("long", 300, base)])
off = tokfile("off.st", [("s", 40, base), ("long", 300, base)])
on_ok = tokfile("on_ok.st", [("s", 40, base), ("long", 300, base[:60] + [1, 2, 3, 4])])
on_bad = tokfile("on_bad.st", [("s", 40, base), ("long", 300, garbage)])
on_short = tokfile("on_short.st", [("s", 40, shift), ("long", 300, base)])
expect("tfagree on within tolerance", run(lib.tfagree, [tf, off, on_ok])[0], 0)
expect("tfagree on much worse", run(lib.tfagree, [tf, off, on_bad])[0], 1)
expect("tfagree short prompt differs", run(lib.tfagree, [tf, off, on_short])[0], 1)
expect("tfagree replay never active", run(lib.tfagree, [tokfile("tfs.st", [("s", 40, base)]), tokfile("os.st", [("s", 40, base)]), tokfile("ns.st", [("s", 40, base)])])[0], 1)


# gatecmp: dumps
def dump(d, files):
    os.makedirs(d, exist_ok=True)
    for what, pos, data in files:
        open(os.path.join(d, f"{what}-o0.p{pos}.bf16.4x8.bin"), "wb").write(data)


POS = [5, 300, 700, 1100, 1372, 1450]
ENC = ["layers.0.hc_post-ffn-", "layers.9.hc_post-ffn-", "layers.19.hc_post-ffn-", "layers.20.attn.rope-index_k-", "layers.20.attn.rope-latent-",
       "layers.20.attn.act_quant-index_k-", "layers.20.attn.act_quant-latent-"]


def mk(tag, on, break_enc=None, floor_ok=True):
    d = os.path.join(T, tag)
    files = [(w, p, b"E%d" % p) for w in ENC for p in POS]
    if break_enc:
        files = [(w, p, (b"X" if (w, p) == break_enc else x)) for (w, p, x) in files]
    L39 = {p: b"A%d" % p for p in (300, 700, 1100, 1372, 1450)}
    if on:
        L39 = {p: (b"B%d" % p if p == 1372 else b"A%d" % p) for p in (1100, 1372, 1450)} if floor_ok else {p: b"A%d" % p for p in L39}
    files += [("layers.39.hc_post-ffn-", p, x) for p, x in L39.items()]
    dump(d, files)
    return d


args = ["--min-files", "40", "--absent", "300,700", "--present", "1100,1450", "--differ", "1372"]
off_d = mk("g_off", False)
expect("gatecmp good run", run(lib.gatecmp, [off_d, mk("g_on", True)] + args)[0], 0)
expect("gatecmp encoder-half dump differs", run(lib.gatecmp, [off_d, mk("g_on_bad", True, break_enc=("layers.19.hc_post-ffn-", 700))] + args)[0], 1)
expect("gatecmp replay not active (layer 39 below the floor)", run(lib.gatecmp, [off_d, mk("g_on_noreplay", False)] + args)[0], 1)
expect("gatecmp floor mask changed nothing", run(lib.gatecmp, [off_d, mk("g_on_nomask", True, floor_ok=False)] + args)[0], 1)
expect("gatecmp empty on dir", run(lib.gatecmp, [off_d, os.path.join(T, "nothing")] + args)[0], 1)

# speedtab
def speed(name, cells):
    p = os.path.join(T, name)
    json.dump({"speed": {"rows": [{"target_prompt_tokens": s, "prompt_tokens": s, "ttft_s": {"median": t}, "prefill_tok_s": {"median": s / t}, "errors": e} for s, t, e in cells]}}, open(p, "w"))
    return p


cells = lambda on_t: [f"512:off:1:{speed('o1.json', [(4096, 4.0, 0), (32768, 40.0, 0)])}", f"512:on:1:{speed('n1.json', [(4096, 4.0 * on_t, 0), (32768, 40.0 * on_t, 0)])}",  # noqa: E731
                      f"512:off:2:{speed('o2.json', [(4096, 4.1, 0), (32768, 40.5, 0)])}", f"512:on:2:{speed('n2.json', [(4096, 4.1 * on_t, 0), (32768, 40.5 * on_t, 0)])}"]
out = os.path.join(T, "speed.md")
expect("speedtab faster", run(lib.speedtab, [out, "--max-ratio", "1.05", "--expect-cells", "4"] + cells(0.6))[0], 0)
expect("speedtab slower", run(lib.speedtab, [out, "--max-ratio", "1.05"] + cells(1.2))[0], 1)
expect("speedtab missing cells", run(lib.speedtab, [out, "--max-ratio", "1.05", "--expect-cells", "8"] + cells(0.6))[0], 1)
expect("speedtab error rows", run(lib.speedtab, [out, "--max-ratio", "1.05", f"512:off:1:{speed('e1.json', [(4096, 4.0, 3)])}", f"512:on:1:{speed('e2.json', [(4096, 2.0, 0)])}"])[0], 1)
expect("speedtab without a criterion never fails", run(lib.speedtab, [out] + cells(1.2))[0], 0)


# quality
def qres(name, ok):
    p = os.path.join(T, name)
    json.dump({"datasets": [{"cases": [{"name": f"it{i}", "success": bool(s), "actual_output": "x"} for i, s in enumerate(ok)]}]}, open(p, "w"))
    return p


N = 100
man = os.path.join(T, "man.json")
json.dump({"items": [{"name": f"it{i}", "idx": i, "depths": [0.1, 0.9], "kind": "hop2", "k": 32} for i in range(N)]}, open(man, "w"))
pairs = os.path.join(T, "pairs.txt")


def q(off_ok, on_ok, extra):
    open(pairs, "w").write(f"s {qres('qo.json', off_ok)} {qres('qn.json', on_ok)}\n")
    return run(lib.quality, [man, pairs, os.path.join(T, "q.md")] + extra)[0]


off_ok = [i % 2 == 0 for i in range(N)]
expect("quality equal arms", q(off_ok, off_ok, ["--expect-items", "90"]), 0)
expect("quality on lost 12 of 50 (>= 0.10)", q(off_ok, [(i % 2 == 0) and i >= 24 for i in range(N)], ["--expect-items", "90"]), 1)
expect("quality on lost 6 (p < 0.05 needs more): 0.06 loss, p 0.03", q(off_ok, [(i % 2 == 0) and i >= 12 for i in range(N)], ["--expect-items", "90"]), 1)
expect("quality on lost 3 (noise)", q(off_ok, [(i % 2 == 0) and i >= 6 for i in range(N)], ["--expect-items", "90"]), 0)
expect("quality too few items", q(off_ok[:50], off_ok[:50], ["--expect-items", "90"]), 1)
expect("quality on garbage (all wrong)", q(off_ok, [False] * N, ["--expect-items", "90"]), 1)


# ---- step 7: the recent-window scorer, the reading rule and the analysis (replay-test-lib.py `recent`)
def sc(reply, exp="QK-4821-MT"):
    return lib.score_code(reply, exp)


expect("scorer: the bare code", sc("QK-4821-MT"), 1)
expect("scorer: code in a sentence with a full stop", sc("The code is QK-4821-MT."), 1)
expect("scorer: lower case", sc("qk-4821-mt"), 1)
expect("scorer: a decoy's code", sc("ZZ-1111-AA"), 0)
expect("scorer: the right code extended by a letter", sc("QK-4821-MTX"), 0)
expect("scorer: the right code extended by a hyphen group", sc("QK-4821-MT-2"), 0)
expect("scorer: a letter before it", sc("AQK-4821-MT"), 0)
expect("scorer: a digit too many", sc("QK-48210-MT"), 0)
expect("scorer: no code at all", sc("I cannot find it."), 0)
expect("scorer: working then the right code last", sc("Not ZZ-1111-AA. FINAL: QK-4821-MT"), 1)
expect("scorer: the right code then a decoy last", sc("QK-4821-MT or ZZ-1111-AA"), 0)
expect("scorer: empty reply is None, not wrong", sc(""), None)
expect("scorer: blank reply is None", sc(" \n "), None)


def fires(cells):
    return lib.rule_fires({k: (24, 24, b, c) for k, (b, c) in cells.items()})


expect("rule: bucket 2, b 4 c 0", fires({2: (4, 0)}), True)
expect("rule: bucket 2, b 3 c 0", fires({2: (3, 0)}), False)
expect("rule: bucket 3, b 5 c 1 (diff 4)", fires({3: (5, 1)}), True)
expect("rule: bucket 1, b 5 c 2 (diff 3)", fires({1: (5, 2)}), False)
expect("rule: bucket 4 (secondary), b 4 c 0", fires({4: (4, 0)}), False)
expect("rule: bucket 4 (secondary), b 5 c 0", fires({4: (5, 0)}), True)
expect("rule: far control, b 5 c 0", fires({6: (5, 0)}), True)
expect("rule: bucket 2, on better by 6", fires({2: (0, 6)}), False)
expect("rule: three primary cells b 3 each (pooled 9, p 0.002)", fires({1: (3, 0), 2: (3, 0), 3: (3, 0)}), True)
expect("rule: three primary cells b 2 each (pooled 6, p 0.016)", fires({1: (2, 0), 2: (2, 0), 3: (2, 0)}), False)
expect("rule: noise only (b 1 c 1 everywhere)", fires({k: (1, 1) for k in range(1, 7)}), False)
expect("p_one_sided(4,0) is 1/16", abs(lib.p_one_sided(4, 0) - 1 / 16) < 1e-12, True)

NI = 4
man7 = os.path.join(T, "man7.json")
items7 = []
for ln, tot in (("4k", 4000), ("32k", 30400)):
    for bk in range(1, 7):
        for i in range(NI):
            items7.append({"name": f"rw-{ln}-b{bk}-{i:02d}", "idx": len(items7), "length": ln, "bucket": bk, "total_tokens": tot + i,
                           "expected": f"QK-{1000 + len(items7)}-MT", "decoys": [{"code": "ZZ-1111-AA"}], "d_start": 10 * bk,
                           "chunks": {"2048": {"window_spans_chunks": False}}})
json.dump({"items": items7}, open(man7, "w"))


def res7(name, fn, ptok=None, cached=0, drop=(), err=(), empty=()):
    cases = []
    for m in items7:
        if m["name"] in drop:
            continue
        rep = fn(m)
        if m["name"] in empty:
            rep = ""
        cases.append({"name": m["name"], "success": rep.strip().endswith(m["expected"]), "actual_output": rep,
                      "error": "boom" if m["name"] in err else None,
                      "speed": {"prompt_tokens": m["total_tokens"] if ptok is None else ptok, "cached_tokens": cached}})
    p = os.path.join(T, name)
    json.dump({"datasets": [{"dataset": "x", "rows": len(items7), "cases": cases}]}, open(p, "w"))
    return p


right7 = lambda m: m["expected"]
def lose7(buckets, n):
    """the first n items (manifest order) of each listed bucket get a decoy's code"""
    seen = {}
    def f(m):
        k = seen[m["bucket"]] = seen.get(m["bucket"], -1) + 1
        return "ZZ-1111-AA" if m["bucket"] in buckets and k < n else m["expected"]
    return f


def rec(off, on, extra=()):
    pr = os.path.join(T, "pairs7.txt")
    open(pr, "w").write(f"2048 {off} {on}\n")
    return run(lib.recent, [man7, pr, os.path.join(T, "r7.md")] + list(extra))[0]


o7 = res7("o7.json", right7)
expect("recent: identical arms", rec(o7, res7("n7.json", right7)), 0)
expect("recent: on loses 4 of 8 in bucket 2 (DEFECT)", rec(o7, res7("n7b.json", lose7({2}, 4))), 2)
expect("recent: on loses 2 of 8 in bucket 2 (b 2: no defect)", rec(o7, res7("n7c.json", lose7({2}, 2))), 0)
expect("recent: on loses in the far control, 4 of 8 (secondary rule needs 5)", rec(o7, res7("n7d.json", lose7({6}, 4))), 0)
expect("recent: an empty reply is INVALID", rec(o7, res7("n7e.json", right7, empty={"rw-4k-b3-01"})), 1)
expect("recent: a call error is INVALID", rec(o7, res7("n7f.json", right7, err={"rw-4k-b3-01"})), 1)
expect("recent: server prompt_tokens differs from the manifest", rec(o7, res7("n7g.json", right7, ptok=4096)), 1)
expect("recent: prefix-cache hit (cached_tokens 7)", rec(o7, res7("n7h.json", right7, cached=7)), 1)
expect("recent: an item missing in the on arm", rec(o7, res7("n7i.json", right7, drop={"rw-32k-b1-00"})), 1)
expect("recent: fewer pairs than --expect-items", rec(o7, res7("n7j.json", right7), ["--expect-items", str(len(items7) + 1)]), 1)
shaj = os.path.join(T, "sha7.json")
os.makedirs(os.path.join(T, "ds7"), exist_ok=True)
open(os.path.join(T, "ds7", "recent.4k.jsonl"), "w").write("x\n")
json.dump({"4k": lib.sha(os.path.join(T, "ds7", "recent.4k.jsonl"))}, open(shaj, "w"))
expect("checksha: dataset equal", run(lib.checksha, [shaj, os.path.join(T, "ds7"), "recent.", "4k"])[0], 0)
open(os.path.join(T, "ds7", "recent.4k.jsonl"), "w").write("y\n")
expect("checksha: dataset differs", run(lib.checksha, [shaj, os.path.join(T, "ds7"), "recent.", "4k"])[0], 1)
expect("checksha: dataset missing", run(lib.checksha, [shaj, os.path.join(T, "ds7"), "recent.", "32k"])[0], 1)

shutil.rmtree(T, ignore_errors=True)
print("SELFTEST FAILED: " + ", ".join(FAILS) if FAILS else "SELFTEST PASSED")
sys.exit(1 if FAILS else 0)
