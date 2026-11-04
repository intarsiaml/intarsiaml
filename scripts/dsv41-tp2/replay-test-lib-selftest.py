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
shutil.rmtree(T, ignore_errors=True)
print("SELFTEST FAILED: " + ", ".join(FAILS) if FAILS else "SELFTEST PASSED")
sys.exit(1 if FAILS else 0)
