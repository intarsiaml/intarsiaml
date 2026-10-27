#!/usr/bin/env python3
"""Lane spf: planted mutants of `rope_multi_kernel` (kernels/cuda/elementwise.cu), run ON spark1 inside one fleetq slot:

    fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 30 --label "spf: rope mutants" -- ssh spark1 'python3 - dev' < scripts/spf/mutants.py

Per mutant: copy the tree (target/ included) to ~/intarsiaml-spf/mut, change the kernel's source (the replacement is applied only
inside rope_multi_kernel's body, the old text must occur there exactly once), cargo build --release --features cuda -p intarsia-cli,
then `intarsia kernels test --backend cuda $KFLAGS` (default: the full set; KFLAGS=--quick) must report a REJECTED kernel (CAUGHT) or the mutant SURVIVED. The clean binary runs
first and must pass.
"""
import os, re, shutil, subprocess, sys

tree = os.path.expanduser("~/intarsiaml-spf/" + (sys.argv[1] if len(sys.argv) > 1 else "dev"))
mut = os.path.expanduser("~/intarsiaml-spf/mut")
env = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + os.environ["PATH"])
MUTANTS = {
    "rope-sign": ("a[j] * fc - b[j] * fs", "a[j] * fc + b[j] * fs"),
    "rope-pos-off-by-one": ("const int64_t pos = pos0 + t;", "const int64_t pos = pos0 + t + 1;"),
    "rope-skip-head-group": ("n0 += kRopeHeads)", "n0 += kRopeHeads + 1)"),
    "rope-freq-denominator": ("-2.0 * (double)k / (double)RD", "-2.0 * (double)k / (double)D"),
    "rope-pair-lane": ("out[base + k + half] = ik_f2b(b[j] * fc + a[j] * fs);", "out[base + k + half] = ik_f2b(b[j] * fs + a[j] * fc);"),
}

def run(cmd, cwd=None, timeout=1500):
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)

def ktest(binary):
    rc, out = run(f"{binary} kernels test --backend cuda {os.environ.get('KFLAGS', '')} 2>&1 | grep -E 'REJECTED|cases in|admission failed'", timeout=1700)
    return out.strip()

print("clean:", ktest(f"{tree}/target/release/intarsia").replace("\n", " | "), flush=True)
names = [n for n in (sys.argv[2:] or MUTANTS)]
for name in names:
    old, new = MUTANTS[name]
    if os.path.exists(mut):
        shutil.rmtree(mut)
    run(f"cp -a {tree} {mut}")
    p = f"{mut}/kernels/cuda/elementwise.cu"
    s = open(p).read()
    a = s.index("__global__ void rope_multi_kernel")
    b = s.index("// the un-rotated tail", a)
    body = s[a:b]
    if body.count(old) != 1:
        print(f"{name}: anchor found {body.count(old)} times: NOT APPLIED")
        continue
    open(p, "w").write(s[:a] + body.replace(old, new) + s[b:])
    run(f"find {mut} -path {mut}/target -prune -o -name '*.cu' -print0 | xargs -0 touch")
    rc, out = run("cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3", cwd=mut)
    out_t = ktest(f"{mut}/target/release/intarsia")
    verdict = "CAUGHT" if "REJECTED" in out_t else "SURVIVED"
    print(f"{name}: {verdict}: {out_t.replace(chr(10), ' | ')[:300]}", flush=True)
shutil.rmtree(mut, ignore_errors=True)
