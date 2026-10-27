#!/usr/bin/env python3
"""Lane cmx: planted mutants of `matmul_mxfp4_mma` (kernels/cuda/wq_gemm.cuh FMT_MXFP4 paths, kernels/cuda/mxfp4_gemm.cu), run ON spark1
inside one fleetq slot:

    fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 30 --label "cmx: mxfp4 mma mutants" -- ssh spark1 'python3 - dev' < scripts/cmx/mutants.py

Per mutant: copy the tree (target/ included) to ~/intarsiaml-cmx/mut, change one source text (the old text must occur exactly once in its file),
cargo build --release --features cuda -p intarsia-cli, then `intarsia kernels test --backend cuda --op matmul_mxfp4 $KFLAGS` must report a
REJECTED kernel (CAUGHT) or the mutant SURVIVED. The clean binary runs first and must pass.
"""
import os, shutil, subprocess, sys

tree = os.path.expanduser("~/intarsiaml-cmx/trees/" + (sys.argv[1] if len(sys.argv) > 1 else "dev"))
mut = os.path.expanduser("~/intarsiaml-cmx/mut")
env = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + os.environ["PATH"])
W = "kernels/cuda/wq_gemm.cuh"
MUTANTS = {
    "e8m0-bias": (W, "__uint_as_float(b << 23))", "__uint_as_float((b + 1u) << 23))"),
    "scale-per-piece": (W, "8 * (b >> 1) : 0", "8 * b : 0"),
    "scale-row-stride": (W, "bs + (int64_t)wrow * (K / 32) + khalf", "bs + (int64_t)wrow * (K / 16) + khalf"),
    "scale-step-stride": (W, "rawsc = *reinterpret_cast<const uint16_t*>(sp + st * (BK / 32));", "rawsc = *reinterpret_cast<const uint16_t*>(sp + st * (BK / 16));"),
    "out-scale": (W, "(FMT == FMT_NVFP4 ? gs[0] : 1.f)", "(FMT == FMT_NVFP4 ? gs[0] : 0.5f)"),
    "khalf-scale": (W, "bs + (int64_t)wrow * (K / 32) + khalf", "bs + (int64_t)wrow * (K / 32)"),
}

def run(cmd, cwd=None, timeout=1500):
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)

def ktest(binary):
    rc, out = run(f"{binary} kernels test --backend cuda --op matmul_mxfp4 {os.environ.get('KFLAGS', '')} 2>&1 | grep -E 'REJECTED|cases in|admission failed'", timeout=1700)
    return out.strip()

print("clean:", ktest(f"{tree}/target/release/intarsia").replace("\n", " | "), flush=True)
names = [n for n in (sys.argv[2:] or MUTANTS)]
for name in names:
    f, old, new = MUTANTS[name]
    if os.path.exists(mut):
        shutil.rmtree(mut)
    run(f"cp -a {tree} {mut}")
    p = f"{mut}/{f}"
    s = open(p).read()
    if s.count(old) != 1:
        print(f"{name}: anchor found {s.count(old)} times: NOT APPLIED")
        continue
    open(p, "w").write(s.replace(old, new))
    run(f"find {mut} -path {mut}/target -prune -o \\( -name '*.cu' -o -name '*.cuh' \\) -print0 | xargs -0 touch")
    rc, out = run("cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3", cwd=mut)
    out_t = ktest(f"{mut}/target/release/intarsia")
    verdict = "CAUGHT" if "REJECTED" in out_t else "SURVIVED"
    print(f"{name}: {verdict}: {out_t.replace(chr(10), ' | ')[:300]}", flush=True)
shutil.rmtree(mut, ignore_errors=True)
