#!/usr/bin/env python3
"""Lane cgv: planted mutants of the multi-row W4A16 GEMV (`mxfp4_gemv_mr_kernel`, kernels/cuda/mxfp4.cu), run ON spark1 inside one fleetq slot:

    fleetq run --host spark1 --quiet gpu,cpu --mem 20 --expect 40 --label "cgv: mxfp4 gemv_mr mutants" -- ssh spark1 'python3 - dev' < scripts/cgv/mutants.py

Per mutant: copy the tree (target/ included) to ~/intarsiaml-cgv/mut, change one source text (the old text must occur exactly once in its file),
cargo build --release --features cuda -p intarsia-cli, then `intarsia kernels test --backend cuda --op matmul_mxfp4` must report a REJECTED kernel
(CAUGHT) or the mutant SURVIVED. The clean binary runs first and must pass. Mutants that only break the tail (N not a multiple of R) are
expected to be caught by tests/mxfp4_gemv_mr.rs (odd N), not by the sweep, and are listed with the check that catches them.
"""
import os, shutil, subprocess, sys

tree = os.path.expanduser("~/intarsiaml-cgv/trees/" + (sys.argv[1] if len(sys.argv) > 1 else "dev"))
mut = os.path.expanduser("~/intarsiaml-cgv/mut")
env = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + os.environ["PATH"])
F = "kernels/cuda/mxfp4.cu"
MUTANTS = {
    "x-row-stride": (F, "x + (int64_t)m * K + (int64_t)u * kG", "x + (int64_t)m * (K + 32) + (int64_t)u * kG"),
    "scale-row-of-r": (F, "sr[r] = sc + nr * (K / kG);", "sr[r] = sc + n0 * (K / kG);"),
    "weight-row-of-r": (F, "wr[r] = reinterpret_cast<const uint4*>(w + nr * (K / 2));", "wr[r] = reinterpret_cast<const uint4*>(w + n0 * (K / 2));"),
    "store-column": (F, "ik_store(out + (int64_t)m * N + n0 + r, t);", "ik_store(out + (int64_t)m * N + n0, t);"),
    "group-scale": (F, "acc[r][m] = fmaf(t, s, acc[r][m]);", "acc[r][m] = fmaf(t, s * 0.5f, acc[r][m]);"),
    "e8m0-of-wrong-row": (F, "const float s = mx_e8m0(sb[r]);\n            float wf[kG];\n            ik_fp4x8_to_f32(wv[r].x, wf);\n            ik_fp4x8_to_f32(wv[r].y, wf + 8);\n            ik_fp4x8_to_f32(wv[r].z, wf + 16);\n            ik_fp4x8_to_f32(wv[r].w, wf + 24);\n#pragma unroll\n            for (int m = 0; m < MT; ++m) {", "const float s = mx_e8m0(sb[0]);\n            float wf[kG];\n            ik_fp4x8_to_f32(wv[r].x, wf);\n            ik_fp4x8_to_f32(wv[r].y, wf + 8);\n            ik_fp4x8_to_f32(wv[r].z, wf + 16);\n            ik_fp4x8_to_f32(wv[r].w, wf + 24);\n#pragma unroll\n            for (int m = 0; m < MT; ++m) {"),
    # tail-only: caught by tests/mxfp4_gemv_mr.rs (N = 1, 3, 5, 9, 135, 1001), not necessarily by the sweep
    "tail-store-guard": (F, "if (lane == 0 && n0 + r < N) ik_store", "if (lane == 0) ik_store"),
    "tail-row-clamp": (F, "const int64_t nr = n0 + r < N ? n0 + r : N - 1;", "const int64_t nr = n0 + r;"),
}

def run(cmd, cwd=None, timeout=1500):
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)

def ktest(binary):
    rc, out = run(f"{binary} kernels test --backend cuda --op matmul_mxfp4 {os.environ.get('KFLAGS', '')} 2>&1 | grep -E 'REJECTED|cases in|admission failed|gemv_mr'", timeout=1700)
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
