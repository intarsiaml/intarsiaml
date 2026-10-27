#!/usr/bin/env python3
"""Lane scc: planted mutants of the row-exact fp8 / nvfp4 GEMV kernels' new paths (kernels/cuda/fp8.cu, nvfp4.cu), run ON spark2 in one fleetq slot:

    fleetq run --host spark2 --mem 8 --expect 40 --label "scc: rows mutants" -- ssh spark2 'python3 -' < scripts/scc/mutants.py [name ...]

Edits the synced tree ~/intarsiaml-scc/src in place (restored at the end; the next scripts/scc/build.sh re-syncs it anyway). Per mutant:
rebuild the rows_exact test binary, run the matching bitwise test (fp8_rows / nvfp4_rows: every kernel form, every shape). The replacement
is applied only inside the named region of the kernel's source and the old text must occur there exactly once. The clean tree runs first and
must pass. A mutant is CAUGHT when the test fails.
"""
import os, subprocess, sys

tree = os.path.expanduser("~/intarsiaml-scc/src")
env = dict(os.environ, PATH=os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + os.environ["PATH"], INTARSIA_REQUIRE_GPU="1")
F8 = ("kernels/cuda/fp8.cu", "fp8_rows")
F8_CHUNK = F8 + ("fp8_gemv_rows_kernel(const ik_bf16", "// ---- Row-exact multi-row GEMV, persistent variant", None)
F8_FULL = F8 + ("fp8_gemv_rows_full_kernel(const ik_bf16", "inline int sm_count()", None)
NV = ("kernels/cuda/nvfp4.cu", "nvfp4_rows")
NV_TILE = NV + ("nvfp4_gemv_tile_kernel(const ik_bf16", "// Outputs per warp of the row-exact launcher", None)
# name: (file, test filter, region start, region end, old, new)  -- end None: the region ends at the first end marker found after the start
MUTANTS = {
    # rounding-only: the 16-long fmaf chain walked backwards (same products, different rounding)
    "fp8-chunk-fma-order": (F8_CHUNK, "for (int e = 0; e < 16; ++e) sdot = fmaf(wf[j][e], xf[e], sdot);", "for (int e = 15; e >= 0; --e) sdot = fmaf(wf[j][e], xf[e], sdot);"),
    "fp8-chunk-store-col": (F8_CHUNK, "ik_store(out + (int64_t)m * N + nb + j, t * s_out);", "ik_store(out + (int64_t)m * N + nb + ((j + 1) % NT), t * s_out);"),
    "fp8-chunk-drop-tail": (F8_CHUNK, "if (lane == 0 && nb + j < N)", "if (lane == 0 && nb + j + 1 < N)"),
    "fp8-chunk-weight-row": (F8_CHUNK, "wr[j] = reinterpret_cast<const uint4*>(w + (nb + j < N ? nb + j : 0) * K);", "wr[j] = reinterpret_cast<const uint4*>(w + (nb < N ? nb : 0) * K);"),
    "fp8-chunk-partial-warp": (F8_CHUNK, "if (nb < N) {  // a dead warp", "if (nb + NT - 1 < N) {  // a dead warp"),
    "fp8-full-fma-order": (F8_FULL, "for (int e = 0; e < 16; ++e) sdot = fmaf(wf[j][e], xf[e], sdot);", "for (int e = 15; e >= 0; --e) sdot = fmaf(wf[j][e], xf[e], sdot);"),
    "fp8-full-store-col": (F8_FULL, "ik_store(out + (int64_t)m * N + nb + j, r * s_out);", "ik_store(out + (int64_t)m * N + nb + ((j + 1) % NT), r * s_out);"),
    "fp8-full-drop-tail": (F8_FULL, "if (lane == 0 && nb + j < N)", "if (lane == 0 && nb + j + 1 < N)"),
    "fp8-full-acc-alias": (F8_FULL, "acc[j][m] += sdot;", "acc[0][m] += sdot;"),
    "fp8-full-tile-stride": (F8_FULL, "t += (int64_t)gridDim.x * W)", "t += (int64_t)gridDim.x * W + 1)"),
    "fp8-full-stage-half": (F8_FULL, "xs[(m * 2 + (j & 1)) * KV + (j >> 1)] = v;", "xs[(m * 2 + ((j & 1) ^ 1)) * KV + (j >> 1)] = v;"),
    "fp8-full-no-actq": (F8_FULL, "        if (actq) {\n            float f[8];", "        if (false && actq) {\n            float f[8];"),
    "fp8-full-lane-skew": (F8_FULL, "for (int u = lane; u < KV; u += IK_WARP) {", "for (int u = (lane + 1) % IK_WARP; u < KV; u += IK_WARP) {"),
    "nv-tile-fma-order": (NV_TILE, "for (int e = 0; e < 16; ++e) s = fmaf(wf[j][e], xf[e], s);", "for (int e = 15; e >= 0; --e) s = fmaf(wf[j][e], xf[e], s);"),
    "nv-tile-store-col": (NV_TILE, "ik_store(out + (int64_t)m * N + nb + j, r * g);", "ik_store(out + (int64_t)m * N + nb + ((j + 1) % NT), r * g);"),
    "nv-tile-scale-row": (NV_TILE, "sr[j] = bs + n * (K / 16);", "sr[j] = bs + nb * (K / 16);"),
    "nv-tile-acc-alias": (NV_TILE, "acc[j][m] = fmaf(s, sc[j], acc[j][m]);", "acc[0][m] = fmaf(s, sc[j], acc[0][m]);"),
    "nv-tile-partial-warp": (NV_TILE, "if (nb >= N) return;", "if (nb + NT - 1 >= N) return;"),
}

def run(cmd, timeout=1700):
    r = subprocess.run(cmd, shell=True, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)

def test(filt):
    rc, out = run(f"cargo test -p intarsia-backend-cuda --release --test rows_exact {filt} -- --test-threads=1 2>&1 | grep -aE '^test |test result|G-B FAIL|^error' | cut -c1-260")
    return out.strip()

names = sys.argv[1:] or list(MUTANTS)
origs = {}
try:
    for filt in ("fp8_rows", "nvfp4_rows"):
        print(f"clean {filt}:", test(filt).replace("\n", " | "), flush=True)
    for name in names:
        (path, filt, start, end, _), old, new = MUTANTS[name]
        if path not in origs:
            origs[path] = open(f"{tree}/{path}").read()
        orig = origs[path]
        a = orig.index(start)
        b = orig.index(end, a)
        body = orig[a:b]
        if body.count(old) != 1:
            print(f"{name}: anchor found {body.count(old)} times: NOT APPLIED", flush=True)
            continue
        open(f"{tree}/{path}", "w").write(orig[:a] + body.replace(old, new) + orig[b:])
        run("find kernels -name '*.cu' -print0 | xargs -0 touch")
        out = test(filt)
        verdict = "CAUGHT" if ("FAILED" in out or "G-B FAIL" in out or "panicked" in out) else ("BUILD-ERROR" if "error" in out else "SURVIVED")
        print(f"{name}: {verdict}: {out.replace(chr(10), ' | ')[:330]}", flush=True)
        open(f"{tree}/{path}", "w").write(orig)
finally:
    for path, orig in origs.items():
        open(f"{tree}/{path}", "w").write(orig)
    run("find kernels -name '*.cu' -print0 | xargs -0 touch")
