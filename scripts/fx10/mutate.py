#!/usr/bin/env python3
"""Lane FX10 mutation check: break one kernel line at a time in a scratch worktree, run the named test, expect it to FAIL, revert.
usage (inside a fleetq slot): scripts/fx10/mutate.py <scratch-worktree> <target-dir> <out-dir>"""
import subprocess, sys, pathlib

wt, tgt, out = map(pathlib.Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
# (id, kernel file, old text, new text, test binary, test filter)
M = [
 ("gate_split-gate-from-q-half", "gdn.metal", "gate[p.s2 + gid] = x[p.s0 + src + p.b];", "gate[p.s2 + gid] = x[p.s0 + src];", "gdn", "gate_split_is_an_exact_per_head_split"),
 ("sigmoid_mul-1pct-gain", "gdn.metal", "f2bf(bf2f(a[p.s0 + gid]) * gdn_sigmoid(", "f2bf(1.01f * bf2f(a[p.s0 + gid]) * gdn_sigmoid(", "gdn", "sigmoid_mul_matches_the_reference"),
 ("sigmoid_mul-saturation-clamp", "gdn.metal", "f2bf(bf2f(a[p.s0 + gid]) * gdn_sigmoid(bf2f(g[p.s1 + gid])))", "f2bf(bf2f(a[p.s0 + gid]) * gdn_sigmoid(clamp(bf2f(g[p.s1 + gid]), -6.0f, 6.0f)))", "gdn", "sigmoid_mul_matches_the_reference"),
 ("delta-scale-hardcoded-dk128", "gdn.metal", "const float scale = 1.0f / precise::sqrt((float)DK);", "const float scale = 0.08838834764831845f;", "gdn", "delta_rule_chunked_prefill_equals_one_call_and_the_reference"),
 ("delta-key-head-map", "gdn.metal", "const uint kh = h / (p.hv / p.hk);", "const uint kh = h % p.hk;", "gdn", "delta_rule_at_the_35b_layer_shape"),
 ("conv-wide-C-input-offset", "gdn.metal", "h[j] = pos >= 0 ? bf2f(x[p.s0 + (ulong)pos * C + c])", "h[j] = pos >= 0 ? bf2f(x[p.s0 + (ulong)pos * C + c + (C > 4096 ? 1 : 0)])", "gdn", "conv_at_the_35b_channel_count"),
 ("conv-tile-carry-off-by-one", "gdn.metal", "const long pos = t0 - (K - 1) + j;", "const long pos = t0 - (K - 1) + j + (t0 > 0 ? 1 : 0);", "gdn", "conv_at_the_35b_channel_count"),
 ("nvfp4-gemv-global-scale", "nvfp4.metal", "const float scale = gs[0] * NV_FOLD;", "const float scale = gs[0] * NV_FOLD * 1.02f;", "quant", "nvfp4_at_the_35b_projection_shapes"),
 ("fp8-mma-drop-last-k-tile", "fp8.metal", "for (uint k0 = 0; k0 < K; k0 += FP8_BK) {", "for (uint k0 = 0; k0 + FP8_BK < K; k0 += FP8_BK) {", "quant", "fp8_at_the_35b_projection_shapes"),
]
only = sys.argv[4:] 
res = []
for mid, f, old, new, binary, flt in M:
    if only and mid not in only: continue
    p = wt / "kernels/metal" / f
    src = p.read_text()
    if src.count(old) != 1:
        res.append((mid, f"BAD MUTATION: pattern found {src.count(old)} times")); continue
    p.write_text(src.replace(old, new))
    try:
        r = subprocess.run(["cargo", "test", "--release", "-p", "intarsia-backend-metal", "--test", binary, flt],
                           cwd=wt, env={**__import__("os").environ, "CARGO_TARGET_DIR": str(tgt)}, capture_output=True, text=True)
    finally:
        p.write_text(src)
    (out / f"{mid}.log").write_text(r.stdout + r.stderr)
    failed = r.returncode != 0 and "test result: FAILED" in r.stdout
    res.append((mid, "KILLED (test failed)" if failed else f"SURVIVED or build error (rc {r.returncode})"))
    print(mid, res[-1][1], flush=True)
(out / "summary.txt").write_text("\n".join(f"{a}: {b}" for a, b in res) + "\n")
print("\n".join(f"{a}: {b}" for a, b in res))
