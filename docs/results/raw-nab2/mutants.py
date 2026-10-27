#!/usr/bin/env python3
"""Plant one mutant at a time in the ShortConv Metal kernel / launcher, rebuild, run `kernels test --op short_conv`, restore.
Each mutant must make the admission FAIL (non-zero exit / not ADMITTED)."""
import subprocess, sys, os
os.chdir('/opt/code/intarsiaml/.worktrees/nab2')
M = [
 ("silu-left-on", "kernels/metal/gdn.metal",
  "y[p.s2 + (ulong)t * C + c] = f2bf(bf2f(x[p.s0 + (ulong)t * 3 * C + C + c]) * acc);",
  "y[p.s2 + (ulong)t * C + c] = f2bf(bf2f(x[p.s0 + (ulong)t * 3 * C + C + c]) * gdn_silu(acc));"),
 ("state-from-x-not-Bx", "kernels/metal/gdn.metal",
  "state[(ulong)c * (K - 1) + j] = e < K - 1 ? state[(ulong)c * (K - 1) + e] : gdn_conv_in<SHORT>(x, skip, e - (K - 1), C, c);",
  "state[(ulong)c * (K - 1) + j] = e < K - 1 ? state[(ulong)c * (K - 1) + e] : (SHORT ? bf2f(x[skip + (e - (K - 1)) * 3 * C + 2 * C + c]) : gdn_conv_in<SHORT>(x, skip, e - (K - 1), C, c));"),
 ("reversed-taps", "kernels/metal/gdn.metal",
  "for (uint j = 0; j < K; ++j) taps[j] = bf2f(w[p.s1 + (ulong)c * K + j]);\n#pragma unroll\n    for (uint j = 0; j < K - 1; ++j) {\n        const long pos",
  "for (uint j = 0; j < K; ++j) taps[j] = bf2f(w[p.s1 + (ulong)c * K + (SHORT ? K - 1 - j : j)]);\n#pragma unroll\n    for (uint j = 0; j < K - 1; ++j) {\n        const long pos"),
 ("multi-tile-state-pass-skipped", "crates/intarsia-backend-metal/src/gdn.rs",
  "self.dispatch(state_pass, &[&x, &st], &p, Grid::Threads(c))?;",
  "if !short { self.dispatch(state_pass, &[&x, &st], &p, Grid::Threads(c))?; }"),
 ("gate-by-B-not-C", "kernels/metal/gdn.metal",
  "f2bf(bf2f(x[p.s0 + (ulong)t * 3 * C + C + c]) * acc)",
  "f2bf(bf2f(x[p.s0 + (ulong)t * 3 * C + c]) * acc)"),
 ("product-uses-C-not-x", "kernels/metal/gdn.metal",
  "return bf2f(r[c]) * bf2f(r[2 * C + c]);",
  "return bf2f(r[c]) * bf2f(r[C + c]);"),
]
only = sys.argv[1:]
res = []
for name, path, old, new in M:
    if only and name not in only: continue
    src = open(path).read()
    assert old in src, (name, "pattern missing")
    open(path, 'w').write(src.replace(old, new, 1))
    try:
        b = subprocess.run(["fleetq","run","--host","endor","--expect","5","--load","cpu","--label",f"nab2 mutant build {name}","--","cargo","build","--release","--features","metal","-p","intarsia-cli"],capture_output=True,text=True)
        if b.returncode: print(name, "BUILD FAILED", b.stderr[-500:], b.stdout[-500:]); res.append((name,"build-failed")); continue
        r = subprocess.run(["fleetq","run","--host","endor","--expect","3","--quiet","gpu","--label",f"nab2 mutant test {name}","--","sh","-c","./target/release/intarsia kernels test --backend metal --op short_conv 2>&1; echo EXIT=$?"],capture_output=True,text=True)
        out = r.stdout
        open(f"raw/nab2/mutant-{name}.txt","w").write(out)
        killed = "FAIL" in out and "EXIT=0" not in out.split("EXIT=")[-1][:3] or ("ADMITTED" not in out)
        line = [l for l in out.splitlines() if l.startswith("short_conv")]
        print(name, "KILLED" if killed else "SURVIVED", line)
        res.append((name, "killed" if killed else "SURVIVED"))
    finally:
        open(path, 'w').write(src)
print(res)
