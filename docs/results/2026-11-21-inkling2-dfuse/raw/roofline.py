"""Roofline of Metal decode at the tip from job L's multi-dispatch class benches (raw/L_*), calls per token from probe P's
trace (same code), bytes from raw/roof_bytes.py (native) and the P12 bench's GB/s (bytes the tip reads). 614 GB/s floor."""
BW = 614e9
# (class, us per call, calls per token, MB per call read at the tip, MB per call native bf16 / as stored)
p12_inproj = 532.0 * 56.36e-6 * 1e3      # GB/s x us -> MB per call (P12 bench S=1)
p12_inproj3 = 524.0 * 57.22e-6 * 1e3
p12_wo = 507.0 * 20.96e-6 * 1e3
p12_head = 576.1 * 1201.85e-6 * 1e3
kv_mb = 48 * 4096 * 150 / 1e6 / 48       # KV read per visit at a mean context of 150 (short prompt), MB
rows = [
    ("attn projections: InkInProj S=1 (p12)", 56.36, 20, p12_inproj, 1902.2 / 48),
    ("attn projections: InkInProj S=3 (p12)", 57.22, 28, p12_inproj3, 1902.2 / 48),
    ("attn projections: wo (p12)", 20.96, 48, p12_wo, 679.5 / 48),
    ("attention core: fused decode, swa (ctx <= 512)", 23.09, 40, kv_mb, kv_mb),
    ("attention core: fused decode, full (ctx <= 512)", 24.34, 8, kv_mb, kv_mb),
    ("experts NVFP4 (gate+up, down)", 81.8, 44, 8 * 4.478976, 8 * 4.478976),
    ("experts MXFP8 (gate+up, down)", 125.9, 4, 8 * 8.211456, 8 * 8.211456),
    ("router (ink_route)", 8.53, 48, 28.3 / 48, 28.3 / 48),
    ("unembed (p12)", 1201.85, 1, p12_head, 921.9),
    ("argmax (v4t)", 13.48, 1, 0.8, 0.8),
    ("ctrl: ConvMix attn plain", 6.34, 24, 0.0, 0.0),
    ("ctrl: ConvMix attn + blend S=1 (4 disp)", 19.92, 23, 0.0, 0.0),
    ("ctrl: ConvMix mlp plain Q3 (bench shape > model)", 10.35, 26, 0.0, 0.0),
    ("ctrl: ConvMix mlp + blend S=3 (4 disp; bench shape > model)", 31.19, 23, 0.0, 0.0),
    ("ctrl: LoopBlend (4 disp, S=3 bound)", 19.76, 2, 0.0, 0.0),
    ("ctrl: AttnResMix msgs (per-dispatch stream, upper bound)", 22.81, 4, 0.0, 0.0),
    ("other: rmsnorm x2, embedding, small (per-dispatch stream)", 4.5, 4, 7.3 / 4, 7.3 / 4),
]
CTRL_MB = 113.2  # controllers' weights per token (spread over the ctrl rows)
step = 13.31
tot_t = tot_b = tot_n = 0.0
cls = {}
print("| class | us / call | calls / token | ms / token | MB / token read (tip) | GB/s | floor ms (614) | gap ms |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for name, us, n, mb, mbn in rows:
    ms = us * n / 1e3
    mbt = mb * n
    if name.startswith("ctrl"):
        mbt = CTRL_MB * (us * n) / sum(u * c for nm, u, c, _, _ in rows if nm.startswith("ctrl"))
    fl = mbt * 1e6 / BW * 1e3
    gbs = mbt / ms if ms else 0
    tot_t += ms; tot_b += mbt
    print(f"| {name} | {us:.2f} | {n} | {ms:.3f} | {mbt:.0f} | {gbs:.0f} | {fl:.3f} | {ms - fl:.3f} |")
    key = name.split(":")[0].split(" (")[0]
    c = cls.setdefault(key, [0.0, 0.0]); c[0] += ms; c[1] += fl
print(f"| **sum of the classes** | | | **{tot_t:.3f}** | **{tot_b:.0f}** | | **{tot_b * 1e6 / BW * 1e3:.3f}** | **{tot_t - tot_b * 1e6 / BW * 1e3:.3f}** |")
print(f"| measured step (job L reps 1-3 median, busy {13.281:.3f}, idle 0.001) | | | {step:.3f} | | | | |")
print(f"| **not seen by the class benches** (step - sum) | | | **{step - tot_t:.3f}** | | | | |")
print()
for k, (ms, fl) in sorted(cls.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
    print(f"CLASS {k}: {ms:.3f} ms, floor {fl:.3f}, gap {ms - fl:.3f}")
print(f"dispatch cost: 543 x 0.88 us (near-empty kernel, serial encoder) = {543 * 0.88 / 1e3:.3f} ms per step (inside the class times)")
print(f"floor at native bytes 5.492 GB: {5.492e9 / BW * 1e3:.3f} ms; at the tip's bytes {tot_b / 1e3:.3f} GB: {tot_b * 1e6 / BW * 1e3:.3f} ms")
