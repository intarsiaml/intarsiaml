# Inkling-2-Mini attention on HIP (lane ink-hip-attn, package A, 2026-11-25)

Branch `ink/hip-attn` (origin/main `d8c3c7148` + `ink/hip` skeleton `6c5ee5d67`). Ops `InkKvPrologue` (70), `RelSinkAttn` (71);
stretch `InkAttnDecode` (75). Brief: `.worktrees/hip-swarm/COMMON.md`; port map: `docs/results/2026-11-25-inkling2-hip.md` (ink/hip).

## Briefs received (kept against context loss)

- 2026-11-25 ~21:45 EST, coordinator: the lead's skeleton is committed, `6c5ee5d67` on `ink/hip`; run `git merge ink/hip` now
  (done). Module shape and notes are in the last section of COMMON.md. Op 75 is optional for G2, so land 70 and 71 first; use the
  scalar split-KV `ink_attn_s1_k` + comb form first. Write this message into the results doc (this entry).

- 2026-11-25 ~22:10 EST, coordinator (from the ink-hip lead): when 70/71 pass their unit window, send "ready to merge at
  <commit>". Checkpoint on tatooine `/data/models/intarsia/Inkling-2-Mini` (dev image `/models/Inkling-2-Mini`). Copy (don't edit)
  `docs/results/2026-11-25-inkling2-hip/raw/window.sh` and `run.sh` from `ink/hip`. Set own `INTARSIA_TARGET_VOL=intarsia-cargo-target-<lane>`
  and own container name. No need to re-merge ink/hip. Lead merges lanes after their unit window, then admission + G1 + G2 in one window.
  Window rules unchanged (takes-down radiance, <= 20 min, <= 3 windows, none after 06:00, radiance up before release).

## Step 0

`git log --all -i --grep=hip` and a grep of every `ink/*` branch for `ink_kv_prologue` / `rel_sink_attn` under `kernels/hip` and
`crates/intarsia-backend-hip`: no earlier HIP Inkling attention work exists.

## Status

| Op | Launcher | State |
|---|---|---|
| `InkKvPrologue` 70 | `ink_kv_prologue_bf16` | ADMITTED vs cpu-ref, full sweep 117/117; rows-exact |
| `RelSinkAttn` 71 | `rel_sink_attn_bf16` (scalar split-KV, every T) | ADMITTED vs cpu-ref, full sweep 63/63; rows-exact |
| `InkAttnDecode` 75 | not built | optional (the model runs 70 + 71) |

## Design (kernels/hip/ink_attn.hip)

A straight port of CUDA's correctness form (`ink_prologue_k`, `ink_ring_write_k`, `ink_attn_s1_k<G>`, `ink_attn_comb_k`,
`ink_kv_write_k` of kernels/cuda/inkling2_attn.cu; CUDA's `INTARSIA_INK_DECODE=splitkv` path), same arithmetic and reduction
order: log2-domain scores, the rel bias (d_rel <= 32, zero at distances >= E), the sink once in the final normalisation, the
sigmoid output gate, SWA window via `ink_lo` (ring row `p % Cr`) and full layers (linear cache). HIP changes:
- wave32 shuffles `__shfl_xor(v, o, 32)`, run by every lane (the `!ok` skip stays after the score shuffle); no `__ldg`, PDL,
  cp.async; one-argument `__launch_bounds__`.
- `#pragma clang fp contract(off)` at file scope plus explicit `fadd` / `fmul` / `__builtin_fmaf` where CUDA used `_rn`
  intrinsics (hipcc's default contraction is fast-honor-pragmas), so `build.rs` needs no `NO_FP_CONTRACT_SOURCES` entry.
- The attention runs a call as slices of at most 64 rows in order (as CUDA's tile path does with 16): a slice reads the cache
  rows earlier slices wrote, so rows stay bitwise. Scratch is `min(T, 64) * H * 32 * (Dh + 2) * 4` B (25.6 MB at the real
  model), the same expression in `inkattn.rs::attn_scratch` and the launcher's check. Unsliced it would be 818 MB at a 2048 chunk.
- LDS: static `InkTg<3>` 7.8 KB + dynamic bias <= 3 x E x 4 = 12 KB (< 48 KB). Grids are sized from the capacity bound when
  `pos` is non-null (never from the host `pos0`).

## Tests (tatooine R9700, card 0; fleetq #11932 build, #11939 window, radiance down 21:47:58-21:48:32 = 34 s)

Raw log: `docs/results/2026-11-25-inkling2-hip-attn/raw/window1-11939.log`; stage script `raw/stage_unit.sh`.

| Test (`crates/intarsia-backend-hip/tests/ink_attn.rs`) | Result |
|---|---|
| `admission_full_sweep`, quick sweep (`INK_ATTN_QUICK=1`) | prologue 54/54 (max_abs 2.3e-3), attention 25/25 (max_abs 1.5e-3): ADMITTED |
| `admission_full_sweep`, full sweep (small shapes + real model: window 1024 / ring 1040, full linear to 10240, T 1/4/16/2048) | prologue 117/117 (max_abs 5.5e-3, rel_l2 <= 4.3e-4), attention 63/63 (max_abs 1.7e-3): ADMITTED at BF16_TOL (1.5e-2, 6e-3) |
| `rows_exact_real_shapes` (T 2..256 as one call vs T one-row calls, host and device positions; SWA ring wraps, full layers, pos 7900) | 63 cases bitwise |
| host tests on endor (stub build: `cargo test -p intarsia-backend-hip --lib --test quick_coverage`) | 31 + 1 pass (ABI header, coverage, quick cases) |

## Decode bench (T = 1, real shape 24 / 8 heads, Dh 128; `bench_decode`, eager, 500 launches)

KV bytes per call = 2 x 8 x keys x 128 x 2. Roofline = bytes / 627 GB/s (Neon's number, unverified). The bench re-reads the same
K / V every launch, so the 64 MB Infinity Cache serves part of it (16k keys = 67 MB): numbers are cache-warm, an upper bound
for real decode, where 24 layers' KV do not fit.

| Case | Prologue | Attention | KV GB/s | vs roofline |
|---|---|---|---|---|
| SWA, 1024 keys (4 MB) | 11.7 us | 20.7 us | 203 | 32% (latency bound: 8 x 16 x 1 blocks, 3 launches) |
| full, 4k keys | 12.3 us | 36.9 us | 454 | 72% |
| full, 8k keys | 12.1 us | 57.6 us | 583 | 93% |
| full, 16k keys | 12.6 us | 99.6 us | 674 | 107% (cache) |

Per decode token at ~4k context: 20 SWA layers x (11.7 + 20.7) + 4 full x (12.3 + 36.9) = 0.85 ms of attention. Speed levers
(after G2): fuse prologue + attention (op 75, removes a launch and the q / rel round trip), more splits on SWA rows (more
blocks for the 1024-key case), a combine-free path for one split.

## Next

- Lead: merge `ink/hip-attn` (HANDOFF.md). Op 75 is the stretch, not started.
