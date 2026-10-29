# MDEC: Metal decode attribution, and what did not move it (2026-10-29)

Lane `mdec`, branch `lane-mdec/metal-decode`, base `main` `7aebefb83` (#250). endor (Apple M5 Max, 128 GB, macOS 27.0), Metal,
Qwen3.6-35B-A3B NVFP4 (MTP off, the Metal default; MTP measured opt-in) and Qwen3-8B BF16. Raw logs and scripts:
[raw/](2026-10-29-mdec-metal-decode/raw/). Every timed run was in a `--quiet gpu,cpu` fleetq slot (jobs below).

## Result

**One shippable change: Metal takes `--max-batch` up to 16** (was 8). 9..16 rows run the multi-row GEMVs as two launches of at
most 8 rows (lane mr3's chunking, `lane-mr3/cap16-experiment`, rebased) and the per-pair MoE kernels as one launch; every row is
bitwise the solo step (gates below). Nothing changes at `--max-batch` 8 or below or for a single stream (same kernels, same plan).

Served 35B, localeval `--sizes 1024 --gen 512 --reps 1`, `--max-batch 8` (main's cap) vs `16`, a fresh server per arm,
interleaved (job #7076: 8, 16, 16, 8, 8, 16), aggregate decode tok/s:

| streams | `--max-batch 8` r1 / r2 / r3 | `--max-batch 16` r1 / r2 / r3 | median change |
|---:|---|---|---:|
| 1 | 107.3 / 118.1 / 113.3 | 109.0 / 113.0 / 113.3 | 0 (same path) |
| 4 | 217.4 / 248.0 / 227.1 | 219.7 / 222.5 / 230.6 | -2.0 % (same path; pairs +1.1 / -10.3 / +1.5, the r2 `8` arm ran fast on every cell) |
| 8 | 273.4 / 283.1 / 285.7 | 277.0 / 282.6 / 286.8 | -0.2 % (same path) |
| **16** | 276.4 / 281.7 / 283.8 | 295.9 / 298.2 / 304.8 | **+5.9 %** (pairs +7.1 / +5.9 / +7.4) |

The earlier window (#7061, N = 8 and 16 only, order 8/16, 16/8, 8/16) gave N = 16 309.3 / 288.0 / 289.8 vs 294.5 / 298.1 / 304.9
(+2.9 % median, pairs -4.8 / +3.5 / +5.2). Over all six pairs the median is +5.5 %. Time to first token at 16 streams drops
from 8.6 s to 2.7 s (the second wave no longer waits for the first). The gain is the second wave's prefill overlapping
decode, not a faster step: a 16-row step is 37.4 ms against 20.5 ms for 8 rows (section 2).

Single stream: no lever found that moves it. The 35B decodes 124.5 / 120.7 / 109.5 tok/s at 1k / 4k / 16k (47 % of its
~2.4 GB/token floor at 4k); the 8B is at 98 % of its floor. Section 1 is the attribution, section 2 what was tried (all
bitwise, all within noise or slower), section 3 what is left.

## 1. Attribution: where one 35B decode token goes

Method: `generate`, 129 tokens after a 973 / 4096 / 16384-token prompt (the longctx prompts), three runs per context:
plain (tok/s), `INTARSIA_METAL_PROFILE=cb` (step period, GPU busy / idle, dispatches) and `cb` with
`INTARSIA_METAL_COMMIT_EVERY=1` (one command buffer per dispatch, labelled, in a continuous stream). In the `ce1` run consecutive
command buffers can overlap on the GPU (Metal orders them only by buffer hazards), so a dispatch is charged only the time
after the previous one ended ("exclusive"); the exclusive sums come to 8.0 / 8.15 / 9.05 ms per step, within 2 % of the plain
periods, so the split is usable. Overlapped time lands on the earlier dispatch: the GDN qkv / z pair is reported together.

Step (cb trace): 1k 8.13 ms (GPU busy 7.87, idle 0.26), 4k 8.33 (8.10 / 0.23), 16k 9.41 (9.05 / 0.36); 770 dispatches per
token at every context. Plain decode 124.5 / 120.7 / 109.5 tok/s. The host is not the problem: idle is 3-4 % of the step.

Bytes per token at 4k: 1.65 GB of labelled weights (fp8 projections, shared experts, lm_head, router), about 0.57 GB of routed
experts (8 x 1.77 MB x 40 layers; the MoE launches carry no byte label), 0.13 GB GDN state (f32, read + write) and 0.08 GB KV,
about **2.4 GB**, so the 614 GB/s floor is **3.9 ms** (254 tok/s). Measured 8.33 ms: 47 % of the floor.

| part (35B, 4k, per token) | launches | us (exclusive) | share | bytes | GB/s | at 560 GB/s |
|---|---:|---:|---:|---:|---:|---:|
| fp8 GDN qkv + z, attn q+gate (8192x2048, 4096x2048) | 70 | 2247 | 27.6 % | 923 MB | 411 | 1648 |
| fp8 out projections (2048x4096: GDN out, attn o) | 40 | 1164 | 14.3 % | 336 MB | 288 | 599 |
| fp8 attn k / v (512x2048) | 20 | 34 | 0.4 % | 21 MB | (overlap) | 38 |
| routed experts (gate-up, down) | 80 | 1438 | 17.6 % | ~566 MB | ~394 | 1011 |
| shared expert (nvfp4 gate/up, down; silu, combine) | 270 | 646 | 7.9 % | 63 MB | 98 | 113 |
| lm_head (nvfp4 248320x2048, f32) + argmax | 2 | 508 | 6.2 % | 254 MB | 520 | 454 |
| attention decode (split-KV) | 10 | 449 | 5.5 % | 84 MB | 187 | 150 |
| GDN delta rule, conv, a/b pair, gated norm | 121 | 635 | 7.8 % | ~134 MB | - | 240 |
| router pair + top-k | 80 | 515 | 6.3 % | 42 MB | - | 75 |
| rmsnorm_add, qk_norm_rope_kv, gate/sigmoid, embedding | 112 | 520 | 6.4 % | 0 | - | ~0 |
| **total** | 764 | **8156** | | **~2.4 GB** | 294 | **~4300** |

Context: attention decode is 244 us at 1k, 449 at 4k, 1326 at 16k (the whole 1k-to-16k slowdown, 8.0 to 9.05 ms). The
rest does not move with context.

Reading it: the fp8 GEMVs are 42 % of the step at 371 GB/s and are the largest gap (about 1.15 ms against 560 GB/s). The
second gap is the ~380 launches that move no bytes (norms, top-k, gates, conv, combine), about 1.3 ms. Routed experts are
the third, about 0.4 ms.

**Qwen3-8B BF16, 4k:** 28.0 ms GPU busy per token, 0.05 ms idle, 364 dispatches, 35.8 tok/s. Weights + KV are about 16.9 GB
per token at 4k, so the floor is about 27.5 ms: **98 %**. The GEMVs run at 588-597 GB/s (the spike's best GEMV reached
91-96 % of 614). The 8B's +3 % bar is not reachable by kernel work, so this lane did none there. (The 8B `ce1` trace is not
used: `COMMIT_EVERY=1` did not take effect on its path, 620 records for 129 steps.)

**MTP (opt-in), 35B 4k:** 165.6 tok/s (vs 120.7 off), 87.1 % of drafts accepted, 2.72 tokens per verify pass, about
16.4 ms per pass. The cb trace shows **2.2 ms of GPU idle per pass (12.6 %)**: 1.3 ms after the verify pass's last buffer
(host-side acceptance / rollback before the drafts are encoded) and about 0.5 ms over three short draft syncs. At 1k: 168.9
tok/s, 76.2 % accepted.

**Batched (served, `--max-batch 8`):** a decode-only iteration at 8 rows is 20.5 ms (encode 1.2 ms, overlapped; GPU wait
19.3 ms; GPU idle gap 0.04 ms), i.e. 390 tok/s of step throughput; served aggregate at 8 streams is 275-330 tok/s (gen 256 /
512; between-server spread is about 10 %). The difference is the prefill of arriving requests (about 260 ms per 1k prompt,
which stalls the decoding rows) inside the measured window, not host overhead. **N = 16 flattens because Metal's cap is 8**
(`Backend::batch_row_cap` default, `SERVE_MAX_BATCH`): 16 streams run as two waves of 8.

## 2. What was tried

All of these keep the bits (greedy ids identical to main on the 4k prompt, md5 of the 257-token output, every arm).

| lever | change | result |
|---|---|---|
| fp8 one-row GEMV, rows per simdgroup / loads in flight | RS 1 / 2 / 4 x UN 2 / 4 / 8 (the lane's blocks and chain do not depend on them) | no gain: 2048x4096 27.9-29.2 us in every form (exclusive, ce1), decode 116-121 vs 119-122 tok/s |
| fp8 one-row GEMV on the `_xd` form | q(x) prepared once into scratch, no per-threadgroup staging | the GEMVs get faster (8192x2048 47.5 to 41.1 us) but the extra prep dispatch costs more: **-2.0 %** (119.8 vs 122.3 tok/s, 3 interleaved rounds, all 3 the same way) |
| fp8 one-row GEMV, loads before staging | `_m1` with the first weight loads issued before the q(x) staging barrier | no gain: 115.5-116.0 tok/s vs 115.5-117.2 off and 115.1-118.2 main (3 interleaved rounds, #7076); dropped |
| MoE decode, all expert loads first | gate-up and down kernels that issue every weight block load before the first use | not better in the one clean pair (98-118 vs 120-121 tok/s, noisy); dropped |
| cap 16 for batched decode | lane mr3's chunking (two <= 8-row launches of the multi-row GEMVs) + the per-pair MoE kernels in one launch at 9..16 rows | N = 16 served +5.9 % (median of 3 interleaved pairs, #7076), +5.5 % over 6 pairs; N <= 8 unchanged: **kept** (the PR) |
| 16-row fp8 simdgroup-matrix tile (R = 2) | one launch for 9..16 rows | slower than two 8-row chunks: 2.40x the B = 8 time vs 2.0x; dropped |

**Why B = 16 does not pay:** per-kernel at B = 8 vs B = 16 (real-traffic step, ce1, ms per step): routed experts 7.96 to 14.20
(x1.8: about 1.8x the distinct experts), fp8 rows 5.49 to 11.50 (x2.1; the multi-row fp8 kernels are ALU-bound per row),
lm_head 2.05 to 4.09 (x2.0, chunked: read twice), GDN delta rule 1.10 to 2.23, attention 0.90 to 1.67. The step goes 20.2 to
37.9 ms: 2x the rows for 1.9x the time. The batched kernels are compute-bound per row, so more rows buy almost nothing.

## 3. What is left to reach the floor (35B, 4.3 ms at 560 GB/s vs 8.3 ms)

1. **fp8 q(x) produced upstream** (about 0.3 ms, 4 %): the `_xd` run shows the staging costs about 6 us per 8192x2048 GEMV,
   but a separate prep dispatch costs more than it saves. The producer (rmsnorm_add / gated norm) writing q(x) as well would
   keep the saving without the dispatch. Needs a model-level op change; bitwise.
2. **The ~380 byte-free launches** (about 1.3 ms): norms, top-k (8 us each), conv, combine, gates. The mdl folds (2026-10-22)
   fused them into the GEMVs and lost (each threadgroup redid the norm); folding them into each other (combine + next
   rmsnorm_add, router top-k + combine weights) does not have that cost. 
3. **fp8 GEMV bandwidth** (about 0.6 ms): 371 GB/s in-stream vs 560; neither the geometry nor the staging explained it here.
4. **MTP host gap** (opt-in mode): 2.2 ms idle per verify pass; device-side acceptance would be about +12 % on MTP decode.
5. **Batched decode:** multi-row kernels on simdgroup matrices / NAX with a row-exact proof (as mm did for fp8 at 8 rows),
   and a 16-row lm_head, before cap 16 can pay.

## 4. Gates

The change is bitwise (no numerics moved), so the Metal gates are the row-exact ones; KL / top-1 are untouched by
construction (single-stream and `--max-batch` <= 8 run the same kernels and plan as main).

| gate | result | log |
|---|---|---|
| `tests/rows_exact.rs` (Metal kernels, M = 2..16: fp8 / nvfp4 / bf16 pair / MoE rows bitwise the one-row calls; registrations pinned 2..16; row-parallel kernels) | 7 / 7 pass (#7061, #7076) | `raw/w3/rows_exact.log`, `raw/w4/rows_exact.log` |
| negative control `INTARSIA_METAL_NEGCTL=fp8_mmt` | fails the fp8 gate, as it must (#7061) | `raw/w3/window.log` |
| `real_35b_batched_decode_is_bitwise_the_solo_step`, N = 2..9, 11, 12, 13, 15, 16, 130 steps each, beside prefill chunks | **14430 (row, step) pairs bitwise identical** (#7080) | `raw/w4/gc-bitwise.log` |
| `real_35b_batched_engine_equals_the_local_engine`, 4 / 8 / 16 slots | pass (#7076) | `raw/w4/gc.log` |
| `batch_metal` (Qwen3-8B dense batching) | pass (#7076) | `raw/w4/dense.log` |
| greedy ids, 35B 4k 257 tokens, main vs branch | identical md5 every arm (#7076) | `raw/w4/window.log` |
| `cargo test --workspace --release` (endor) | __WS__ | `raw/w4/wstest.log` |
| long prompts 2.6k / 4k / 16k at 16 slots (`--ignored real_35b_batched_engine_long_prompts_*`) | not run (window budget) | |

`real_lfm2_batched_decode_is_bitwise_the_solo_step` fails under `INTARSIA_REQUIRE_GPU=1` only because `INTARSIA_LFM2_DIR` is
not set on endor (it is SKIPPED otherwise); not related. The first run of the bitwise gate (#7076) failed on a test bug carried
over from mr3's experiment (a 1-token prompt: nothing to prefill); fixed (prompt lengths 41 and 23), then it passed (#7080).

## 5. fleetq jobs

| job | what | slot |
|---|---|---|
| #7023, #7034, #7039, #7055, #7060, #7064, #7075, #7079 | builds (branch, `quant_bw`, main baseline) | `--load cpu` |
| #7027 | window 1 (first try): died in its first second (bash 3.2 empty-array bug), no GPU work | `--quiet gpu,cpu` |
| **#7030** | **window 1**: attribution (35B 1k/4k/16k plain + cb + ce1, MTP, 8B 4k, served N = 1..16 + host / cb profile) | `--quiet gpu,cpu` |
| #7036 | short check 1: fp8 / MoE variants, 35B 4k (MoE load-first on in most arms: confounded) | `--quiet gpu,cpu` |
| #7041 | short check 2: `quant_bw` fp8 variants (unusable: 2x below in-stream, 40-170 us swings) | `--quiet gpu,cpu` |
| **#7058** | **window 2**: rows_exact M <= 16, B = 16 step timing, fp8 narrow forms A/B + ce1, served cap 16 first look | `--quiet gpu,cpu` |
| **#7061** | **window 3**: rows_exact + negative control, fp8 M = 1 `_xd` A/B + ce1, B = 8 / 16 per-kernel profile, served 8 vs 16 x 3 | `--quiet gpu,cpu` |
| **#7076** | **window 4**: gates, fp8 load-first A/B vs main, served 8 vs 16 x 3 (N = 1..16) | `--quiet gpu,cpu` |
| #7080 | short check 3: bitwise batched gate N <= 16 | default |
| #7083 | workspace tests | default |
