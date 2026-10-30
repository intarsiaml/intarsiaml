# HGV: small bf16 GEMVs and the shared expert over the grid; all-reduce LL path at 16 rows (HIP, R9700)

Lane hgv, 2026-10-30. Two changes from the Fable review of the HPROF profile
(`docs/results/2026-10-30-hprof-r9700-16stream-profile.md`, sections 5 and 8), Qwen3.6-35B-A3B NVFP4 on the R9700s (gfx1201). Every number is from
a run on tatooine under fleetq; raw logs, traces' reports and window logs are in `2026-10-30-hgv-hip-small-gemv-grid/raw/`.

## Summary

* Profiled 16-row step, TP=1: **40.24 -> 36.63 ms** (-3.6 ms, kernel busy 37.03 -> 33.42), TP=2 rank 0: **29.58 -> 25.50 ms** (-4.1 ms). The same-run main
  profile is `prof-tp1-hgvbase-final`; the HPROF profile of the same code gave 39.91 ms, so the profiler overhead is stable. In unprofiled terms the
  38.8 ms step becomes about 35.8 ms by the served-throughput ratio (370.0 -> 400.7 tok/s), about 35.2 ms by the profiled delta; the target was <= 35.5 ms.
* Served aggregate decode, median of 3 interleaved rounds, `--max-batch 16`, 1k prompts, gen 512: TP=1 16 streams **370.0 -> 400.7 tok/s (+8.3 %)**,
  8 streams +2.3 %, 4 and 1 stream unchanged; TP=2 16 streams **512.8 -> 585.6 tok/s (+14.2 %)**, 8 streams +3.3 %, 4 and 1 stream unchanged.
* Identity: bit for bit. Greedy text of 50 streams (1 / 8 / 16 streams, same prompt and 16 different prompts, 384 tokens, temperature 0) is byte identical
  between main and hgv at TP=1 and at TP=2; the TP=2 main run repeated is byte identical to itself. `tests/rows_exact.rs`, `tests/moe_block.rs`
  (grouped), `tests/comm.rs`, `ar_norm.rs`, `ar_push.rs` pass on the hgv tree.
* Both changes hold up and ship. One part of the first design did not: `shared_down_combine_rows_kernel` in groups of 4 tokens was slower (22.3 -> 25.4 us), so
  it keeps one block for all tokens (section 2).

## 1. What changed

1. `kernels/hip/matmul.hip`: `matmul_bf16_gemv_rows` / `logits_bf16_gemv_rows` with M > 4 rows launch `gemv_rows_grp_kernel<2, 4>`: groups of 4 rows along
   `blockIdx.y`, each group the body of `gemv_kernel<4, 2, 4>` (`gemv_body`, now with a `TAIL` variant that masks the missing rows of the last group of an
   M that is not a multiple of 4). `gemv_kernel<16, 2, 4>` (256 VGPR + 440 B scratch per thread) is no longer used for these shapes: the new kernel uses
   160 VGPR and no scratch. Only matrices of at most 4 MiB take it (the router 1 MiB, GDN in_a / in_b, the shared-expert gate): a bigger matrix
   (the 27B / 8B lm_heads at M = 5..8, the 8B projections) would be streamed once per group of rows, so those keep the old launch (they were
   unchanged in the timing, table 1b, and were slower in a first build without the gate: 4096x4096 at M = 5 went 58 -> 101 us).
2. `kernels/hip/moe_block_grouped.hip`: `shared_glu_rows_kernel` runs 4 tokens per block (`blockIdx.y` = group of tokens), 4x the blocks; 104 -> 80 VGPR.
   `shared_down_combine_rows_kernel` is unchanged (all tokens in one block): a first version with groups of 4 measured 25.35 us against 22.27.
3. `crates/intarsia-backend-hip/src/comm.rs`: `LL_MAX_DEFAULT` 64 KiB -> 128 KiB. At hidden 2048 the second all-reduce of a layer is an f32 partial of
   16 x 2048 x 4 B = 128 KiB at 16 rows: it was on the fence path (`ar2_kernel<2>`, 43.4 us), it now stays on the LL path (`ar2ll_kernel<2>`, 26.2 us). 128 KiB is the
   smallest value that covers it (`ll_fits` is `bytes <= limit`). Sizes that move from the fence to the LL path: 64 KiB..128 KiB, i.e. f32 9..16 rows
   and bf16 17..32 rows (a short prompt, or the last chunk of a longer one, at hidden 2048). A prefill chunk of 33+ bf16 rows (and every 1024- or 2048-row
   chunk: 4 / 8 MiB) is above the limit and unchanged; the 1-row decode (8 KiB) is unchanged. `tests/comm.rs` passes explicit limits, so it does not
   exercise the default; the served TP=2 identity run does.

Each (weight row, activation row) result is the same lane-ascending fma chain and warp sum as before; only the mapping of rows to blocks changed.

## 2. Per-kernel before / after

In situ, rocprofv3 `--kernel-trace`, 16 rows (`same16` + `div16`), steady steps, main = tree `hgvbase` (origin/main 3acb65809 kernels, tests only differ), same windows
shape as the HPROF profile (`scripts/hprof/window.sh`, `scripts/hgv/profcmp.py`).

TP=1:

step wall 40.24 -> 36.63 ms, kernel busy 37.03 -> 33.42, idle in step 3.21 -> 3.21, launches 1001 -> 1001

| kernel | before calls x us | before ms/step | after calls x us | after ms/step |
|---|---|---:|---|---:|
| `gemv_kernel<16` | 60 x 23.06; 40 x 24.33; 40 x 23.83 | 3.31 | - | 0.00 |
| `gemv_rows_grp_kernel` | - | 0.00 | 40 x 5.46; 60 x 3.62; 40 x 3.05 | 0.56 |
| `shared_glu_rows` | 40 x 28.95 | 1.16 | 40 x 13.85 | 0.55 |
| `shared_down_combine_rows` | 40 x 22.27 | 0.89 | 40 x 22.28 | 0.89 |

TP=2 (rank 0):

step wall 29.58 -> 25.50 ms, kernel busy 26.24 -> 22.16, idle in step 3.34 -> 3.35, launches 1075 -> 1075

| kernel | before calls x us | before ms/step | after calls x us | after ms/step |
|---|---|---:|---|---:|
| `gemv_kernel<16` | 100 x 22.73; 40 x 22.58 | 3.18 | - | 0.00 |
| `gemv_rows_grp_kernel` | - | 0.00 | 100 x 3.27; 40 x 5.18 | 0.53 |
| `shared_glu_rows` | 40 x 24.74 | 0.99 | 40 x 7.89 | 0.32 |
| `shared_down_combine_rows` | 40 x 21.48 | 0.86 | 40 x 21.72 | 0.87 |
| `ar2_kernel` | 40 x 43.39 | 1.74 | - | 0.00 |
| `ar2ll_kernel` | 40 x 11.70 | 0.47 | 40 x 26.24; 40 x 10.31 | 1.46 |

Resources (rocprofv3 kernel trace): `gemv_kernel<16,2,4>` 256 VGPR / 440 B scratch -> `gemv_rows_grp_kernel<2,4>` 160 VGPR / 0 scratch;
`shared_glu_rows_kernel` 104 -> 80 VGPR (`raw/kernel-resources.txt`). Step totals: TP=1 wall 40.24 -> 36.63 ms; TP=2 29.58 -> 25.50 ms.

The small GEMVs are where the estimate held (3.31 -> 0.56 ms of a 0.3 ms ceiling). The shared expert gained 0.6 of the 1.75 ms the profile named:
`shared_glu_rows` halves, but `shared_down_combine_rows` stays at 22.3 us (K = 512, one lane iteration per row: its time is not the token loop, so more blocks do
not help; I did not investigate further).

### 2b. Micro-timing, back-to-back launches (`tests/rows_exact.rs` `timing_gemv_rows` / `timing_mxfp4_rows`, quiet GPU)

Taken on the build before the final shared-down change; the bf16 GEMV and lm_head code did not change after.

### bf16 GEMV rows (us per launch, back-to-back, `matmul_bf16_gemv_rows`)

| matrix | M | main | hgv | ratio |
|---|---:|---:|---:|---:|
| bf16 1x2048 | 1 | 3.22 | 3.20 | 0.99 |
| bf16 1x2048 | 2 | 4.05 | 4.06 | 1.00 |
| bf16 1x2048 | 4 | 5.08 | 5.16 | 1.02 |
| bf16 1x2048 | 5 | 5.36 | 5.23 | 0.98 |
| bf16 1x2048 | 6 | 5.60 | 5.19 | 0.93 |
| bf16 1x2048 | 8 | 6.50 | 5.21 | 0.80 |
| bf16 1x2048 | 10 | 20.72 | 5.18 | 0.25 |
| bf16 1x2048 | 12 | 15.38 | 5.19 | 0.34 |
| bf16 1x2048 | 14 | 21.43 | 5.21 | 0.24 |
| bf16 1x2048 | 16 | 27.45 | 5.19 | 0.19 |
| bf16 256x2048 | 1 | 5.01 | 4.98 | 0.99 |
| bf16 256x2048 | 2 | 5.55 | 5.58 | 1.01 |
| bf16 256x2048 | 4 | 6.28 | 6.37 | 1.01 |
| bf16 256x2048 | 5 | 6.66 | 6.29 | 0.94 |
| bf16 256x2048 | 6 | 7.03 | 6.33 | 0.90 |
| bf16 256x2048 | 8 | 8.06 | 6.44 | 0.80 |
| bf16 256x2048 | 10 | 22.71 | 6.61 | 0.29 |
| bf16 256x2048 | 12 | 17.65 | 6.86 | 0.39 |
| bf16 256x2048 | 14 | 22.76 | 6.99 | 0.31 |
| bf16 256x2048 | 16 | 29.17 | 6.95 | 0.24 |
| bf16 32x2048 | 1 | 3.28 | 3.29 | 1.00 |
| bf16 32x2048 | 2 | 4.70 | 4.70 | 1.00 |
| bf16 32x2048 | 4 | 5.32 | 5.64 | 1.06 |
| bf16 32x2048 | 5 | 5.76 | 5.78 | 1.00 |
| bf16 32x2048 | 6 | 6.15 | 5.69 | 0.93 |
| bf16 32x2048 | 8 | 7.20 | 5.70 | 0.79 |
| bf16 32x2048 | 10 | 22.76 | 5.64 | 0.25 |
| bf16 32x2048 | 12 | 17.33 | 5.62 | 0.32 |
| bf16 32x2048 | 14 | 22.81 | 5.64 | 0.25 |
| bf16 32x2048 | 16 | 29.20 | 5.63 | 0.19 |
| bf16 4096x4096 | 1 | 55.93 | 55.90 | 1.00 |
| bf16 4096x4096 | 2 | 56.82 | 56.82 | 1.00 |
| bf16 4096x4096 | 4 | 57.58 | 57.65 | 1.00 |
| bf16 4096x4096 | 5 | 58.25 | 58.29 | 1.00 |
| bf16 4096x4096 | 6 | 58.80 | 58.83 | 1.00 |
| bf16 4096x4096 | 8 | 60.75 | 60.75 | 1.00 |
| bf16 4096x4096 | 10 | 134.87 | 135.11 | 1.00 |
| bf16 4096x4096 | 12 | 111.09 | 112.62 | 1.01 |
| bf16 4096x4096 | 14 | 142.08 | 144.23 | 1.02 |
| bf16 4096x4096 | 16 | 215.56 | 218.57 | 1.01 |

### lm_head 248320x5120 bf16 -> f32 (`logits_bf16_gemv_rows`, us)

| M | main | hgv | ratio |
|---:|---:|---:|---:|
| 1 | 4242.8 | 4242.7 | 1.000 |
| 2 | 4263.0 | 4262.9 | 1.000 |
| 3 | 4280.2 | 4279.4 | 1.000 |
| 4 | 4343.2 | 4342.2 | 1.000 |
| 5 | 4436.6 | 4433.8 | 0.999 |
| 6 | 4458.8 | 4455.8 | 0.999 |
| 8 | 4505.4 | 4508.3 | 1.001 |
| 12 | 4607.8 | 4584.9 | 0.995 |
| 16 | 4638.6 | 4640.1 | 1.000 |


## 3. A/B, served aggregate decode (tok/s)

`scripts/hgv/window.sh` (MODE=perf): one fresh server per arm (`--max-batch 16 --spec off`, TP=2 `--tp 2 --devices 0,1`), `localeval speed --sizes 1024 --gen 512
--concurrency 4,8,16`, 3 rounds, arm order interleaved (round 2 reversed), main and hgv built from the same toolchain and image. The "1" row is the
single-stream 1k decode of the same server (max-batch 16, not a max-batch 1 server).

| TP | streams | main median | hgv median | change | main rounds | hgv rounds |
|---|---:|---:|---:|---:|---|---|
| tp1 | 1 | 164.2 | 164.4 | +0.1 % | 164.2 / 162.1 / 167.2 | 164.1 / 164.4 / 164.6 |
| tp1 | 4 | 289.1 | 290.4 | +0.4 % | 290.0 / 289.1 / 287.2 | 291.7 / 290.4 / 288.3 |
| tp1 | 8 | 354.4 | 362.4 | +2.3 % | 354.0 / 354.4 / 356.1 | 362.2 / 362.5 / 362.4 |
| tp1 | 16 | 370.0 | 400.7 | +8.3 % | 370.0 / 368.8 / 370.1 | 400.8 / 399.7 / 400.7 |
| tp2 | 1 | 192.7 | 192.7 | +0.0 % | 192.2 / 192.7 / 192.7 | 192.7 / 192.7 / 192.8 |
| tp2 | 4 | 358.2 | 358.7 | +0.1 % | 358.2 / 359.3 / 357.1 | 350.5 / 359.6 / 358.7 |
| tp2 | 8 | 486.0 | 502.1 | +3.3 % | 485.6 / 487.2 / 486.0 | 502.1 / 503.9 / 502.0 |
| tp2 | 16 | 512.8 | 585.6 | +14.2 % | 512.2 / 512.8 / 513.9 | 585.6 / 586.6 / 583.3 |

Rounds agree within about 1 % (TP=2 4 streams, round 1 hgv: 350.5, an outlier; the 4-row path is not touched by either change).

## 4. Identity

* `scripts/hgv/window.sh` MODE=ident + `scripts/hgv/ident.py`: per server, 1 / 8 / 16 streams of one prompt (no nonce) and 1 / 8 / 16 streams of different
  ~1k-token prompts cut from docs, gen 384 forced tokens, temperature 0; `scripts/hgv/ident_cmp.py` compares the text of every stream.
* TP=1 main vs hgv: 50 / 50 streams byte identical (84 876 chars). TP=2 main vs hgv: 50 / 50 (78 731 chars); TP=2 main vs main again: 50 / 50.
  (The TP=1 main run was run twice but the second overwrote the first's file, a bug fixed afterwards; so the TP=1 main-vs-main repeat is not recorded. TP=1 vs
  TP=2 differ in every stream, for main as well as hgv: different reduction order, expected.)
* Kernel level: `rows_exact` (bf16 matmul and logits rows M = 1..16, solo kernel vs rows kernel bitwise, reversed order, shapes (256,2048), (32,2048), (1,2048),
  (16,2048), (130,2056), 4096x4096 ... and the scalar path), `moe_block_grouped_*` (T = 1..16 tokens, residues of T mod 4 included at the 35B shape; token t bitwise the T = 1
  block), `comm`, `ar_norm`, `ar_push`: all pass on the final build (`raw/gate-0146`, `raw/fleetq-gate-final.log`). The first gate run was on a stale hiptest binary
  (a build raced the tree sync) and is kept under `raw/stale-first-gate` only because its timing showed the missing size gate; everything above is from the rebuilt tree.

## 5. Limits

* Main is the build of commit 486e0812d: origin/main 3acb65809 plus test-file edits only (`tests/rows_exact.rs`, `tests/moe_block.rs`); kernels and engine are main's.
* Profiled numbers are one profile per arm (rank 0 for TP=2); the served A/B has 3 rounds. The 38.8 ms unprofiled figure is the HPROF doc's.
* The all-reduce change was not run at a model with another hidden size; for hidden 4096 or 5120 (the 27B on this engine) the 128 KiB limit moves a different set of row counts
  (bf16 8..16 rows at 4096), and nothing here measures that.
* `shared_down_combine_rows` (0.89 ms of the step) is untouched.

## Reproduce

`BASE=486e0812d scripts/hgv/build.sh` (CPU) builds both trees; `scripts/hgv/gate.sh` (gates and micro-timing), `scripts/hgv/chain-prof.sh`, `scripts/hgv/chain-ab.sh`
(each run from the Mac in the worktree, every job under `fleetq run --takes-down radiance --quiet gpu,cpu`), `scripts/hgv/perftab.py`, `profcmp.py`, `timetab.py`.
