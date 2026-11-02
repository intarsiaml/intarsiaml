# K4FP8: HIP fp8 rows kernel at long K, and why it was slow at 16 rows (R9700, Qwen3.6-35B-A3B NVFP4)

Lane k4fp8, 2026-10-30. Branch `lane-k4fp8/main` on main `78de87bf7` (HFP8 #274 and WMR2 #275 merged). Target: the open item of
HFP8 (`2026-10-30-hfp8-hip-fp8-rows16.md`, section 5): the K = 4096 projections took 54 us at 16 rows against 17 us at one row.

## Summary

* **16 streams, aggregate decode: TP=1 462.9 -> 475.6 tok/s (+2.7 %), TP=2 651.3 -> 660.6 tok/s (+1.4 %).**
  - Medians of 3 interleaved rounds; `--max-batch 16`, 1k prompts, gen 512, spec off.
  - 8 streams: +2.5 % (TP=1) / +2.7 % (TP=2). 4 streams: +0.7 % / +1.1 %.
  - One stream is unchanged, and so is `--max-batch 1` at TP=1: 164.2 -> 164.4 tok/s.
  - The 16-row step goes from 32.3 to 31.4 ms (TP=1) and from 23.7 to 23.1 ms (TP=2).
  - Against the lane's ~2 % bar at 16 streams, TP=1 clears it and TP=2 does not. K = 4096 exists only at TP=1: at TP=2 the row-split
    `out_proj` / `o_proj` shards are 2048x2048, so the 16-warp long-K lever never runs there. At TP=2 the change is a smaller,
    bit-exact gain with no regression.
* **Bitwise identical to main.**
  - Every row of every M = 1..16 equals the one-row kernel's output: 1248 checks on the final kernel, ragged shapes and W8A16
    included.
  - The repo's fp8 row tests pass.
  - `real_35b_batched_decode_is_bitwise_the_solo_step` passes at TP=1 and TP=2.
  - 16 concurrent greedy streams give byte-identical text on main and on this branch at TP=1 and TP=2.
  - No KL gate is needed.
* **The cause, for K = 4096 and for every shape at M >= 8: the kernel did not overlap its weight stream with its compute.**
  - The weight loads sat in branches and the prefetch was conditional, so the compiler put an `s_wait_loadcnt 0x0` at the top of the
    compute. Every tile waited for the next tile's prefetched weights before it started.
  - At K = 4096, the per-block staging of 16 x 4096 activations for only 16 weight rows was about half the time on top of that
    (ablation: no staging 58.6 -> 32.5 us).
* **What changed** (all in `fp8_gemv_rows_kernel`):
  - branch-free loads and an unconditional prefetch (the compiler now waits for the current batch only);
  - activation rows in groups of 4, with the next group's LDS reads in flight;
  - the hardware fp8 conversion for the weights (an exact 2^8 rescale);
  - 16 warps per block at K > 2048.
* **Per launch at M = 16**, main -> new:

  | shape | main | new |
  |---|---:|---:|
  | 2048x4096 | 53.8 us | 41.1 us (0.76) |
  | 8192x2048 | 54.0 us | 48.3 us |
  | 4096x2048 | 38.6 us | 33.2 us |
  | 2048x2048 | 26.9 us | 24.8 us |
  | 512x2048 | 18.8 us | 16.4 us |
  | 256x2048 | 18.0 us | 15.6 us |

  At M = 8 the gain is 11-22 %.

## 1. Shapes per decode step

The batched step sends every dense fp8 projection of the 35B through `matmul_fp8_gemv_rows` (`kernels/hip/fp8.hip`, one launch per
projection since HFP8). All of them are W8A8: every fp8 weight of the checkpoint has an `input_scale` (`act_quant`,
`crates/intarsia-core/src/quant/fp8.rs:107`) and the kernel reads `sc[1] > 0` (the "W8A16-kind" of the folds in WMR2 is only the
registry attribute of the M = 1 fold kernels). Shapes and calls per step, from the model dimensions (they reproduce HPROF's 300 group
launches per 16-row step on the pre-HFP8 main: 40 x 2 + 30 x 2 + 40 x 3 + 20 x 2):

| N x K | projections | calls/step TP=1 | TP=2 shard | calls/step TP=2 |
|---|---|---:|---|---:|
| 8192x2048 | GDN `in_proj_qkv` (30), attention `q_proj` (10) | 40 | 4096x2048 | 40 |
| 4096x2048 | GDN `in_proj_z` (30) | 30 | 2048x2048 | 30 |
| 2048x4096 | GDN `out_proj` (30), attention `o_proj` (10) | 40 | 2048x2048 (row split) | 40 |
| 512x2048 | attention `k_proj`, `v_proj` (10 + 10) | 20 | 256x2048 | 20 |

So K = 4096 exists only at TP=1; TP=2 runs 4096x2048 (40), 2048x2048 (70) and 256x2048 (20).

Main at 16 rows against one row (microbench, `raw/mb1`, two rounds identical to 0.1 us): 2048x4096 3.3 x, 512x2048 3.6 x, 256x2048
3.9 x, 2048x2048 2.5 x, 4096x2048 2.3 x, 8192x2048 1.7 x.

## 2. Where the time went

Microbench: `scripts/k4fp8/bench_fp8.hip`. This is HFP8's harness: it includes `fp8.hip` and times `ik_matmul_fp8_gemv_rows` with the
weights rotated over >= 400 MB of copies, best of 5 reps, tatooine card 0. I added `BENCH_HOT=1`, but it turned out to be useless: the
weight loads are nontemporal, so a cache-resident copy times the same as DRAM. Raw: `raw/mb1` .. `raw/mb6`, each with its `steps.log`.

**Ablations at 2048x4096, M = 16** (`raw/mb1`). The binaries come from one ablation copy of `fp8.hip` (`raw/prototypes/`). That copy's
unmodified build reads 58.6 us and main's reads 53.3 us in the same window, so single cells of different binaries move by up to 10 %.

| variant | M=1 | M=8 | M=16 |
|---|---:|---:|---:|
| unmodified (ablation copy) | 18.0 | 33.5 | 58.6 |
| no division in the staging | 17.9 | 27.8 | 51.4 |
| no quantisation in the staging (bf16 bits stored) | 17.9 | 26.3 | 49.2 |
| no staging at all | 17.9 | 21.8 | 32.5 |
| no LDS reads of x (constants) | 17.9 | 31.5 | 53.8 |
| 16 warps per block (bitwise, 72 checks) | 17.9 | 28.5 | 45.1 |

The "no compute" and "no epilogue" ablations are not usable: the compiler dropped the weight loads or most of the chains. They stay in
`raw/mb1` for the record.

At K = 4096 there is no tile loop, and the activations are staged per block and per segment. So each block of 16 weight rows stages
and quantises all 16 x 4096 activations, which is about half the M = 16 time. Two prototypes attacked that directly (`raw/mb2`):

* **e4m3 bytes in LDS.** 16 x 4096 then fits one 64 KiB stage, so the tile loop serves K = 4096 too. Result: 2048x4096 M=16 went from
  54.5 to 46 us and M=12 from 45 to 35. But 8192x2048 M=16 went from 54 to 70 us, because the activations need a conversion per use
  (M x 8 per chunk). A higher-occupancy version (`CAP=2048`, 32 KiB blocks) behaved the same. Not kept.
* **16 warps per block at K > 2048.** The staging then serves 32 weight rows. Result: 53.3 to 45.1 us, with no extra ALU. Kept.

The bytes prototype added ALU per row and made every large-N shape much slower. That showed the 16-row kernel does not overlap its ALU
work with its weight stream. The ISA (`raw/isa/main-rows16-bf16.s.gz`, `fp8_gemv_rows_kernel<16, 2, bf16>`) shows why:

1. **The compute waits for the next tile's weights.**
   - Each weight load was `(p[r] && u < nu) ? ik_ldcs(p[r] + u) : 0`, in its own `execz` branch, and the next tile's prefetch was
     conditional.
   - Across that control flow the waitcnt pass cannot count the loads. It emits `s_wait_loadcnt 0x0` at the top of the compute, after
     the prefetch was issued.
   - So every tile waits for the next tile's weights before it computes. Loads and dot2 work run one after the other, and the prefetch
     buys nothing.
   - Above M = 8 at 8192x2048, time grows ~1.8 us per row (the dot2 work) on top of the 32 us weight stream.
2. **LDS latency is exposed per row.** Each m's two `ds_load_b128` were issued right before use and followed by `s_wait_dscnt 0x0`
   (199 per kernel). Only the two weight rows' chains interleaved, so an `s_delay_alu VALU_DEP_2` follows every dot2 pair (1709 in the
   kernel).

## 3. What changed

Only `fp8_gemv_rows_kernel` and its launcher in `kernels/hip/fp8.hip` changed.

1. **Loads the compiler can count.**
   - Every lane always loads from a valid address. A row past N or a missing next tile reads chunk 0 of row 0; a chunk past K reads chunk
     0 of its own row.
   - The value is used as it is, so nothing needs zeroing. A row past N has its own chains and is never stored; a chunk past K is skipped
     by the compute loop. (A select right after the load forces the wait too: `raw/isa/selectfree-*`.)
   - The next tile's prefetch is issued on every path. With no next tile it reads row 0's chunk 0, an L2 hit.
   - The compute now waits with `s_wait_loadcnt 0x8`, so the 8 prefetch loads stay in flight.
2. **Grouped rows.** The W8A8 inner loop takes the activation rows in groups of 4. The next group's two LDS chunks are read while this
   group computes, and the q loop (8 dot2 per chunk) runs outside the rows, so 8 chains interleave instead of 2. Each `acc[r][m]` still
   sees its dot2s in the same order: chunk by chunk, q = 0..7.
3. **Weight conversion by the fp8 conversion instruction.**
   - `v_cvt_pk_f32_fp8` + `v_cvt_pk_rtz_f16_f32` take 2 instructions per fp16 pair, instead of 4 for the perm / shift / and / and-or
     bit trick.
   - The result is the exact fp16 value of the e4m3 weight, which is 2^8 times the old `e4m3 * 2^-8` encoding. So every product, chain
     sum and lane sum is exactly 2^8 times its old value. A power of two changes no rounding, and the values stay far from f32 underflow
     and overflow (the smallest nonzero product is 2^-26). The output scale becomes 2^8 instead of 2^16.
   - The bits are therefore the same. The bitwise checks confirm it, and their random weights include e4m3 subnormals.
   - NaN codes (0x7f / 0xff) now become NaN instead of a finite value. Checkpoints have none, and `fp8.hip` did not preserve them before
     either.
4. **16 warps per block at K > 2048** (`kWarpsLongK`). There is no tile loop there, so the staging now serves 32 weight rows instead of
   16. This gives 4 waves per SIMD at 234 VGPR, with no scratch.

Unchanged:

- The dispatch (`INTARSIA_HIP_FP8_ONEPASS_MIN`: M >= 3 at K <= 2048, M >= 5 above).
- The M = 1, 2 kernels. So the solo step and single-stream decode run exactly main's code.
- The W8A16 inner loop gets only items 1 and 4. This model does not use it.

VGPR at M = 16 goes from 214 to 234, with no scratch (`-Rpass-analysis`).

Before / after per launch (`raw/mb6`), in us. Each cell is the mean of two interleaved rounds. "main" is `bench-main`, built from main's
`fp8.hip`; "new" is the final kernel.

| N x K | calls/step TP=1 / TP=2 | | M=1 | M=3 | M=4 | M=5 | M=8 | M=12 | M=16 | M=16 new / main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192x2048 | 40 / 0 | main | 32.5 | 33.5 | 34.4 | 35.1 | 40.0 | 46.9 | 54.0 | |
| | | new | 32.4 | 32.2 | 32.7 | 33.2 | 35.7 | 43.8 | 48.3 | 0.90 |
| 4096x2048 | 30 / 40 | main | 17.0 | 19.3 | 20.3 | 20.6 | 25.2 | 31.4 | 38.6 | |
| | | new | 17.0 | 18.0 | 18.5 | 19.0 | 21.5 | 28.0 | 33.2 | 0.86 |
| 2048x4096 | 40 / 0 | main | 16.3 | 19.6 | 20.6 | 24.1 | 31.2 | 45.1 | 53.8 | |
| | | new | 16.3 | 19.4 | 20.7 | 22.3 | 27.8 | 36.2 | 41.1 | 0.76 |
| 512x2048 | 20 / 0 | main | 5.2 | 6.5 | 6.9 | 8.2 | 11.1 | 14.7 | 18.8 | |
| | | new | 5.2 | 6.0 | 6.6 | 6.9 | 8.7 | 14.2 | 16.4 | 0.87 |
| 2048x2048 | 0 / 70 | main | 10.8 | 12.7 | 13.8 | 14.2 | 17.4 | 22.5 | 26.9 | |
| | | new | 10.8 | 11.4 | 11.9 | 12.4 | 15.2 | 21.7 | 24.8 | 0.92 |
| 256x2048 | 0 / 20 | main | 4.6 | 5.6 | 6.1 | 6.6 | 10.2 | 13.9 | 18.0 | |
| | | new | 4.6 | 5.5 | 5.8 | 6.1 | 8.5 | 13.2 | 15.6 | 0.86 |

The levers separately, at M = 16 (`raw/mb3` .. `raw/mb6`), as a fraction of main:

| levers applied | M = 16 vs main |
|---|---|
| grouped rows alone | 0.91-0.94 |
| + branch-free loads and the unconditional prefetch (`up4`) | 0.90-0.97; the gain is mostly at M = 3..12 (M = 8: 0.77-0.89, except 2048x4096 at 1.00) |
| + the hardware conversion (`n1-w8`) | 0.85-0.94 |
| + 16 warps at long K (final) | 2048x4096 goes from 0.94 to 0.76 |

Weighted by the calls per step of section 1:

| rows | TP | fp8 projections main | new | saved |
|---:|---|---:|---:|---:|
| 8 | TP=1 | 3.82 ms/step | 3.36 ms/step | 0.47 ms |
| 8 | TP=2 | 2.43 ms/step | 2.09 ms/step | 0.33 ms |
| 16 | TP=1 | 5.85 ms/step | 4.90 ms/step | 0.95 ms |
| 16 | TP=2 | 3.79 ms/step | 3.38 ms/step | 0.41 ms |

Measured in the model (batched step of the bitwise test: 16 rows, in-process, graphs on, per step; main = tree `hfp8new2`, job #7407,
`raw/steps-main`; new = jobs #7405 / #7406, `raw/gates-a`, `raw/gates-b`):

| rows | TP=1 main | TP=1 new | TP=2 main | TP=2 new |
|---:|---:|---:|---:|---:|
| 4 | 12.6 ms | 12.4 ms | 10.4 ms | 10.2 ms |
| 8 | 18.6 ms | 18.1 ms | 14.4 ms | 13.9 ms |
| 16 | 32.3 ms | 31.4 ms (-0.9 ms, -2.8 %) | 23.7 ms | 23.1 ms (-0.6 ms, -2.5 %) |

The solo steps of the same test are unchanged (16 solo steps: 100.2 vs 100.1 ms TP=1, 86.7 vs 86.7 ms TP=2).

## 4. Correctness

* **Standalone bitwise check** (`raw/mb2`, `raw/mb3`, `raw/mb4`, `raw/mb6`, files `chk-*` / `w16chk*`).
  - Check: row m of the M-row call equals `ik_matmul_fp8_gemv_v2` on row m alone, bit for bit, with bf16 and f32 out.
  - Coverage: M = 1..16 with `INTARSIA_HIP_FP8_ONEPASS_MIN=1`, so the new kernel serves every M. Both W8A8 (default tiles and tiles = 3)
    and W8A16.
  - Shapes, 13 in the final run: 2048x4096; 33x2064 (ragged chunk tail, two segments); 16x16400 (9 segments); 64x6144; 1x4096; 8x1040;
    9x16; 300x2048 (rows past N in the last tile); 2048x2048; 8192x2048; 2047x4112 (ragged N and K, three segments); 4096x2048; 512x2048.
  - Result on the final kernel: 3 x 416 = 1248 checks, 0 mismatches. Every intermediate variant was checked the same way: 2692 more
    checks, 0 mismatches.
* **Repo tests** (tree `k4fp8`, job #7405, `raw/gates-a`): `rows_exact` 13 passed, `fp8_f8` 5, `rows_registry` 2, `frag_layout` 5,
  `quick_coverage` 1. The panics in their logs are the `should_panic` cases.
* **`docs_env_vars`** (local, CPU): 2 passed. I updated the `docs/tuning.md` line references of `INTARSIA_HIP_FP8_ONEPASS_MIN` and
  `_TILES`, and the TILES row now notes the 16 warps at long K.
* **Real 35B, batched == solo bit for bit** (`real_35b_batched_decode_is_bitwise_the_solo_step`: N = 2..16, 130 steps each, f32 logits
  and argmax, 17 550 (row, step) pairs): passed at TP=1 (#7405, `raw/gates-a`) and TP=2 (#7406, `raw/gates-b`).
* **Served greedy identity** (`scripts/k4fp8/greedy.py`, job #7410, `raw/w-greedy`): 16 concurrent different ~1.1k-token prompts,
  temperature 0, 192 tokens each (3072 tokens, 16 distinct texts). Main (tree `hfp8new2`) and the branch give byte-identical JSONL at
  TP=1 and at TP=2 (`cmp`).
* **KL / top-1 gate: not run.** No reduction order changed: items 1-4 reorder independent chains and loads, and item 3 is an exact
  power-of-two rescaling. Everything above compares bits with main.

## 5. End to end (Qwen3.6-35B-A3B NVFP4, `serve --max-batch 16 --spec off`, 1k prompts, gen 512, `localeval speed`)

Before = tree `hfp8new2`: crates and kernels identical to main, a real build from lane HFP8. After = tree `k4fp8` (this branch).

- Each round starts a fresh server per arm. Arms are interleaved, and the order is reversed on even rounds.
- One fleetq job per TP holds both R9700 cards with Radiance down (`--takes-down radiance --quiet gpu,cpu`).
- Radiance answered again 15 s after every window (`downtime.log` in each raw dir).
- tok/s are aggregate decode rates; for 1 stream it is that stream's decode rate.
- The localeval result files are copied to `raw/w-*/results/`.

TP=1 (job #7408, `raw/w-tp1`):

| streams | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 163.9 | 164.4 | 164.5 | 164.6 | 164.3 | 164.6 | 164.4 | 164.6 | 1.001 |
| 4 | 304.0 | 299.8 | 299.0 | 302.0 | 299.8 | 304.7 | 299.8 | 302.0 | 1.007 |
| 8 | 394.6 | 396.5 | 396.6 | 406.8 | 406.4 | 406.0 | 396.5 | 406.4 | 1.025 |
| 16 | 460.3 | 463.1 | 462.9 | 477.6 | 475.6 | 475.2 | 462.9 | 475.6 | 1.027 |

TP=2 (job #7409, `raw/w-tp2`):

| streams | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192.5 | 192.7 | 192.4 | 192.7 | 192.9 | 192.7 | 192.5 | 192.7 | 1.001 |
| 4 | 369.6 | 370.7 | 371.0 | 374.8 | 375.5 | 373.6 | 370.7 | 374.8 | 1.011 |
| 8 | 526.8 | 528.3 | 528.6 | 542.8 | 544.3 | 541.7 | 528.3 | 542.8 | 1.027 |
| 16 | 651.6 | 651.3 | 648.6 | 660.9 | 660.4 | 660.6 | 651.3 | 660.6 | 1.014 |

At 8 and 16 streams every round of the branch is above every round of main, at both TP sizes.

Single-stream decode at 1k with `--max-batch 1` (spec off; job #7411, `raw/w-b1`, TP=1, 3 rounds):

| streams | base r1 | base r2 | base r3 | new r1 | new r2 | new r3 | base median | new median | new / base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 164.6 | 162.0 | 164.2 | 164.5 | 162.5 | 164.4 | 164.2 | 164.4 | 1.001 |

There is no regression: M = 1, 2 never reach the changed kernel. The 1-stream rows of the two tables above come from `--max-batch 16`
servers.

The TP=2 gain is smaller because TP=2 has no K = 4096 shape: the row-split `out_proj` / `o_proj` shards are 2048x2048. The largest
single lever, 16 warps at long K, therefore does not apply there. The microbench predicted -0.41 ms/step at TP=2 and -0.95 ms at TP=1;
the model's step measured -0.6 and -0.9 ms.

## 6. What is left

* The 16-row call still costs 1.5 x (8192x2048) to 2.5 x (2048x4096) a one-row call at the large shapes, and 3.2-3.4 x at N = 512 / 256. The remaining time is ALU and LDS:
  - 1024 `v_dot2` per tile and wave at M = 16. That count is fixed by the bit-exact chains.
  - 128 LDS loads per tile and wave.
  - The weight conversion.
  At M = 16 this work is of the same order as the weight stream.
* Overlapping it further means deeper prefetch (two tiles) or fewer VGPRs per wave (R = 4 rows per warp halves the LDS reads per dot2).
  Both are untested here.
* The e4m3-byte staging prototype (`raw/prototypes/`) is the other lever for K = 4096 at higher M. It needs a cheaper per-use
  conversion to stop losing at large N.
* Small N (512, 256: 16-18 us at M = 16 against 5 us at M = 1) is latency bound with 16-32 blocks. Fusing the k / v pair into one launch
  (HFP8's note) is still open.
* The same load idiom (`cond ? ik_ldcs(...) : 0`, which compiles to loads in branches) is in the one-row `fp8_gemv_body`
  (`fp8.hip:164`, the M = 1, 2 kernel, deliberately untouched here) and in the MoE GEMV kernels (`moe.hip:604`, `:724`, `:926`, `:992`,
  ...). Each needs its ISA read before anyone assumes the same stall applies.
* `kOnePassMinLong = 5` (the fewest rows the one-pass kernel takes at K > 2048) was set against the old one-pass kernel. In `raw/mb6`,
  M = 3, 4 at K = 4096 still ran the per-group kernel in both arms (default dispatch), so whether the new kernel should take them was
  not timed.

## Reproduce

```
scripts/k4fp8/bench.sh <tree> <tag> [-DFLAGS]   # compile bench_fp8.hip (CPU job; INC=<dir> puts a modified fp8.hip first)
RUNS="main@bench-main:BENCH_SHAPES=8192x2048,2048x4096;BENCH_MS=1,8,16 new@bench-new:..." OUT=<dir> \
  fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu ... -- scripts/k4fp8/mbwin.sh
STEPS="hip tp1 tp2" [TREE=hfp8new2] OUT=<dir> fleetq run ... -- scripts/k4fp8/tests.sh   # rows tests + batched == solo (step times)
ARMS="tp1:base tp1:new" ROUNDS=3 OUT=<dir> fleetq run ... -- scripts/k4fp8/window.sh     # MODE=greedy, MB=1 CONC= as in section 5
python3 scripts/k4fp8/table.py <window dir>
```

Trees on tatooine:

- `hfp8new2` is the main arm. Its crates and kernels are identical to main 5f7ec5996 / 78de87bf7 (from lane HFP8).
- `k4fp8` is this branch, built by `scripts/tp2-hip/build.sh` with `HIPTEST=1 HIPTEST_MODELS=1` (job #7404).

Fleetq jobs (all `claude-908e11e8`):

| purpose | jobs |
|---|---|
| microbench windows | #7388 (`mb1`), #7390 (`mb2`), #7393 (`mb3`), #7397 (`mb4`), #7399 (`mb5`), #7403 (`mb6`) |
| bench / asm compiles (CPU) | #7387, #7389, #7391, #7392, #7394, #7395, #7396, #7398, #7401 (failed: non-constant builtin argument), #7402 |
| tree build | #7404 |
| gates | #7405, #7406 |
| main step time | #7407 |
| A/B | #7408 (TP=1), #7409 (TP=2) |
| greedy | #7410 |
| single stream | #7411 |

The branch-free and select-free intermediate builds (#7394, #7395) were only read as ISA (`raw/isa/`). They were not timed.
