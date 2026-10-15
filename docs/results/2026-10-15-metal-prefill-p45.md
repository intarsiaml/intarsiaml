# Metal prefill P4/P5: NAX GEMM tile pick, prefill qk_norm_rope_kv, and the fusions that did not pay (2026-10-15)

Branch `metal-p45-gemm` (from spike/metal 766c723, spike/metal a710f35 merged in), Qwen3-8B BF16, endor (M5 Max), `--backend metal`. Raw logs and scripts: `docs/results/raw/2026-10-15-metal-prefill-p45/`.

## Outcome

Merge-ready: **yes** (all gates below pass; every output is bitwise identical to the 766c723 binary).

Two changes kept, two levers measured and dropped by bound or by measurement, one tried twice and dropped:

| lever | verdict | what it does |
|---|---|---|
| (a) per-shape / per-M NAX tile pick | **kept** | three new tiles, picked for 128 <= M <= 512 (the 256-row default chunk and its tail); M outside that range keeps the P1 rule. Chunk 256: q/o 54 -> 58 TFLOPS, down 52 -> 57, M=222 tail +19 to +27% on q/o and down |
| `qk_norm_rope_kv_bf16_pf` (not on the lever list, found in the trace) | **kept** | prefill form of the B4 op: 85 us/call -> 16 us at chunk 256, 564 -> 88 us at chunk 2048. Bitwise the decode form, no core change |
| (b) q+k+v triple GEMM | **not built**: bound 0.5% | single 6144-column proxy kernel measured; see below |
| (c) SwiGLU epilogue | **dropped**: slower, and wrong bits | prototype measured +68 us/call at M=256 over the pair kernel; the separate `silu_glu` it would replace is 23 us/call |
| (c) residual add into o/down epilogue | **not built**: bound 0.7% | all `rmsnorm_add` dispatches are 10 ms of a 1.6 s prefill |
| (d) lm_head at T>1 | **already done by the engine** | one `logits_bf16_gemv` M=1 call per prefill (2.2 ms, 0.15%) |

Speed, plain prefill (`generate --max-tokens 1`, ABBA, pre = 766c723 binary, median of 6 runs for 1k/4k, 4 runs for 16k):

| prompt | chunk 256: pre -> new (ms) | chunk 256 B/A | chunk 2048: pre -> new (ms) | chunk 2048 B/A |
|---|---|---|---|---|
| 1k | 503 -> 488 | 0.971 | 417 -> 400 | 0.959 |
| 4k | 1842 -> 1719 | **0.933** | 1515 -> 1545 | 1.020 (mean 0.990) |
| 16k | 12150 -> 11667 | 0.960 | 10943 -> 10910 | 0.997 |

localeval speed (`local-model-bench`, prefill tok/s, 3 reps per server, pre new new pre, medians of the reps; chunk-256 pre uses the second pre server, see the caveat in the method notes; decode unchanged 36.6 tok/s at 128 and 1k, 35.5 at 4k):

| prompt | chunk 256 pre -> new | | chunk 2048 pre -> new | |
|---|---|---|---|---|
| 1k | 3232 -> 3485 | +7.8% | 3575 -> 3680 | +2.9% |
| 4k | 2690 -> 2836 | +5.4% | 3055 -> 3130 | +2.5% |

So: **chunk 256 4k prefill is 5 to 7% faster, chunk 2048 is 1 to 3% faster (below the 3% bar, but it comes for free from the same change set and is mostly the qk kernel)**. At chunk 2048 the matmuls already run at 58 to 65 TFLOPS (peak 64.6 on gate/up), so there is no tile-level headroom left there: the sum over all four shapes at M=2048 is at most about 3% of prefill even if every one reached peak. After the F1 attention branch and P3 chunk merge, expect this change to be worth the chunk-2048 number (~2 to 3%), not the chunk-256 one.

Decode is untouched: 364 dispatches/step both binaries; step 27.29 -> 27.34 ms (short), 28.41 -> 28.09 ms (4k); the `pick_variant` change only reaches M >= 16 and the qk_pf kernel only T > 1.

## Per-shape TFLOPS (Qwen3-8B shapes)

Microbench `examples/gemm_nax pick2`: old = P1 rule, new = shipped `pick_variant`, kernels as the engine launches them (single for q/o and down, pair for k+v and gate+up), weights from a rotating 3 GiB pool (DRAM-streamed), 15 interleaved rounds x 10 reps, median. Raw: `micro/pick2.txt`. Run-to-run drift between bursts is up to 8% in absolute TFLOPS (q/o M=256 read 57.8 in one burst and 52.7 in the next, `qkv/pick2-again.txt`); only the old/new pair inside one run is comparable, and the gains in the table repeat in both runs.

| shape (N x K) | M | old tile | old TFLOPS | new tile | new TFLOPS | change |
|---|---|---|---|---|---|---|
| q/o 4096x4096 | 256 | 64x64w4 | 54.0 | 64x64w2 | 57.8 | +7.1% |
| q/o | 222 | 64x64w4 | 47.1 | 32x128w2 | 56.2 | +19.4% |
| q/o | 205 | 64x64w4 | 43.5 | 32x128w2 | 52.0 | +19.5% |
| k+v pair 2048x4096 | 256 | 64x64w4 | 50.7 | 32x32w1 | 53.3 | +5.2% |
| k+v pair | 222 | 64x64w4 | 43.0 | 32x32w1 | 44.0 | +2.3% |
| gate+up pair 24576x4096 | 256 | 128x64w4 | 62.1 | 64x64w2 | 64.0 | +3.2% |
| gate+up pair | 222 | 128x64w4 | 53.8 | 64x64w2 | 55.6 | +3.3% |
| down 4096x12288 | 256 | 64x64w4 | 51.5 | 64x64w2 | 57.3 | +11.1% |
| down | 222 | 64x64w4 | 44.3 | 32x128w2 | 55.7 | +25.8% |
| down | 205 | 64x64w4 | 40.7 | 32x128w2 | 51.7 | +26.9% |
| every shape | 973, 1024, 2014, 2048 | P1 rule | unchanged | same | unchanged | within +-0.4% (down 2048: -1.3%, noise) |

At M=2048 (P1 rule, unchanged): gate+up 65.0, k+v 60.6, q/o 64.1, down 56.6 in the microbench; 64.3 / 54.3 / 58.0 / 61.8 in the engine kernel trace (kernel-mode carries about 30 us/dispatch of per-command-buffer overhead, so the engine numbers sit below the microbench on the small shapes).

Which shapes were below ~60 TFLOPS and why. At chunk 256 every shape except gate+up (62 to 64) was: q/o 54, down 51.5, k+v 50.7. The tile sweep (21 tiles, `sweep/`) says the cause is the tile, not the kernel: at M=256 the tiles that win are the ones with **32x64 output per simdgroup** (64x64 on two simdgroups, 32x128 on two, 32x64 on one) or the finest grid (32x32 on one simdgroup for the narrow k+v). The P1 pick (64x64 with four simdgroups, i.e. 32x32 per simdgroup) reuses each loaded A fragment for half as many MMAs, and the 128x64 tile leaves only 128 to 192 threadgroups on 40 cores (3 to 5 waves with a ragged last one). Hypothesis, supported only by the ranking (no hardware counters). The 4096-column shapes at an M tail (M=205/222: `m % 64` in 1..32) additionally waste up to half of every 64-row tile; the 32-row tile removes that, which is where the +19 to +27% come from.

Tile rule shipped (`gemm_nax::pick_variant`, any choice gives the same bits):

| M | n (columns, pair = n0+n1) | tile |
|---|---|---|
| < 128 or > 512 | any | P1 rule: 128x64w4 if it leaves >= 192 threadgroups else 64x64w4 |
| 128..512 | m*n < 700,000 (k+v pair at chunk 256) | 32x32w1 |
| 128..512 | < 8192 and `m % 64` in 1..=32 | 32x128w2 |
| 128..512 | otherwise | 64x64w2 |

Model of the 4k chunk-256 matmul time from the microbench sweep (`sweep/rule_model.txt`, the shipped rule applied to the measured per-tile times): P1 rule 1045 ms, shipped rule 991 ms (-5.1%), per-case oracle 979 ms. Measured in-engine GPU busy (cb mode): -82 ms of 1497 (-5.5%) at chunk 256.

## Prefill kernels, 4k prompt

`INTARSIA_METAL_PROFILE=kernel:` traces, one command buffer per dispatch (`trace-t2/`). Caveats: kernel mode adds ~30 us per dispatch, and the script ran `pre` before `new` in each configuration so any GPU clock ramp favours `new`: the attention kernel, which this branch does not touch, reads 772 us (pre) vs 699 us (new) per call at chunk 256. Use these tables for the *shape* of the time, and the cb-mode and ABBA rows for gains.

| class (kernel mode) | chunk 256 pre | chunk 256 new | chunk 2048 pre | chunk 2048 new |
|---|---|---|---|---|
| matmul | 1169.8 ms (69.3%) | 1002.7 (69.7%) | 910.7 (68.1%) | 908.4 (70.0%) |
| attention | 444.5 | 402.5 | 352.2 | 351.2 |
| elementwise/row | 74.6 | 32.9 | 73.5 | 38.4 |
| GPU busy, cb mode (interleaved, no per-dispatch CB) | 1497.3 ms | 1415.4 (-5.5%) | 1330.2 | 1295.5 (-2.6%) |

The kernel-mode matmul gain at chunk 256 (-14%) is overstated by the order bias and CB overhead; the cb-mode busy time, which has none of the two, is the number to trust (-82 ms total, matmul + elementwise).

Elementwise after the change at chunk 256: 32.9 ms = `qk_norm_rope_kv_pf` 9 ms (576 x 16 us), `silu_glu` 13 ms (576 x 23 us), `rmsnorm_add` 10 ms (1152 x 9 us). This is the whole pool the P5 fusions could address.

## Levers in detail

**(a) Tile and threadgroup tuning: kept.** Attempt 1 (21 variants, microbench on all four shapes at M in {32..2048}, then an in-engine run per tile with `INTARSIA_NAX_TILE`): the rule above, -5.4% at chunk 256. Attempt 2 (extend the 128x64 -> 64x64w4 swap to 512 < M <= 1536 for 4096-column outputs, `plain-t3-*`, `trace-t3/`): trace 99.6% to 103% of the P1 times, within noise, reverted. A rule variant with the 256x64w8 tile for M >= 1024 and wide outputs (microbench prediction -2.5% of matmul time at chunk 2048) was *slower* in the engine (gate+up M=2048: 6689 vs 6371 us for 128x64w4, down 3725 vs 3366, `engine-sweep/summary-us-per-call.txt`) and was not shipped; the 3 GiB-pool microbench and the engine disagree on that tile. Check of the rule's upper end at chunks 384 / 512 / 128 (`engine-sweep2/`): 64x64w2 or 64x64w4 are within noise of each other at M=384..512; at M=94 (a chunk-128 tail) the 32x32w1 tile is 2x faster than 64x64w4 on the narrow cases, left alone (tails of 16..127 rows cost under 1% of a prompt).

**qk_norm_rope_kv prefill form: kept.** `qk_norm_rope_kv_bf16` spent most of its time in `precise::pow / sin / cos`, evaluated by all 128 threads of every (token, head) although the angle only depends on (position, pair): 4096 evaluations per token for 64 distinct ones. `qk_norm_rope_kv_bf16_pf` (T > 1, head_dim = rotary_dim = 128) builds the 64 (cos, sin) pairs once per threadgroup with the same expressions and runs one head per simdgroup (12 heads per threadgroup, four simd_sums replace the threadgroup reduction, no barrier per head). Same registry entry, chosen by the launcher; bitwise the decode form (`qkv_fold` tests). Kill switch for A/B: `INTARSIA_METAL_NO_QK_PF=1`. Isolated ABBA (`plain-t2-qk-c*`, 4k): chunk 2048 -2.7%, chunk 256 -1.3%.

**(b) q+k+v triple GEMM: not built.** Proxy: one 6144-column kernel (`gemm_nax bench qkv`, `qkv/bench-qkv.txt`) reaches 56.7 TFLOPS at M=256 (128x64w4) against ~53 for q/o + k+v issued separately. q + k/v pair are 252 us per layer-chunk at chunk 256: the best case saves 14 us x 540 = 7.7 ms, **0.5% of a 4k prefill**, and needs a new core op, CPU reference and model wiring. At chunk 2048 q/o runs 58 and k+v 54 TFLOPS vs 65 peak: <= 1.5 ms per 4k prompt. Below the 3% bar by a factor of six; stopped without building.

**(c) SwiGLU epilogue: dropped.** Prototype `gemm_nax_glu_t*` (both products into bf16 cooperative destinations, `silu(g)*u` in registers, one store; `glu/swiglu-epilogue-prototype.diff`). `glu/bench-glu.txt`, against the shipped pair tile (64x64w2) plus the `silu_glu` dispatch it replaces: M=256 glu 882 us vs 814 + 23 = 837 us, M=222 1038 vs 964 + ~21, M=2048 9059 vs 8131 + 255 = 8386 us (+8%); it loses at every M measured. (Against the pair kernel alone, the 64x64w4 glu tile at M=256 is 17 us faster, 895 vs 913 us, but that pair tile is not the shipped one and the saving is below the 23 us dispatch.) The epilogue also came out wrong (element 0: 0x33bd vs 0x3d35; not diagnosed, most likely the two cooperative destinations do not share an element mapping, or the in-register read-after-`run` ordering) so it failed the bitwise test against pair + `silu_glu`. One attempt, plus the bound (`silu_glu` is 13 ms of 1.6 s, 0.9%, even if its fusion were free), is enough to stop. **Residual-add into the o/down epilogue** has the same bound: all `rmsnorm_add` dispatches are 10 ms (0.7%), the add being fused into the rmsnorm already (B-phase). Not built.

**(d) lm_head at T>1: already last-row only.** The prefill trace shows exactly one `logits_bf16_gemv` M=1 (151936 x 4096) call per prompt, 2.2 ms (0.15% of 4k prefill), at 0.58 TFLOPS = 577 GB/s, i.e. bandwidth bound on one row. The model's `LastLogits` / `LastArgmax` head modes skip the other rows; nothing to do.

## Gates

| gate | result |
|---|---|
| `intarsia kernels test --backend metal` | 22 kernels, 265 cases, all ADMITTED (`gates/kernels-test.txt`) |
| `cargo test -p intarsia-backend-metal --release` | 69 passed, 0 failed (`gates/cargo-test-metal.txt`, at the code commit); `qkv_fold` re-run after adding the new test: 6 passed (`gates/test-qkv-fold.txt`, #1954) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | exit 0 (`gates/cargo-test-workspace.txt`) |
| `cargo check --workspace` | exit 0 (`gates/cargo-check-workspace.txt`) |
| `intarsia accuracy` prefill KL / top-1, short | 2.686e-4, 99.77% (= pre) |
| prefill, long | 4.342e-4, 99.65% = 1429/1434 (= pre) |
| decode KL, short / long golden | 3.739e-4 / 3.731e-4, top-1 100% (identical to the 766c723 binary). The brief's 4.042e-4 / 4.071e-4 are the pre-P1 values (`2026-10-14-metal-prefill-p1.md` table: decode 4.042e-4 -> 3.737e-4 and 4.071e-4 -> 3.708e-4 when the NAX GEMM landed); the current baseline is the 3.7e-4 pair, and this branch moves nothing |
| prefill accuracy at chunks 128 / 512 / 2048 (short) | all 2.686e-4, 99.77% (chunk invariant) |
| greedy ids, short and 4k prompts, 48 and 256 tokens, vs 766c723 | **identical** in all four (no token-49 near-tie difference either) |
| chunk invariance, 4k prompt, ids at chunks 128/256/512/1024/2048 | identical across chunks for both binaries, and identical between binaries at each chunk |
| `intarsia qualify --backend metal` | **10 PASS, 2 SKIP** (ab, tp), fresh run, `gates/qualify/full-report.md` |
| decode dispatches/step | 364 (both), `gates/decode/` |
| guardrail (KL > 5e-4, long top-1 < 99.6%, id diffs) | none triggered |

New tests: `tests/gemm_nax.rs::nax_tile_pick_is_by_shape` (the rule, including 205/222/973/2014 tails and the P1 fallbacks), `nax_launch_picks_tiles_without_changing_bits` (extended: M=222 and 256 at N=4096, real shapes, bitwise across every tile, 64x64w2 and 32x128w2 included), all NAX tests run against the f64 reference with M tails, K tails, row independence, odd offsets and guard bytes over every table entry; `tests/qkv_fold.rs::prefill_form_head_counts_off_the_threadgroup_group` (head totals 3, 7, 8, 15, 18, 24 around the 12-head group, positions up to 30,000 at theta 1e6, `w_offset` 0 and 1, device position and cache end inside the token block), plus the existing `prefill_rows_are_bitwise` (T 5..300) and `operands_at_two_byte_offsets`, which already hit the pf kernel; all bitwise against the unfused sequence.

## Core files changed

**None.** `git diff --stat 766c723..HEAD -- crates/intarsia-core` is empty. Changed files:
`kernels/metal/gemm_nax_bf16.metal`, `kernels/metal/qk_rope_kv.metal`, `crates/intarsia-backend-metal/src/{gemm_nax.rs,dense.rs,kernels.rs}`, `crates/intarsia-backend-metal/tests/{gemm_nax.rs,qkv_fold.rs}`, `crates/intarsia-backend-metal/examples/gemm_nax.rs`. The CUDA/HIP no-op check has nothing to cover in `intarsia-core`. Not touched: attention kernels, `flash.rs`, `attn.rs`, `gemv_bf16.metal`.

New env knobs (measurement only): `INTARSIA_NAX_TILE=<shader name or suffix>` forces one NAX tile for every call; `INTARSIA_METAL_NO_QK_PF=1` selects the decode form of the qk kernel for prefill rows.

## Method notes and caveats

- fleetq slots on endor, all owner claude-31e10c55 and all released: #1933, #1934, #1935 (tile microbench sweeps), #1938 and #1942 (NAX and qkv_fold tests), #1940 and #1943 (kernel traces; #1939 and #1944 were released unused after the client went to background / a script path typo), #1941 (in-engine tile sweep), #1945 (first plain ABBA), #1946 (SwiGLU prototype), #1947 (cb traces, qk isolation), #1948 (the 512 < M <= 1536 rule), #1949 (correctness gates), #1950 (final speed burst and localeval), #1951 (cargo test, kernels test, qualify), #1952 (chunk 512/384/128 tile check), #1953 (qkv_fold, failed on a bad test config, fixed), #1954 (qkv_fold + qkv proxy), #1955 (localeval chunk 256 rerun). endor-kev and endor-tensorfold untouched.
- GPU idle sampled (3 `ioreg` samples) before each timed phase: the first sample is the previous process's tail (up to 98%), the next two read 0 to 2%; `pgrep -fl zoom.us` empty (`zoom.txt`).
- The first localeval pass at chunk 256 (`localeval-c256/`) overwrote its own run labels (script bug, fixed in `localeval-abba.sh`); the numbers in the table come from the rerun `localeval-c256b/` (pre new new pre). The first server of the pair, `pre-1`, ran 4k at 2497 / 1921 / 2102 tok/s (the scatter the brief warned about) while `pre-2`, and all three 4k reps of the first pass, ran 2681 to 2694; the table uses the `pre-2` server (conservative: the six-rep median over both servers would be 2589 and the gain +9.5%; over all nine pre reps 2684, +5.7%). The new servers are tight (2804 to 2852 over six reps, median 2836).
- 4k chunk 2048 ABBA (6 runs): median 1.020, mean 0.990; localeval +2.5%; cb-mode GPU busy -2.6%. The honest reading is a 1 to 3% gain, not distinguishable from zero in 6 plain runs.
- One burst of `plain-t2-broken-script` is kept in the raw dir for the record: its first version of `plain-abba.sh` did not split the chunk list and ran nothing useful.
- Everything here is the Qwen3-8B dense shape set (4096 hidden, 12288 FFN, GQA 32/8, head_dim 128). The qk_pf kernel only fires at head_dim = rotary_dim = 128; other models take the old kernel.

## Dead ends

1. 128x128w8, 64x128w4/w8, 128x64w8, 128x32w2, 64x32w1, 32x64w1 (best in one small-pool run, behind in the 3 GiB-pool one), 64x64w1, 64x256w4, 32x256w2, 16-row tiles: none beat the shipped picks by more than noise in the 3 GiB-pool sweep (`sweep/bench2-*.txt`, `bench3-*.txt`, `engine-sweep/`).
2. 256x64w8 for M >= 1024 and wide N (slower in the engine), and 64x64w4 for 4096-column outputs at 512 < M <= 1536 (no gain).
3. SwiGLU epilogue (slower and wrong), residual epilogue and q+k+v triple (bounds under 1%).
4. Nothing was tried on the `attn_prefill` side (other branches).

## What would still move prefill

Per the cb-mode trace, chunk 256 pays ~4% GPU idle between command buffers (47 gaps, mean 1.3 ms, the first two 27 to 29 ms) that does not exist at chunk 2048 (0.0%): that is the P3 chunk lever, not a GEMM one. With P3 and F1 merged, matmul is ~70% of a 4k prefill at 58 to 65 TFLOPS against a 64.6 peak, i.e. at most ~3% more from GEMM tuning.
