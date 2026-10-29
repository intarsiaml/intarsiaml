# qkv16: row-exact attention q + k + v triple, bitwise but flat; no measurable step gain

Lane qkv16, 2026-10-29, branch `lane-qkv16/triple-gemv` from main 867b97145, merged with main 7aebefb83 (#250, HIP only) before
every measurement below. spark1 (spark-4e77, GB10 sm_121), `nvidia/Qwen3.6-35B-A3B-NVFP4`. Target: the GDNX follow-up
(docs/results/2026-10-29-gdnx-decode-kernels.md section 2): the 10 full-attention layers' q (+ output gate, 8192 x 2048), k and
v (512 x 2048 each) fp8 projections of the batched step as one launch. **Status: draft PR** (the B = 16 step does not improve
beyond noise; see Gates). Everything is bitwise exact and default on from 12 rows; the A/B below says it buys nothing, so the
coordinator may prefer to close it.

## Change

| piece | what |
|---|---|
| `kernels/cuda/fp8.cu` | `fp8_gemv_rows_q_pair_kernel` generalised to `fp8_gemv_rows_q_multi_kernel<NSEG>` over a by-value `QSegs<NSEG>` array (`rows_q_seg` unchanged, the tile base accumulates per segment). The pair is its NSEG = 2 instance (same registry name `matmul_fp8_pair_gemv_rows`, same launch shape and conditions); the new `ik_matmul_fp8_triple_gemv_rows` is NSEG = 3 at shape 2x8x2 (what the 8192-row q alone gets at M >= 12) |
| `Op::MatmulFp8Triple` (7 inputs, 3 outputs, appended to `Op::ALL`) | a thin new op because op arity is fixed (`MatmulFp8Pair` is 5 in, 2 out) and the pair's registry entry must not change. CPU reference = three `MatmulFp8`s (`fused_ref.rs`), `quant::cases` sweep / label / plan case, `quant::ops` check and shape, HIP lists it `UNIMPLEMENTED`, Metal has no kernel (the model falls back) |
| `matmul_fp8_triple_gemv_rows` (CUDA registry) | `rows_exact`, priority -1, bf16 out, registered only for `INTARSIA_CUDA_FP8_TRIPLE_ROWS` (default 12) .. 16 rows. Below 12 the single q takes the persistent bf16-staged kernel and the model's side stream overlaps k + v with it (moe16b), so the model keeps q beside the pair there. `0` = never registered |
| `Model::attn_project` / `qkv_triple` | tried before the side-stream fork, only in a row-exact multi-row call (`verifying()`, t >= 2: a batched or verify step), all three weights fp8; on success no fork, no q projection, no k / v pair; the gate split and the biases are unchanged |
| `crate::fusion::FUSIONS` | CUDA entry `MatmulFp8Triple` = `[MatmulFp8; 3]`, test `rows_exact.rs::fp8_triple_rows_are_bitwise_the_one_row_gemvs`. Without it `check_batch_ops` fails the 35B batched plan load (solo step = q + pair, batch = triple) |

`plan show` 8B, 35B and Gemma-4 (json and text) are byte-identical to main (`raw/plan.txt`): `plan show` (run without batching) records
the solo step, which never takes the triple, so there is no 35B diff to explain. The batched plan at B = 16 shows the triple
(batch_cuda_real loads it; `check_batch_ops` accepts it, unit test `batch_check_accepts_the_qkv_triple_for_q_and_the_kv_pair`).

## Exactness (all pass)

- `rows_exact` 11 passed: the new `fp8_triple_rows_are_bitwise_the_one_row_gemvs` runs M = 1..16, shapes 8192 + 512 + 512, 4096 + 300 + 9,
  33 + 70 + 17 (K 2064), 9 + 8 + 7 (K 16), nine input-scale sets (all equal; each matrix different from the other two, so the x restage
  between segments is exercised; 3.0 saturating; W8A16 in each position and all three), against the one-row `matmul_fp8_gemv` per row, at the launcher's bar (12), bar 2 (one launch at
  every M, `ROWS_Q` default and `all`) and bar 0. `row_tables` 6, `fused_decode` 9 (the pair at M = 1 included), `quantmoe` 13 pass (`raw/gates*.txt`).
- `batch_cuda_real` 35B: 4290 (row, step) pairs bitwise identical, 2 passed (B = 2, 3, 4, 8, 16; the triple engages at 16: nsys
  below), and again with `INTARSIA_CUDA_FP8_TRIPLE_ROWS=2` (the triple at every B): 4290 identical (`raw/batch-real*.txt`).
- Served `equal.py` (base 7aebefb83 vs dev, both `serve --max-batch 16`): 21/21 cells identical, 9408 token ids (`raw/ident/`).
- Mac `cargo test --workspace --release --no-fail-fast`: green (EXIT 0).

## Per kernel at B = 16 (nsys, VPROF's units, `scripts/qkv16/prof.sh`, `raw/dev-b16`, `raw/dev-off-b16`)

| per attention layer | triple off (= main's path) | triple on |
|---|---:|---:|
| q + gate (8192 x 2048), `fp8_gemv_rows_q_kernel` grid 96 | 89.4 us | (in the triple) |
| k + v pair, `fp8_gemv_rows_q_multi_kernel` grid 48 x 512 | 28.0 us | (in the triple) |
| sum / triple (grid 96) | 117.4 us | about 114 us |
| per step (10 layers) | 1.17 ms | 1.14 ms |

The triple's time is the multi kernel's step total minus the 30 GDN pairs (3.83 ms per step, measured in the off run): 4.97 - 3.83 =
1.14 ms. The GDN pair itself is unchanged by the refactor (128.1 us here, 129.4 us in GDNX's doc). The q + k + v bucket goes
**down by 0.03 ms per step (-3 %)**: the saved launch is small and the k / v time that the side stream hid beside q is now serial.
Burst shape: 247 of the 295 batched decode steps ran 16 rows (`raw/dev-b16/metrics.txt`), so the triple is on for most of the window; the
decode pass mean is 42.15 ms on against 42.47 ms off (250 vs 246 passes in the window; noise-sized).

## Step time (`scripts/qkv16/steptime.sh`, `timing_35b_batched_step`, dev binary, env A/B, 2 interleaved rounds off / default / bar 2 / bar 2 / default / off)

Mean of the test's three 60-step runs (min / median / max over its three runs, ms; then the mean over 2 rounds):

| arm | B = 16 min | B = 16 median | B = 8 min | B = 8 median |
|---|---:|---:|---:|---:|
| `TRIPLE_ROWS=0` (q beside the pair, main's path) | 41.236 | 42.547 | 26.026 | 27.167 |
| **default (triple from 12 rows)** | **41.126** | **42.561** | 25.946 | 27.190 |
| `TRIPLE_ROWS=2` (triple at every B) | 41.321 | 42.662 | 25.950 | 27.184 |

B = 16: -0.11 ms on the min, +0.01 ms on the median: inside the run-to-run spread (about 0.1 ms between identical arms). The triple
at B = 8 (bar 2) is flat too, so lowering the bar would not help. `raw/steptime-w1.txt`.

## Serve A/B (localeval speed, base 7aebefb83 vs dev, `serve --max-batch 16`, 3 rounds interleaved, aggregate decode tok/s)

| streams | base r1 / r2 / r3 | dev r1 / r2 / r3 | mean base | mean dev | delta |
|---:|---|---|---:|---:|---:|
| 1 | 80.3 / 80.8 / 80.7 | 80.7 / 80.7 / 80.5 | 80.60 | 80.63 | +0.0 % |
| 4 | 174.0 / 175.9 / 177.1 | 174.5 / 174.6 / 180.7 | 175.67 | 176.60 | +0.5 % |
| 8 | 233.1 / 231.0 / 234.1 | 236.0 / 236.7 / 229.8 | 232.73 | 234.17 | +0.6 % |
| 16 | 301.1 / 305.9 / 307.7 | 302.7 / 305.5 / 302.1 | 304.90 | 303.43 | -0.5 % |

Cells at 1, 4 and 8 streams run the same kernels as main (the triple is off under 12 rows); their differences are noise. The 16-stream
cell is the only one the change touches: -0.5 % on the mean, from one slow dev round (302.1 against 307.7); per round -/+0.5 %, -0.1 %, -1.8 %.
That is at the edge of the -0.5 % gate and inside the round-to-round spread, so nothing here shows a win or a loss. Raw: `raw/w2/`.

## Gates

| gate | result |
|---|---|
| rows_exact (triple vs singles 1..16), row_tables | pass |
| batch_cuda_real 35B all pairs bitwise (default and bar 2) | pass |
| served equal.py 21/21 | pass |
| plan show 8B / 35B / Gemma-4 byte-identical | pass |
| Mac workspace tests | pass |
| step B = 8 / 16 not slower | flat (within noise) |
| q / k / v bucket down | -0.03 ms / step (-3 %) |
| serve A/B no cell slower than -0.5 % | 16 streams -0.5 % (noise); the rest >= 0 |
| step B = 16 improves beyond noise (ready-PR bar) | **no** -> draft |

## Why it is flat, and what would change it

GDNX's estimate (-0.35 ms) assumed the k + v pair was exposed. It is not: INTARSIA_SIDE_PROJ already runs it beside the q GEMV, and q
at 8192 rows fills the GPU, so the pair's 28 us is mostly hidden (or costs the little the DRAM-bound q gives up). Merging removes that
overlap and a launch, and the 8192 + 512 + 512 walk is as bandwidth-bound as q alone (114 against 89 + 28 us), so the best case is the
launch gap (a few us per layer, about 0.03 ms per step). vLLM's 85.7 us for q + k + v (VPROF) is a bandwidth difference (216 GB/s there,
about 190 here), not a launch-count one; the lever is the streaming rate of the e4m3-staged kernel, not merging.

## Reproduce

`scripts/qkv16/` (build.sh, gates.sh, batch-real.sh, steptime.sh, prof.sh, ident.sh, le-window.sh, plan.sh, w1.sh, w2.sh), dev =
this branch, base = origin/main 7aebefb83. Knob: `INTARSIA_CUDA_FP8_TRIPLE_ROWS` (docs/tuning.md). GPU windows: 2 (W1 12:05-12:20, W2 12:27-12:39)
plus the plan show check; spark1 `~/intarsiaml-qkv16` removed afterwards.
