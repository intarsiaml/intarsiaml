# cmx: CUDA MXFP4 W4A16 prefill GEMM, Qwen3.8-27B on the Spark (2026-10-27)

Lane `cmx`, branch `lane-cmx/cuda-mxfp4-prefill`, base `origin/main` `74211fc33` (both A/B trees are built from that commit; the branch was then merged with the later main, see the PR). DGX Spark **spark1** (GB10 sm_121), `Qwen3.8-27B` MXFP4 (Quark, `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`), recipe defaults
(W4A16, `--spec mtp --spec-k 2`, prefill chunk 256, CUDA graphs). Raw logs: `2026-10-27-cmx-cuda-mxfp4-prefill/raw/`. Scripts: `scripts/cmx/`.
Status: **draft PR, not merged** (one gate misses by 0.6 point, see Gates).

## Result

Prefill went from 64 tok/s to about 840 tok/s at every size (13x), TTFT at 4k from 63.6 s to 4.9 s.

| MTP on (default), localeval, 3 interleaved rounds | base prefill tok/s | branch prefill tok/s | x | base decode | branch decode | x |
|---|---:|---:|---:|---:|---:|---:|
| 1k (992 / 989 tokens) | 64 (64-64) | 786 (784-792) | 12.3 | 22.76 | 22.66 | 0.996 (paired rounds 0.984, 0.981, 0.996) |
| 4k (4088 / 4089 tokens) | 64 (64-64) | 841 (841-844) | 13.1 | 22.29 | 23.22 | 1.042 (paired median 1.064) |
| 16k (about 16.6k tokens, ONE sample per arm, not localeval) | 65 (TTFT 255.6 s) | 838 (TTFT 19.8 s) | 12.9 | 20.49 | 20.80 | 1.015 |

`--spec off` control (localeval, 3 rounds): prefill 1k 64 -> 790 (+1135 %), 4k 64 -> 841 (+1213 %); decode 13.61 -> 13.68 (1.005) and 13.41 -> 13.44 (1.002).

TTFT at 4k (localeval, 4088 / 4089 prompt tokens): **63.63 s -> 4.86 s**. Marginal 4k prefill 64 -> 861 tok/s.

Gap to TensorFold's 580 tok/s on one Spark: the branch is **1.45x above it** (841 vs 580) at 4k. The two numbers come from different harnesses and prompts, so read it as
"at or above". There is no gap left to close with this change; the next ceiling is the bf16 `mma.sync` rate of the GB10 (about 50 TFLOP/s in the microbench).

The 16k row is a single-sample script (`scripts/cmx/dec16k.sh`, same as lane s27d), not localeval: a localeval warmup plus 3 reps at 16k would cost 17 min per base round. The
user's rule is localeval for speed numbers; this is the deviation.

## What changed

- `kernels/cuda/mxfp4_gemm.cu` (new): `ik_matmul_mxfp4_mma`, W4A16 GEMM. `kernels/cuda/wq_gemm.cuh` gets `FMT_MXFP4`: the weight tile is cp.async-staged,
  dequantised once per tile and k step to bf16 in shared memory (e2m1 x 2^(e8m0-127) is exact in bf16), bf16 `mma.sync` m16n8k16, f32 accumulate, the NVFP4 pipeline
  otherwise unchanged (3-slot activation ring, full-row stores). Activations stay bf16: no activation quantisation, no block-scaled FP4 MMA. K % 128 == 0.
- `kernels/cuda/mxfp4.cu`: `ik_matmul_mxfp4_gemv_rows`, the `rows_exact` twin of the GEMV (the same launcher: the GEMV is already bitwise per row at any M).
- Registry (`crates/intarsia-backend-cuda/src/quantmoe.rs`): `matmul_mxfp4_mma` serves **ordinary** (not `exact_rows`) **W4A16** calls with M >= 5 and K % 128 == 0,
  priority 20 over the GEMV's 10. `exact_rows` calls (verify passes, batched steps) get `matmul_mxfp4_gemv_rows` only. M = 1..4 and W4A4 / W4A8 calls keep the GEMV.
  `ShapeInfo::mxfp4_w4a16` (new) lets a kernel require a W4A16 call at plan time. Threshold 5 comes from the microbench: the GEMM is flat at 271-297 us for gate/up
  (17408 x 5120) for every M <= 64 (170 GB/s), the GEMV is 196 / 209 / 241 / 283 / 444 / 638 us at M = 2 / 3 / 4 / 5 / 8 / 9 and 3558 us at M = 64.
- Knob `INTARSIA_CUDA_MXFP4_MMA=0` keeps the GEMV for every M (`docs/tuning.md` row; bitwise the old path, shown below).

## Numerics and row invariance

The GEMM is not bitwise the GEMV (different f32 summation order). Each output element is one fixed k-ordered chain of mma steps whatever M, tile height (16, 32, 64) or
row position, so a row's bits do not depend on M or on its neighbours. `tests/mxfp4_mma.rs` (GPU): rows of M = 1..255 calls equal the same rows of a 300-row call byte for
byte, a reversed window gives the reversed rows, repeat runs are identical; 4 shapes x bf16/f32 output. The microbench repeats the check on the six 27B shapes at 17 M values: all `rowinv 1`.
GEMM vs GEMV: max relative difference 2.0e-3 to 4.1e-3 of max |out| (bf16 outputs), rel_l2 5.8e-5 in the admission sweep.

## Gates

| Gate | Result |
|---|---|
| prefill 1k/4k >= 4x base, 3 interleaved rounds | **PASS**: 12.3x / 13.1x; paired rounds 1k +1126 %, +1134 %, +1122 %; 4k +1209 %, +1215 %, +1211 % |
| decode unchanged, `--spec off` >= 0.99x | **PASS**: 1.005 (1k), 1.002 (4k) |
| decode unchanged, MTP on >= 0.99x | **MISS at 1k, by 0.6 point**: paired rounds 0.984, 0.981, 0.996 (median round 0.996 / aggregate 0.984 by `agg.py`); 4k 1.042-1.064; 16k single sample 1.015. See "MTP-on decode" below. |
| KL vs 27B W4A16 golden (`qwen3.8-27b-mxfp4-hfquark-w4a16-top20`, sha256 `32e2b96d`), `--mxfp4-act none`, `--mode both`, 10 prompts, 1441 positions | **PASS**, no recorded noise floor exists for this golden, so the static floor 3.0e-2 binds. Prefill KL21 (scored rows): base 5.60e-4, branch **4.39e-3**, branch with knob 0 5.60e-4 (identical to base). Decode (teacher-forced, 16 positions per prompt): base 1.08e-3, branch 6.34e-4, knob 0 1.08e-3. top-1 99.58 % (branch) vs 99.44 % (base). The prefill KL rises 8x and stays 7x under the floor; the worst prompt (58 positions) goes 2.8e-3 -> 8.9e-2 per prompt (one top-1 flip) and is not gated per prompt. |
| MMLU n=1425 >= 0.81, once on the branch | **PASS: 0.818** (1425 replies, 0 truncated, thinking off, `serve` recipe defaults; HIP W4A16 reference 0.822, `2026-10-27-mmlu27-27b-quality.md`) |
| greedy ids, `--spec off`, 256 tokens, identical to base on short prompts | **PASS** on 4 prompts (41, 34, 218, 556 tokens): IDENTICAL, also with knob 0 and with `--spec mtp`. The 3870-token prompt diverges **at id 5 of 256** (knob 0 is identical to base there), as expected for changed prefill numerics. |
| batched rows == solo | **PASS**: `real_35b_batched_decode_is_bitwise_the_solo_step` on the 27B, N = 2..16, 4290 (row, step) pairs bitwise identical |
| MTP spec-on == spec-off | **PASS**: `--spec mtp` ids == `--spec off` ids on all 5 probe prompts (the 3870-token one included) |
| verify rows == single-row steps (27B, `verify_rows_real`, q <= 4, ctx 17..4000) | **PASS with the engine's per-row verify attention** (`INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=1`, what `--spec mtp` runs): 0 diffs, with and without the GEMM. Without it the probe reports 4 windows with diffs (max |dlogit| 0.115); **the same diffs appear with `INTARSIA_CUDA_MXFP4_MMA=0` (0.106)**, so they come from the multi-row attention, not from this change |
| `kernels test` admits the kernel, >= 3 planted mutants caught | **PASS**: `matmul_mxfp4_mma` ADMITTED (10 cases, rel_l2 5.8e-5), `_gemv` and `_gemv_rows` admitted; full CUDA sweep 79 kernels / 1597 cases, no REJECTED. **6 of 6 mutants caught** (e8m0 bias, scale per piece, scale row stride, scale step stride, output scale, k-half scale offset; `scripts/cmx/mutants.py`) |
| `plan show` byte-identical 8B, 35B, Gemma-4 on CUDA | **PASS**: json and text for all three (md5 `b1a1150e`/`7742cef8`, `ca8cba7e`/`2c7deccd`, `82ea8282`/`418cbfae`). The 27B plan differs, as intended (new kernel rows) |
| Mac `cargo test --workspace` | **PASS**: 1044 passed, 0 failed (`raw/cargo-test-workspace-mac.log`) |
| CI green | see the PR |
| knob `INTARSIA_CUDA_MXFP4_MMA=0` restores the old path | **PASS**: prefill 57 / 63 / 64.5 tok/s as base, ids identical to base on every prompt including the 3870-token one, KL equal to base to all digits |

GPU unit tests on spark1 (`INTARSIA_REQUIRE_GPU=1`): `mxfp4_mma` 3/3, `rows_exact` (new `mxfp4_rows_are_bitwise_the_one_row_gemv`) 6/6, `rows_registry` 3/3, `quick_coverage` 1/1.

### MTP-on decode, the one miss

The decode path is not touched: M = 1 and every verify/batched row use the GEMV launcher (`gemv_rows` is the same function), proven bitwise by the verify and batched-row probes
above, and `--spec off` decode is 1.005 / 1.002. What differs with MTP on is the token stream: prefill numerics change the KV, so the 128 generated tokens differ and the draft
acceptance rate differs with them. The 1k cell shows -1.6 %, the 4k cell +4 to +6 %, the 16k sample +1.5 %. Not proven noise: the per-round spread of one arm is up to 3 % at 1k.
The lane had used its 6 quiet windows (microbench, GPU unit tests, two A/B windows, ids, 16k) and could not repeat the 1k MTP-on pair, so the PR stays draft.
To close it: one decode-only A/B (1k, `--gen 256`, 5 rounds, `--spec mtp`) in a single window of about 20 min; merge if >= 0.99.

## Microbench (scripts/cmx/mxfp4bench, raw/microbench.log)

Gate/up 17408 x 5120, us: M = 5 GEMV 283 / GEMM 271; M = 16 886 / 275; M = 64 3558 / 295; M = 128 GEMM 448; M = 256 943; M = 1024 3677 (49.7 TFLOP/s);
M = 4096 32316 (22.6 TFLOP/s, an outlier, not on the served path: chunk 256). Other 27B shapes reach 44-55 TFLOP/s at M >= 256.

## Leftovers

1. The MTP-on 1k decode pair above (one window).
2. Hybrid batching with prefill chunks inside an `exact_rows` call still uses the slow GEMV; the GEMM serves only ordinary calls (by design: its bits differ from the GEMV's).
3. M = 4096 in one call is half the TFLOP/s of M = 1024 (microbench only; chunk 256 serves). Not investigated.
4. HIP/Metal unchanged. The `plan show` of the 27B on CUDA now lists the two new kernels.
5. `~/intarsiaml-s27d` on spark1 deleted at the end of the lane (see the report); its checkpoint copy was used from there until then, so a rerun of `scripts/cmx/*` needs `MODEL=` pointed at a checkpoint.
