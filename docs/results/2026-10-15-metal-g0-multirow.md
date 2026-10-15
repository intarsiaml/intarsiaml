# Metal G0: multi-row bit-exact decode GEMV (speculative-decoding verification kernel go/no-go)

Date 2026-10-15. Branch `metal-g0-multirow` (off `metal-next-levers` 215faed). Gate G0 of `docs/research/metal-next-levers.md`.
Machine: endor, Apple M5 Max.

## Status

* Kernels, host launchers, registry entries, bitwise tests and the benchmark are built and committed.
* **Bitwise equality with the serving 1-row decode GEMV: verified** (section 2).
* **Timing (T(R)/T(1)): NOT RUN in this session.** The exclusive fleetq slot on endor was never granted before the report
  (see section 5), and the rules for this task forbid GPU measurements outside a slot. There is therefore **no GO / NO-GO
  verdict**. Everything needed to produce it is in place: one command, `docs/results/raw/2026-10-15-metal-g0-multirow/run_slot.sh`,
  then `analyze_g0.py` (section 4).

## 1. What was built

All new code; the serving decode path, `crates/intarsia-core`, CUDA and HIP are untouched.

| File | Content |
|---|---|
| `kernels/metal/gemv_multirow.metal` | Five shader families, each a template over `R` activation rows and `P` output rows per threadgroup (per simdgroup for logits), host names `<kernel>_mr<R>x<P>`, `R` in {1..8, 16}, `P` in {1, 2, 4} |
| `crates/intarsia-backend-metal/src/multirow.rs` | Launchers (`Matmul`, `MatmulPair`, `Logits`), encoders for benches and tests, name table, fallback to R ordinary 1-row launches for alignments the 1-row path serves with its scalar kernels |
| `crates/intarsia-backend-metal/tests/multirow.rs` | Bitwise tests against the serving 1-row launches (11 tests) |
| `crates/intarsia-backend-metal/examples/gemv_g0.rs` | The G0 bench (gemv_b3 method) |
| `docs/results/raw/2026-10-15-metal-g0-multirow/` | `run_slot.sh`, `analyze_g0.py`, logs |

The families mirror the kernels the serving path picks (`kernels.rs`, `plan_gemv`, `fused.rs`, `dense.rs`):

| Op (Qwen3-8B use) | Serving 1-row kernel | Multi-row shader |
|---|---|---|
| `Matmul`, 16-byte aligned (q, o, down; 28-29 of 36 layers) | `gemv_bf16_rt_u2s8` (one threadgroup of 8 simdgroups per row, 2 x 16-byte loads per lane) | `gemv_bf16_rt_u2s8_mr<R,P>` |
| `Matmul`, 8-byte aligned (shard 1, header 8 mod 16; 7-8 of 36 layers) | `gemv_bf16_rt_a8u4s8` | `gemv_bf16_rt_a8u4s8_mr<R,P>` |
| `MatmulPair` (k+v, gate+up), both alignments | `gemv_pair_bf16_rt_u2s8`, `..._a8u4s8` | `gemv_pair_bf16_rt_{u2s8,a8u4s8}_mr<R,P>` |
| `Logits` M=1 (lm_head, f32 out; lm_head is 16-byte aligned) | `gemv_f32o_v` | `gemv_f32o_v_mr<R,P>` |

Design: every thread keeps `R` (times `P`) independent accumulators. The lane-to-chunk mapping, the `dot8`/`dot4` fma chain,
the `simd_sum`, and the fixed-order sum of the 8 per-simdgroup partials are those of the mirrored kernel, so row `r` of the
output is bitwise the 1-row result for activation row `r` regardless of `R` and `P`. Weights are loaded once per thread and
used against all `R` rows. `P > 1` additionally shares each loaded x chunk between `P` weight rows. The reason: x is re-read
from cache by every threadgroup, so at `P = 1` the cache traffic for x is `R` times the 1-row kernel's. That traffic (not DRAM
bytes, not obviously ALU) is the thing I expect to limit `R = 8`; `P` is the lever. This is a hypothesis until the bench runs.

Output layout is `x [R,K]`, `y [R,N]` row-major. The launcher default for `P` is a stub (`default_per_tg` returns 1) to be set
from the bench.

## 2. Bitwise equality (measured, passed)

`crates/intarsia-backend-metal/tests/multirow.rs` compares, for every activation row, the multi-row output byte for byte
(including guard bytes around the output) with `R` separate launches of the serving kernels (`matmul_bf16_gemv`,
`matmul_pair_bf16_gemv`, `logits_bf16_gemv`), with distinct, differently scaled x rows:

* Qwen3-8B shapes, random weights: 4096x4096, K=12288 down, 1024 and 12288 pairs, lm_head 151936x4096, all `R` in {1..8, 16}.
* Every `(R, P)` instance of every family, by direct encoding (`all_variants_*`), including K tails (K=4104, 8200, 520, 1032),
  odd N (1031, 33, 5, 3), 8-byte aligned weights (offset 8) and `K % 4 == 0` only.
* Scalar-kernel alignments (2-byte offsets, K odd): fallback path, per-row launches.
* **Real Qwen3-8B weights**, mmap'd and wrapped with `wrap_host` exactly as the loader does (so the true 16-byte and 8-mod-16
  alignments occur): layer 0 (shard 1, 8 mod 16) and layer 20 (shard 3) q, o, down, k+v, gate+up, and lm_head (shard 5), M in
  {2..8, 16} (`real_qwen3_8b_weights_bitwise`).
* The test has teeth: reversing the accumulation order of the two chunks per lane in the 16-byte shader makes it fail.

Result: all 11 tests in `tests/multirow.rs` pass (`cargo-test-metal.log`).

The bench repeats the check on its own buffers before timing (every `(R, P)` against `R` serving launches) and aborts on a
difference.

## 3. Registry, `kernels test`, and what could not be done

* `matmul_bf16_gemv_rows` (Op::Matmul, 2 <= M <= 8), `logits_bf16_gemv_rows` (Op::Logits, 2 <= M <= 8) and
  `matmul_pair_bf16_gemv_rows` (Op::MatmulPair, 2 <= M <= 8, K % 8 == 0) are registered with priority -1: below the tiled GEMM
  (0) and the NAX GEMM (5), so serving never selects them. A test asserts the registry's pick for M = 2, 3, 8, 15 is unchanged.
* The pair entry is **not listed by `Backend::kernels()` unless `INTARSIA_METAL_MULTIROW=1`**. Reason: no pair kernel accepts
  M = 2..8 today, so a listed entry would silently make the model fuse short prefill chunks with it, which changes prefill
  numerics. A first attempt (predicate gated on the env var) made the default `kernels test` fail with "NO CASES (untested)", hence
  the unlisted form.
* `intarsia kernels test --backend metal -m ~/models/Qwen3-8B`: default run admits the two other entries (25 kernels, 281
  cases, exit 0); with `INTARSIA_METAL_MULTIROW=1` all three are admitted (26 kernels, 293 cases, exit 0)
  (`kernels-test-metal-default.log`, `kernels-test-metal-multirow.log`).
* **The harness cannot grade "bitwise vs the 1-row reference".** `admission::run_admission` grades every candidate against the
  cpu backend's reference kernel; `Exactness::Exact` there means bit equality with the CPU result, which no Metal GEMV (the
  1-row one included) satisfies. So the entries are `Exactness::Deterministic` (graded at the usual bf16 / f32 tolerance:
  matmul max_abs 2.2e-4, logits 1.1e-7) and the bitwise-vs-1-row property is proved by `tests/multirow.rs` instead. Admitting it as
  `Exact` in `kernels test` needs a change in `crates/intarsia-core` (a per-kernel `bit_reference: Option<kernel name>` that makes
  `run_admission` compare against the same backend's named kernel run row by row), which the task excludes.

## 4. How to produce the numbers

```
fleetq acquire --host endor --exclusive --mem 20 --label "metal G0 multirow gemv" --wait   # note the id N
FLEETQ_JOB=N docs/results/raw/2026-10-15-metal-g0-multirow/run_slot.sh                       # tests, kernels test, 3 bench runs, macmon at 100 ms
fleetq release N
python3 docs/results/raw/2026-10-15-metal-g0-multirow/analyze_g0.py docs/results/raw/2026-10-15-metal-g0-multirow
```

The bench (`examples/gemv_g0.rs`): per shape (qo, down, kv pair, gu pair, lm), one command buffer of a serial dispatch chain over a
rotating 3 GiB pool of distinct weight buffers (DRAM-streaming, like a decode step), timed by GPU start/end; candidates
interleaved with a rotating order inside each round; 30 rounds; raw samples with epoch stamps. Candidates: the serving 1-row
kernel, multi-row `R` in {1..8, 16} x `P` in {1, 2, 4}, the tiled GEMM at M = 2..8, 12, 15, 16 (a pair is two tiled GEMMs, as the
model runs it today) and the NAX GEMM at M = 16. `analyze_g0.py` joins rounds to the macmon trace and drops every round in which
the GPU clock fell below 1450 MHz, takes T(R)/T(1) as a median of within-round ratios, and prints the per-shape tables, the
layer sum (2 x q/o + down + k/v + gate/up) against the pass criterion (T(4)/T(1) <= 1.10 and T(8)/T(1) <= 1.20 summed over a
layer's shapes), the census-weighted layer (16-byte vs 8-mod-16 mix of the real shards) and the per-step estimate
(36 layers + lm_head over `R` rows; attention with q_len > 1 is not covered).

## 5. Why there are no timings

Slot #1978 (endor, exclusive) was queued from 07:30 behind #1975 (another agent's manual hold, "metal p3 4k noise") and
#1977 (the 4k fp32 CPU golden, which started at about 08:15 and was expected to run for hours). Per instructions neither was
touched and no GPU measurement was made outside a slot. Correctness runs (`cargo test`, `kernels test`) were run without a
slot because they are short functional checks, not measurements.

## 6. Expectation, flagged as such

Not a result. At `R = 8` the arithmetic per weight byte is eight times the 1-row kernel's; 580 GB/s of bf16 weights is
about 290 G weights/s, i.e. about 2.3 T weight-row FMAs/s at `R = 8` plus the x converts, against an fp32 FMA peak of the order of
8 T/s (not measured here), and the x cache traffic described in section 1. Either could break the 1.20 limit at `P = 1`, which is
why the `P` variants exist. The verdict needs the bench.
