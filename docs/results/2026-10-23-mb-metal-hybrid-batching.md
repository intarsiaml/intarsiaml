# Hybrid continuous batching on Metal (Qwen3.6-35B-A3B NVFP4), lane mb, 2026-10-23

Branch `lane-mb/metal-hybrid-batching` (base `origin/main` 240e9d2a), box endor (M5 Max, 110 GB unified memory). Metal twin of
`docs/results/2026-10-23-rb2-cuda-batching.md` (CUDA, PR #117). Draft PR, not merged.

![One batched decode step on Metal](2026-10-23-mb-metal-hybrid-batching/step.svg)

## 0. Summary

- `intarsia serve --backend metal --max-batch N` (N = 2..8) now serves the 35B (GatedDeltaNet + MoE + gated attention) several
  sequences at a time. **Every decode row is bitwise the solo step**: 4,550 (row, step) pairs for N = 2..8 over 130 steps, and the
  batched engine equals `LocalEngine` token for token on the 35B, including prompts of 2.6k, 4k and 16k tokens prefilled beside a
  decoding neighbour. Without `--max-batch` nothing changes (section 5).
- G-B (row r of an M-row kernel is bitwise the M = 1 launch, M = 2..8): **all kernels of the 35B decode step pass**. Three kernels
  needed a new `rows_exact` registration (fp8 GEMV, nvfp4 GEMV, MoE experts); the rest of the step already was row-exact (bf16
  GEMVs, `matmul_pair`, `logits`) or row-parallel (rmsnorm, silu_glu, gate_split, sigmoid_mul, moe_topk, moe_combine, argmax, embedding).
- Speed (localeval through `intarsia serve`, 1k-token prompt, aggregate decode tok/s at 1 / 4 / 8 streams): without batching 124.2 / 78.5 / 78.4
  (128 generated tokens), with `--max-batch 8` 122.6 / 101.3 / 107.5; with 512 generated tokens 123.5 / 106.6 / 105.8 against
  123.3 / 158.4 / 173.7. While all streams decode, 4 streams run at 196 and 8 at 224 tok/s aggregate (section 7).
- Cap: **8** (section 6). The MoE kernel reads each distinct expert once per step across the rows (vLLM's grouping), and is
  bitwise; in a micro-benchmark it only pays when rows share experts, and end to end it is not the bottleneck (section 4).
- Open: the per-row GDN conv / delta-rule and attention launches grow the step by about 3.8 ms per extra row; batching them (a
  row axis in those kernels) is the next lever (section 7).

## 1. What changed

| file | change |
|---|---|
| `kernels/metal/fp8.metal`, `crates/intarsia-backend-metal/src/fp8.rs` | `matmul_fp8_gemv_rows` (registered `rows_exact`, M = 2..8): reuses the proven-by-construction `_m8` shader and adds an `_m4` instantiation so K = 4096 (GDN / attention output projections) works at M = 5..8 in two passes of 4 rows; the solo `matmul_fp8_gemv` launch is byte-for-byte unchanged |
| `crates/intarsia-backend-metal/src/nvfp4.rs` | `matmul_nvfp4_gemv_rows` on the existing `nv_gemv_*_m8` shader |
| `kernels/metal/moe.metal`, `src/moe.rs` | `moe_experts_bf16_rows` (T = 2..8): `moe_dec_gu_x` / `moe_dec_dn_x` + the existing `moe_gather`; a threadgroup row per (token, k) pair; the first pair of each expert serves every pair of that expert (found with one simd ballot per 32 ids), so each expert's weight block is read and decoded once for all its pairs |
| `src/kernels.rs`, `src/lib.rs` | registrations, dispatch, kernel list pin |
| `crates/intarsia-core/src/model/batch.rs` | `allow_recurrent_batch` accepts `metal`; `has_hybrid_blocks()` |
| `crates/intarsia-models/src/load.rs` | on Metal, a model with hybrid blocks opts in when `--max-batch > 1`; busy chunk = solo chunk for it |
| tests | `crates/intarsia-backend-metal/tests/rows_exact.rs` (G-B), `crates/intarsia-models/tests/batch_metal_real.rs` (G-C, engine, long prompts, timing), `batch_recurrent.rs` now also runs on Metal (`INTARSIA_BATCH_TEST_BACKEND=metal`) |
| `docs/tuning.md` | the batching row |

Not touched: the scheduler, the fusion table (`crates/intarsia-core/src/fusion.rs`: Metal registers none of the fused ops for
the 35B, so the batched plan has nothing to expand), `plan::check_batch_ops` (it already compares kernel picks and exempts only
`rows_exact` registrations; it runs at `configure_plan` and passed on the first load), the prefix store and MTP refusals (both
are keyed on the batch size, not the backend).

## 2. G-B: row-exact kernels

`tests/rows_exact.rs` runs each op once on the `[M, ...]` batch (the kernel a batched step selects) and M times on 1-row slices
of the same inputs (the solo step's launches), and compares every output byte, guard bytes included. Inputs per row: random,
wide-range (2^-18..2^18), saturating (past e4m3's 448), all zero, one nonzero element, small noise; buffers at aligned and
misaligned offsets (weights 8 and 16 mod 16, activations 0 and 2 mod 4); W8A8 (`input_scale`) and W8A16; bf16 and f32 outputs;
e4m3 NaN codes in some weights.

| kernel | M | cases | rows compared | shapes |
|---|---|---|---|---|
| `matmul_fp8_gemv_rows` (new) | 2..8 | 672 | 3,360 | 8192x2048, 4096x2048, 512x2048, 2048x4096, 4096x4096, odd (130x512, 33x48, 65x16) |
| `matmul_nvfp4_gemv_rows` (new) | 2..8 | 168 | 840 | 512x2048, 2048x512, 248320x2048, odd |
| `moe_experts_bf16_rows` (new) | T = 2..8 | 168 | 840 | 256 experts, top-8, H 2048, I 512; routings: random, all tokens identical, overlapping, invalid ids (-1, 256, i32::MIN), a token with no valid expert, one expert repeated beyond 8 pairs |
| `matmul_pair_bf16_gemv_rows` (existing) | 2..8 | 84 | 420 | router + shared-gate (256, 1, 2048), GDN in_b + in_a (32, 32, 2048) |
| row-parallel ops (same kernel at M rows and at 1 row) | 2..8 | 98 | 490 | rmsnorm (2048, 4096, 32x128, 16x256), rmsnorm_add, silu_glu, embedding, gate_split, sigmoid_mul, moe_topk (with ties), moe_combine, argmax (vocab 248320, f32 and bf16, with a tied maximum) |

Result: 6 of 6 pass (fleetq #4613, then again at the final code in #4621). The bf16 `matmul`, `logits` and attention rows
kernels were proven by lane g0/g0b (`tests/multirow.rs`, `g0b_*.rs`) and are not re-proven here; the GDN conv and delta rule
run once per row with the solo kernel, as on CUDA.

Why the fp8 and nvfp4 kernels were already row-exact: the shaders accumulate each (row, output) in the order `u = lane + 32 n`,
whatever the unroll (`UN`) or rows-per-simdgroup (`RS`) template arguments are, with one `simd_sum` at the end, so the `_m8` form
only changes how many rows share a weight read. The test would catch a break: a deliberately M-dependent association in the
multi-row fp8 shader (`(acc + d/2) + d/2` on odd unroll steps) fails `fp8_rows_are_bitwise_the_one_row_gemv`; I did not commit it.

Variants written: none needed for exactness. Two additions for coverage: the `_m4` fp8 instantiation (K = 4096 does not fit 8 rows
of activations in 32 KB of threadgroup memory) and the MoE grouping kernels (the existing grouped path uses simdgroup matrices and
is M-dependent, so it is not used in batched decode; the existing decode path reads each expert once *per pair*).

## 3. G-C and engine gates

| gate | what | result | fleetq |
|---|---|---|---|
| tiny hybrid on Metal (bf16; dense, MoE W8A16, MoE W8A8) | `Model::forward_batch` N = 2, 3, 4, 8 x 130 steps; decode beside prefill chunks; `BatchedEngine` equals the single-sequence engine over 130+ tokens (2..8 slots); memory plan; batched plan asserted to pick only `_rows` kernels and the solo per-row kernels | 7 of 7 pass | #4614 (first code), #4621 (final) |
| **35B, Metal, `forward_batch`** | N = 2..8, mixed context lengths 5 to 300, greedy and seeded-sampled rows, 130 steps each; decode beside 256-token prefill chunks | **4,550 (row, step) pairs bitwise identical** (logits bit for bit, argmax token for token, state identical) | #4621 |
| **35B engine** | 20 requests (6 prompts, every 4th seeded-sampled, 130 tokens) alone on `LocalEngine`, then through `BatchedEngine` with 4 and 8 slots, one at a time and all at once | **every request token- and text-identical to its solo run** | #4621 |
| **35B engine, long prompts** | R0 (200 tokens, 900 generated) decoding while R1 (4,096), R2 (16,384, sampled) and R3 (2,600) prefill; 2 slots, 4 slots, 4 slots in another arrival order | **all identical to `LocalEngine`**; 2 / 12 / 12 mixed iterations (prefill beside decode), 3 / 13 / 13 prefill chunks | #4621 |
| dense `batch_metal` (Qwen3-8B BF16) | mixed lengths 5..4000, B = 2..8 | 138 (row, step) pairs bitwise identical | #4621 |

Busy chunk: the default busy chunk for a hybrid model on Metal is the solo chunk (2048 for the 35B on Metal, from the load log
"prefill chunk 2048 (recipe qwen3_5_moe, metal)"), as on CUDA, so a request that prefills beside decoding sequences runs its solo
schedule. **Control, not a gate**: with `--busy-chunk 64` the same four requests are also token-identical on Metal (on CUDA 1024
diverged at token 25 to 51). So on Metal the 35B's prefill output did not depend on the chunk size in this test (4 requests, token
level; no logits comparison was done, so this is not evidence of bitwise chunk invariance). The control means the long-prompt test cannot
by itself show that it would catch a schedule mismatch on Metal; the default stays the solo chunk because it is the only value
for which the bitwise claim is by construction.

## 4. MoE decode and expert de-duplication

vLLM reads each distinct expert once per step across the rows (`docs/results/2026-10-23-vr-vllm-maxseqs16.md`). The Metal
version does the same and stays bitwise: pair p of token t computes exactly the arithmetic of the T = 1 decode kernels (same
lane-strided block loop, same `fma` chain, same `simd_sum`, same bf16 roundings; the fp4 decode is hoisted out of the per-pair loop
but feeds the same chain), and `moe_gather` still sums a token's pairs in k order. The grouping only decides which threadgroup
reads a weight block: threadgroup row y is pair y; if an earlier pair has the same expert (and fewer than 8 earlier pairs ride
with it) it exits, else it serves up to 8 pairs of that expert. First versions sorted pairs with the grouped path's `moe_group`
pass (one threadgroup, serial prefix loop); that pass cost more than it saved, and it was replaced by the ballot scan, with no scratch
use beyond the `h` and `y` regions.

Micro-benchmark (`tests/rows_exact.rs timing_rows`, 24 routings cycled call by call, one 256-expert layer, us per call):

| T | route | T one-token calls | rows call | ratio |
|---|---|---|---|---|
| 2 | random | 72.7 | 97.9 | 0.74x |
| 4 | random | 143.9 | 179.2 | 0.80x |
| 8 | random | 284.5 | 349.9 | 0.81x |
| 4 | all tokens, same 8 experts | 125.9 | 126.6 | 0.99x |
| 8 | all tokens, same 8 experts | 244.4 | 227.1 | 1.08x |

With random routing about 57 of 64 pairs are distinct experts at T = 8, so there is little to share and the per-pair work in the
rows kernel is slightly more expensive than the single-pair kernel; it breaks even only when tokens share experts. The 35B step is
not limited by it: the batched step with the first (sorted) version and with this one measured the same (B = 8: 35.4 and 35.7 ms).
Because the rows kernel is never faster than the serial calls on random routing, and real traffic shares experts only
partially, **I did not show an end-to-end gain from de-duplicating expert reads on Metal**. The weights the experts stream
(one 1.5 MB expert = gate/up 1 MB + down 0.5 MB) are small next to the per-row launch cost (section 7).

fp8 / nvfp4 rows kernels (same benchmark, us, M one-row launches against one rows launch):

| kernel | M = 2 | M = 4 | M = 8 |
|---|---|---|---|
| fp8 8192x2048 | 58.3 -> 44.5 (1.31x) | 116.4 -> 82.8 (1.41x) | 231.9 -> 163.5 (1.42x) |
| fp8 2048x4096 | 32.8 -> 29.3 (1.12x) | 65.4 -> 55.9 (1.17x) | 130.0 -> 85.2 (1.53x) |
| nvfp4 512x2048 | 7.9 -> 5.6 (1.42x) | 14.9 -> 8.2 (1.81x) | 28.8 -> 13.7 (2.11x) |

## 5. No-op gates

Batching off (no `--max-batch`), branch `e65fe4ff` against `origin/main` 240e9d2a, both built `--release --features metal`, one
exclusive slot on endor (fleetq #4653, `--exclusive --mem 30`):

| check | result |
|---|---|
| `plan show` text and `--json`, Qwen3-8B and the 35B | all four byte-identical (1,342 / 8,371 B and 2,404 / 10,605 B) |
| 256 greedy ids (`generate --temperature 0 --ignore-eos --show-ids`), short prompt (16 tokens) and long prompt (4,395 tokens for the 8B, 4,503 for the 35B), both models | all four id lists and texts byte-identical |
| `intarsia kernels test --backend metal` (branch) | 44 kernels, 817 cases, 0 rejected, exit 0; the three new kernels ADMITTED |
| `intarsia accuracy` Qwen3-8B against `golden/qwen3-8b-fp32.safetensors`, base and branch | golden gate PASS for both |
| `cargo test --workspace --release` (endor) | 861 passed, 0 failed, 14 ignored (the opt-in real-model tests) |
| dense `batch_metal` (Qwen3-8B) | in section 3 (138 pairs bitwise identical) |

CUDA and HIP no-ops: **not run**. Beyond PR #117 the shared code changed in `Model::has_hybrid_blocks` (new, called only from the
Metal arm of `load.rs`), the metal pattern beside cuda in `allow_recurrent_batch` and in the busy-chunk condition (the cuda
conditions are the same as before), and the error text of `allow_recurrent_batch` for other backends (HIP). A CUDA or HIP run
executes none of those differently. Recorded in `docs/decisions/LOG.md`.

## 6. Cap

Metal cap for the 35B: **8** (`Backend::batch_row_cap` default, unchanged).

- Kernels: every multi-row fp8 / nvfp4 / bf16 GEMV and the MoE grouping take at most 8 rows (`GEMV_MB`, `MOE_XR`); 16 rows would
  need new kernels and a new G-B round (CUDA's 16 came from its own).
- Memory: weights 20.8 GB, workspace 0.7 GB; a sequence adds 62 MiB of recurrent state (8 sequences: 439 MiB, reserved from the
  KV budget by `enable_batch`) and 20 KiB of KV per token (160 MB for an 8k context). Eight sequences at 8k context come to
  about 23 GB of 110 GB. endor keeps about 50 GB of standing services (kev 18 GB, tensorfold 32 GB) and fleetq jobs, so a
  40 GB lease is enough and memory is not what limits the cap.
- Throughput: the step grows by about 3.8 ms per extra row, so aggregate decode flattens (section 7); more rows buy little
  until the per-row launches are batched.

## 7. Speed (localeval through `intarsia serve`, endor, fleetq #4665, #4688, #4694)

`intarsia serve --backend metal --model ~/models/Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --served-model-name intarsia` on endor, one server
per row, `localeval speed --sizes 1024 --reps 3 --concurrency 4,8` (1k-token prompt; `--gen` as stated), inside one
`fleetq run --quiet gpu,cpu` job per arm (script: `serve-and-bench` pattern of `deploy/localeval/serve-and-bench.sh`, started and
stopped by PID). "Aggregate" is localeval's: tokens of all streams over the wall time from the first request to the last token,
so the prefill of every stream (0.6 s per 1k-token prompt, one prompt at a time) is inside it. Single stream is the median of 3 reps.

| arm | gen | 1 stream | 4 streams | 8 streams | run ids |
|---|---|---|---|---|---|
| main (`origin/main`), no `--max-batch` (requests run one at a time) | 128 | 124.2 | 78.5 | 78.4 | `20261023-163455-adhoc-mb-main-nobatch` |
| branch, `--max-batch 8` | 128 | 122.6 | **101.3** | **107.5** | `20261023-163535-adhoc-mb-branch-maxbatch8` |
| main, no `--max-batch` | 512 | 123.5 | 106.6 | 105.8 | `20261023-165451-adhoc-mb-main-nobatch-gen512` |
| branch, `--max-batch 8` | 512 | 123.3 | **158.4** | **173.7** | `20261023-164944-adhoc-mb-branch-maxbatch8-gen512` |
| branch, `--max-batch 8 --busy-chunk 256` | 128 | 123.9 | 93.5 | 98.2 | `20261023-165046-adhoc-mb-branch-maxbatch8-busychunk256` |

Per stream while all streams decode (peak 32-token stretch, median over streams): 49 tok/s at 4 streams (196 aggregate) and
28 tok/s at 8 streams (224 aggregate), the same as the in-process step timing below; the aggregates in the table are lower because
every request first spends 0.6 s in a serialized prefill (TTFT median / max: 1.5 / 2.4 s at 4 streams, 2.8 / 5.0 s at 8 streams,
against 3.0 / 5.5 and 6.3 / 12.0 s without batching). Longer outputs hide the prefill: 512 generated tokens give 158 and 174.

- Against today's single-stream 124 tok/s: the batching engine costs about 1% at one stream (122.6 and 123.3 against 124.2 and
  123.5, inside the 2% spread of the runs). Aggregate against the same load without batching: 1.29x at 4 streams and 1.37x at 8
  with 128 generated tokens (101.3 / 78.5, 107.5 / 78.4), 1.49x and 1.64x with 512 (158.4 / 106.6, 173.7 / 105.8). Against the
  single-stream rate (124) the 512-token cells are 1.28x and 1.40x. (Without batching the aggregate is not the single-stream rate
  because the 1k-token prefills and decodes of the streams run one request after another.)
- The default busy chunk (2048) is also the faster one on this cell: busy chunk 256 gives 93.5 / 98.2 against 101.3 / 107.5,
  because a 1k-token prompt then needs four prefill iterations instead of one.
- CUDA for scale (`docs/results/2026-10-23-rb2-cuda-batching.md`, GB10, same 35B): 73.6 / 123.3 / 166.6 at 1 / 4 / 16 streams.
  The Metal gain per stream count is smaller because the per-row GDN and attention launches are already a large share of a
  Metal step (below).
- Comparators on endor: none. `/opt/code/model-benchmarks/data/records/localeval.jsonl` has no oMLX or mlx record for
  Qwen3.6-35B-A3B on endor. The only other 35B record on endor is llama.cpp UD-Q4_K_M (`20261022-142322-adhoc-qwen36-35b-udq4km-lcpp060`,
  single stream decode 91.3 / 91.8 / 96.7 tok/s, no concurrency numbers), and `docs/results/2026-10-22-md-metal-35b-decode-profile.md`
  records that no Metal comparator can load this checkpoint.

Step timing (`batch_metal_real.rs real_35b_batch_step_timing`, in-process `forward_batch`, all rows LastArgmax, 24 steps):

| B | ms / step | aggregate tok/s |
|---|---|---|
| 2 | 12.8 | 156 |
| 4 | 20.5 | 194 |
| 8 | 35.7 | 224 |

(The test's "solo step" figure of 31 ms is not comparable: it is measured differently from the served 8.05 ms step of
`docs/results/2026-10-22-md-metal-35b-decode-profile.md`, 124 tok/s.) Each extra row adds about 3.8 ms, 47% of a solo step. The
profile (`real_35b_batch_step_profile`, per-launch command buffers, so launch cost dominates every number) shows the ops that scale
with B are the per-row ones: `gdn_conv` and `gdn_delta_rule` (30 layers each), `attn_decode`, `qk_norm_rope_kv`. Their kernels are
row-independent (own state, own KV), so a launch over `[B]` sequences with per-sequence pointers is bitwise by construction
and needs one G-B test; that, and cutting the launch count, is the next step. Not done here.

## 8. Reproduce

```
# G-B, G-C (endor, inside a fleetq slot)
cargo test -p intarsia-backend-metal --release --test rows_exact -- --test-threads=1
INTARSIA_BATCH_TEST_BACKEND=metal INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_recurrent -- --test-threads=1
INTARSIA_QWEN36_DIR=~/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_metal_real -- --test-threads=1
#   ignored: long prompts, control, step timing / profile
INTARSIA_QWEN36_DIR=... cargo test -p intarsia-models --release --test batch_metal_real -- --ignored --test-threads=1 real_35b
INTARSIA_TEST_REAL_MODEL=1 cargo test -p intarsia-models --release --test batch_metal -- --nocapture   # dense gate
cargo test -p intarsia-backend-metal --release --test rows_exact -- --ignored --nocapture timing_rows
# speed: intarsia serve --backend metal --model ~/models/Qwen3.6-35B-A3B-NVFP4 --max-batch 8 --served-model-name intarsia --max-seq 8192
#   uv run --project /opt/code/deepeval localeval speed --url http://127.0.0.1:PORT/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 4,8
```

## 9. What this does not show

- Bitwise equality of the *prefill* chunks across schedules at the logits level on Metal (section 3 control compares tokens only).
- Any batch above 8, `--spec` (ngram / MTP) with batching on Metal (MTP and the prefix store are refused at batch > 1, as on CUDA),
  tensor parallelism, or models other than Qwen3.6-35B-A3B and Qwen3-8B.
- That expert de-duplication helps on Metal (section 4).
- CUDA and HIP no-ops were not run: the shared-code diff is Metal-gated (section 5).

## 10. Jobs

fleetq #4613 (G-B first run), #4614 (tiny hybrid first run), #4621 (35B gates, final-code reruns, timings; released), #4653 (Metal no-op, exclusive), #4665 / #4688 / #4694 (localeval speed arms, `--quiet gpu,cpu`). All released. Owner `claude-f31ccf91-mb`.
