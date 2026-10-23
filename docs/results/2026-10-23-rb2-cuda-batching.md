# Hybrid continuous batching on CUDA (Qwen3.6-35B-A3B NVFP4, stages R0 to R2), lane rb2, 2026-10-23

**Status: R0, R1 and R2 done and gated; the R2 speed target is missed (122.9 against about 140 tok/s at 4 streams).** Draft PR, do not merge.
Branch `lane-rb2/recurrent-batching-cuda`, from `lane-rb/recurrent-batching` (fa596759) with `origin/main` b745e978 merged in. Supersedes the
S1 to S5 plan of PR #114 where the two differ (decision rows in `docs/decisions/LOG.md`, 2026-10-23 06:41 to 09:20). Every number is
**measured** with a fleetq job or localeval run id, unless marked derived or estimate.

## 0. Summary

| | before (main) | now, `serve --max-batch 16` | note |
|---|---|---|---|
| aggregate tok/s, 1 stream | 74.2 | **73.6** (batching off: 73.6) | not slower |
| aggregate tok/s, 4 streams | 75.2 | **122.9** (x1.63) | target about 140: not met |
| aggregate tok/s, 16 streams | 75.2 | **166.5** (x2.21) | |
| decode row r of a B-row step vs the solo step of that sequence | n/a (one sequence) | **bitwise identical**, B = 2 to 16, 130 steps | 35B, Qwen3-8B, tiny hybrid |
| `plan show` + 256 greedy ids vs main, batching off | | **identical** | 35B and Qwen3-8B on CUDA, Qwen3-8B on Metal |
| `--spec mtp` k=2, 256 ids vs main and vs `--spec off` | | **identical**; decode +6 to +8% | 35B on CUDA, section 5 |

Cell: 1024-token prompt, 128 generated tokens, localeval `speed --concurrency 1,4,16`, the same cell as the baseline run
`20261022-050829-adhoc-fx3-35b-intarsia-p1-conc`. Exactness is **decode-exact within the batch path**. Prefill of a busy engine
runs in smaller chunks than a solo prefill, so for a prompt longer than one busy chunk it is **not bitwise the solo prefill**
(the GEMM tiling depends on the chunk); that is a KL question, not a bitwise one, and no KL gate was run in this lane (section 9).

## 1. What a batched step is

![One batched decode step](2026-10-23-rb2-cuda-batching/step.svg)

Shared weights are read once for all B rows by row-exact multi-row GEMVs (G-B). Everything that owns state (the GatedDeltaNet
recurrent state, the KV cache) runs once per row with the solo step's own kernels. The solo step (B = 1) is untouched and keeps
its fusions; at B >= 2 the fused ops run as their unfused chains, allowed only where the fusion has a registered bitwise
equivalence test (`crates/intarsia-core/src/fusion.rs`). The fused path is never chosen by how many sequences are active.

## 2. R0: row-exact kernels (G-B)

G-B: row r of an M-row kernel is bitwise row 0 of the same kernel at M = 1, for every M up to the cap. New kernels, all registered
`.rows_exact()` (selected only by `exact_rows` calls; the ordinary registrations select exactly what they did, pinned by
`tests/rows_registry.rs`):

| kernel | M | cases per M | rows compared (all M) | inputs |
|---|---|---|---|---|
| `matmul_fp8_gemv_rows` | 1..16 | 132 | 17,952 | 11 shapes (every 35B fp8 shape), W8A8 and W8A16, wide-range / zero / saturating rows |
| `matmul_nvfp4_gemv_rows` | 1..16 | 32 | 4,352 | 8 shapes (shared expert, ...) |
| `matmul_bf16_gemv_rows` | 1..16 | 22 | 2,992 | router, GDN in_a / in_b |
| `logits_bf16_gemv_rows` | 1..16 | 21 | 2,856 | lm_head |
| `moe_experts_decode_v2_rows_bf16` | T = 1..16 | 16 | 2,176 | routings incl. repeated and invalid expert ids |
| `moe_experts_decode_v2_gelu_rows_bf16` | T = 1..16 | 8 | 1,088 | Gemma-style MoE |

Row-parallel ops (rmsnorm, rmsnorm_add, add, silu_glu, sigmoid_mul, gated_rmsnorm, embedding, argmax, moe_topk, moe_combine) are
checked row-independent at M = 16 (fleetq #4526, `cargo test -p intarsia-backend-cuda --release --test rows_exact`, 5 passed).
The fp8 rows kernel stages x in K-chunks of 512 x floor(16384 / M / 512), so every lane keeps the 1-row kernel's block order for any
K % 16 == 0, W8A8 or W8A16, in at most 32 KB of shared memory with no opt-in (the fp8-byte staging tried first capped K at 96 KiB
and left a W8A16 hole). The MoE rows kernel runs its down projection with the token as the fastest grid axis, so tokens that route
to the same expert run next to each other and read its weights out of L2 (Fable M4's effect; no pair sort was needed to stay
bitwise, and none was done). Negative controls (a different K partition for m >= 4 fails the fp8 test) were run by the R0 worker;
I did not rerun them. Micro-timing on the 2048 x 2048 fp8 shape, us: M=1 8.2, M=4 14.4, M=8 24.6, M=16 46.8 (`timing_gemv_rows`, #4526).
Other CUDA tests at this head: `cargo test -p intarsia-backend-cuda --release` all green (#4525), `intarsia kernels test --quick`
77 kernels, 534 cases, 0 rejected.

## 3. R1: plan, memory and engine plumbing

| finding | what changed | check |
|---|---|---|
| B1 kernel picks | `plan::check_batch_ops` compares the kernel the batched step picks per op with the solo step's picks, exempting only `rows_exact` registrations; then compares op multisets after expanding the fusions in `crate::fusion::FUSIONS` | `batch_check_catches_a_kernel_switch_under_one_op` (the MoE `decode_v2` -> grouped `moe_experts_bf16` switch: same `Op`, passes the old multiset rule, fails the new one), `batch_check_accepts_only_proven_fusions`, `fusions_name_existing_tests` |
| M3 fusion rule | six CUDA fusions registered with their bitwise-equivalence tests in `tests/fused_decode.rs` (pair, fp8 pair, nvfp4 glu, gated rmsnorm, gated rmsnorm + fp8 projection, rmsnorm_add + pair) | the unit test fails if a named test is renamed or deleted |
| m1 | `exact_rows` off around the GDN per-row conv/delta loops and the per-row out-projection fallback, as in `attn_block_batch` | CUDA tiny-hybrid and 35B identity runs |
| m2 | per-row prologue-fold decisions must agree, else an error instead of a silent unfused fallback | `attn_block_batch` |
| m4 | `enable_batch` reserves (rows - 1) x (live + snapshots) recurrent bytes from the KV budget; slots allocate their snapshot buffers when created | `memory_plan_counts_recurrent_state_and_snapshots_and_slots_preallocate_them` |
| m6 | the `StepOut` of a headless prefill pass is checked to be `None` | engine/batch.rs |
| cap | `Backend::batch_row_cap` (default 8, CUDA 16), `MAX_BATCH_CAP` 16 | `config_limits` |
| MTP, prefix store | `--max-batch N` with `--spec mtp` is refused before the weights are read; on non-cpu backends batched slots keep no snapshots and the scheduler never reuses a recurrent slot's prefix (the cpu reference keeps its 4 snapshots) | `cached_tokens == 0` in the engine tests |

I read the review's findings late (the first read of the file failed) and checked the claims I acted on against the code:
B1 (CUDA picks `moe_experts_decode_v2_bf16` only for `m <= 1`; confirmed by the selection pins), M1 (the fp8 and nvfp4 GEMVs
are row-exact by construction; confirmed by G-B passing, though the staging had to change), M3, m1, m2, m4, m6. Not acted on:
B2 (host sampling, device top-k/top-p: R4), m3 (b = 1 graph recapture when slots alternate: the 1-stream row shows no cost),
m5. m7 is covered for the BF16 arms (the CUDA tiny hybrid runs in bf16 with bf16 in_a / in_b), not for the MXFP4 arms.

## 4. R2: bitwise gates (G-C)

| gate | what | result | fleetq |
|---|---|---|---|
| tiny hybrid, CUDA, bf16, three variants (dense, MoE W8A16, MoE W8A8) | `Model::forward_batch` N = 2, 3, 4, 8, 16 x 130 steps; decode beside prefill chunks; `BatchedEngine` equals the single-sequence engine over 130+ tokens (2 to 16 slots, 2 x slots + 3 requests); memory plan | 7 of 7 pass; batched plan asserted to pick only `_rows` kernels and the solo step's own per-row kernels | #4507, #4508 |
| **35B NVFP4, CUDA** | `forward_batch`, mixed context lengths 3 to 400, greedy and seeded-sampled rows, N = 2, 3, 4, 8, 16, 130 steps each | **4,290 (row, step) pairs bitwise identical** (logits bit for bit, argmax token for token, state identical); decode beside 256-token prefill chunks bitwise, prefill state equal | #4511 |
| Qwen3-8B BF16, CUDA | same gate | **4,290 pairs bitwise identical** | #4512 |
| **35B engine** | 20 requests (6 prompts, every 4th seeded-sampled, 130 tokens) alone on `LocalEngine`, then through `BatchedEngine` with 4 and 16 slots: one at a time (N = 1), and all at once (queueing, overlap, slot reuse) | **every request token- and text-identical to its solo run**; N = 1 through the batched engine equals `LocalEngine` | #4513 |
| final, at the head code | the three gates above rerun after the last code change (busy chunk default, `--busy-chunk` unset) | 35B: 4,290 pairs identical and the engine test pass (#4532); tiny hybrid 7 of 7 and Qwen3-8B 4,290 pairs identical (#4536) | #4532, #4536 |

The 35B and engine gates are in `crates/intarsia-models/tests/batch_cuda_real.rs` (opt-in: `INTARSIA_QWEN36_DIR`, `INTARSIA_QWEN3_DIR`);
the tiny-hybrid file runs on CUDA with `INTARSIA_BATCH_TEST_BACKEND=cuda`. In the engine test the prompts are shorter than one chunk,
so busy and solo chunk schedules coincide and the whole output is compared.

## 5. No-op gates

| | result | fleetq |
|---|---|---|
| CUDA, batching off: `plan show` (JSON and text) and 256 greedy ids, Qwen3-8B and 35B, branch vs `origin/main` b745e978 | all six comparisons byte-identical (8,758 / 1,478 / 1,436 B and 14,659 / 3,729 / 1,401 B) | #4516 |
| Metal, Qwen3-8B (core and models code changed): `plan show` and 256 ids | identical (8,371 / 1,342 / 1,439 B) | #4518 |
| Metal, `batch_metal` (dense batched identity, real Qwen3-8B, the plan check changed) | 138 pairs and 48 rows beside prefill bitwise identical | #4519 |
| **`--spec mtp --spec-k 2` on CUDA** (registering `_rows` kernels moved the verify pass, M = k + 1, onto them: fp8 `mma` -> `gemv_rows`, grouped `moe_experts_bf16` -> `decode_v2_rows`), 4 chat prompts, 256 greedy ids, 35B | main and branch ids identical, and identical to `--spec off`, for all four; acceptance identical (92.7 / 70.1 / 76.6 / 91.1%); decode 113.8 / 93.6 / 101.1 / 111.5 tok/s on main against **120.9 / 101.3 / 108.7 / 119.3** on the branch (+6 to +8%, one run each, noise about 5%); `INTARSIA_MTP_VERIFY_CHECK`: 647 verify-row cells over 163 passes, max logit difference to the single-row decode **0.0** | #4537 |
| `cargo test --workspace` (endor) | all pass, 0 failed | #4524 |

## 6. Speed (localeval, intarsia serve on spark1, 35B NVFP4, 1024-token prompt, 128 generated, 3 reps / one run per stream count)

| run | 1 stream | 4 streams | 16 streams | run id (unanimis) |
|---|---|---|---|---|
| baseline (one sequence at a time) | 74.2 | 75.2 | 75.2 | `20261022-050829-adhoc-fx3-35b-intarsia-p1-conc` |
| this branch, batching off (no `--max-batch`) | 73.6 | 74.1 | 74.2 | `20261023-084926-adhoc-rb2-off` (e59997bc), #4521 |
| `--max-batch 16`, busy chunk 256 | 73.4 | 104.1 | 128.3 | `20261023-085311-adhoc-rb2-on16` (5b478879), #4522 |
| `--max-batch 16`, `--busy-chunk 1024` | 73.4 | 122.4 | 166.8 | `20261023-091032-adhoc-rb2-on16-bc1024`, #4527 |
| **`--max-batch 16`, default (busy chunk 1024 for hybrid MoE on CUDA)** | **73.6** | **122.9** | **166.5** | `20261023-092106-adhoc-rb2-final2-on16` (7bc2abc9), #4531 |

Single-stream decode (per-stream decode tok/s, 3 reps): 83.7 to 84.7 with batching off and on; the aggregate differences are inside the
quoted decode noise of about 5%. Aggregate is completion tokens over the wall time of the whole burst, prefill included. Server flags:
`serve --backend cuda --served-model-name intarsia --max-seq 8192 [--max-batch 16]`.

**Why not 140.** The step itself is on target; the burst is not. In the model loop (`real_35b_batch_step_timing`, #4523, ignored test,
eager, all rows greedy, 900-token contexts):

| B | ms per step | aggregate tok/s | x the eager solo step (19.8 ms) |
|---|---|---|---|
| 2 | 18.1 | 110 | 0.92 |
| 4 | 25.4 | 158 | 1.28 |
| 8 | 38.6 | 207 | 1.95 |
| 16 | 65.3 | 245 | 3.30 |

The burst at 4 streams (final run, `20261023-092106-adhoc-rb2-final2-on16`) takes 512 / 122.9 = 4.17 s: the four 1k-token prefills run one
after another (0.22 s each; the last stream's first token arrives at 0.90 s), then all four decode, and (4.17 - 0.90) / 127 steps =
25.7 ms per step, which is the model-loop step (25.4 ms, table above). So at 4 streams **the step already runs at model-loop speed and
the gap to 140 is 0.9 s of serialized prefill inside a 128-token burst plus the step itself**: 140 tok/s needs a 3.66 s burst, which is
either a step of about 21.7 ms (15% cheaper, with the prefill time unchanged) or less prefill time, and prefill (4.4k tok/s) is compute-bound, so
it can only be hidden by overlapping it with decode, not shortened by batching. At 16 streams the same split is 12.3 s = 3.93 s of prefill phase
plus 127 steps of 65.9 ms (model loop: 65.3 ms), so the step matters more there. (With the busy chunk at 256 the same split gives
4.92 s and 31 ms steps: the earlier 104 tok/s; the stalls of small chunks, not the step, were what the chunk change removed.) The
cost inside the step, from the profiled step (`real_35b_batch_step_profile`, op time per step, profiling syncs after each op so the sum is
above the real step):

| op | B = 4 ms | B = 16 ms |
|---|---|---|
| moe_experts (rows kernel) | 7.8 | 17.8 |
| matmul_fp8 | 7.2 | 14.0 |
| gdn_delta_rule + gdn_conv (per row) | 3.9 | 17.5 |
| matmul_nvfp4 | 2.4 | 5.0 |
| rope + qk_norm + kv_append + attn_decode (per row) | 3.0 | 12.1 |
| matmul (bf16) | 1.5 | 2.6 |
| sum | 27.6 | 70.9 |

The per-row GDN and attention launches are 25% of the step at B = 4 and 42% at B = 16, and the batched step is eager (no CUDA
graph; the solo step replays one). Halving the per-row part would take about 12% off a B = 4 step (short of the 15% that 140 needs by the step
alone) and about 21% off a B = 16 step; with prefill overlap or a graph on top, 140 at 4 streams is in reach. That is review R3 / R5 (a multi-state
GDN kernel taking an array of state pointers, per-row attention tables, a graph per batch bucket) and prefill/decode overlap; none of it is in R0 to R2. The expert reads are bandwidth-bound
(about 14 MB per token per layer when routing does not overlap).

## 7. Decisions

All in `docs/decisions/LOG.md` (2026-10-23 06:41 to 09:20): cap 16 on CUDA per backend; decode row-exact within the batch path and
M-dependent kernels forbidden; solo keeps its fusions, B >= 2 runs unfused only under a registered equivalence; MTP and the prefix
store stay at batch 1; `_rows` kernel design; the two-rule plan check; busy chunk 1024 for hybrid MoE on CUDA; the Spark free-memory
finding.

## 8. Reproduce

```
# G-B (spark1, quiet slot)
cargo test -p intarsia-backend-cuda --release --test rows_exact --test rows_registry -- --test-threads=1
# G-C, tiny hybrid and real models
INTARSIA_BATCH_TEST_BACKEND=cuda INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_recurrent -- --test-threads=1
INTARSIA_QWEN36_DIR=<35B dir> INTARSIA_QWEN3_DIR=<8B dir> INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real -- --test-threads=1
# speed: deploy/localeval/serve-and-bench.sh --served-model-name intarsia --max-seq 8192 --max-batch 16 -- \
#   uv run --project /opt/code/deepeval localeval speed --url http://spark-4e77:8290/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 --concurrency 1,4,16
```

## 9. What this does not show

- **Busy-chunk prefill vs solo-chunk prefill is not bitwise** for a prompt longer than the busy chunk (1024 for the 35B on CUDA, 256
  elsewhere), because GEMM tiling depends on the chunk. The gates above use prompts of one chunk or identical chunk schedules. No KL
  gate of busy against solo prefill was run here.
- Only greedy and seeded top-k/top-p rows were compared; host sampling cost at 16 sampled rows (review B2) was not measured.
- Memory: on the Spark the CUDA `free` figure after loading the 35B was 1.9 GB while 60 GB sat in the page cache, so a 16-slot engine
  planned about 1.5k tokens of KV per slot. The speed runs and tests evicted other models' cached files first
  (`dd iflag=nocache count=0`); a plain `intarsia serve --max-batch 16` on a box with a warm cache gets a short context or a memory
  error. Fixing it in the loader would change `plan show`, so it is left as an open question.
- MTP and the prefix store are refused or bypassed at batch > 1, not implemented (MTP at batch 1 on CUDA is checked in section 5; the `--spec mtp` startup note now says the verify rows run row-exact kernels). MXFP4 and Gemma arms are untouched and refused.
- Runs were on spark1 with the shared vLLM tenant resident (34 GB) in `--quiet gpu,cpu` slots; no comparison against vLLM or SGLang was repeated.
- The R0 worker's negative controls and the registry-pin tests were not rerun by me.

## 10. Jobs

fleetq (owner `claude-f31ccf91-rb2`): G-B and CUDA suite #4525, #4526; G-C #4507, #4508, #4511, #4512, #4513, #4532, #4536 (#4509 failed: the
free-memory finding, section 9); no-op #4516, #4518, #4519, #4524; speed #4521, #4522, #4527, #4529 (default flag not yet applied: 256), #4531;
timing and profile #4523; builds #4504, #4514, #4515, #4517, #4528, #4530. #4533 was my own client killed by PID after a script mistake; its
two orphaned remote test processes were stopped by PID (#4534, #4535). Nothing is held.
