# HIP norm into GEMV: Qwen3.6-35B-A3B NVFP4 decode on one R9700 (branch perf/hip-norm-gemv)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main as PR #5 (`df18c6a5`); "not pushed" / "nothing merged" wording in this page describes the state when written.

Base `dbdd40e` (`perf/hip-fp8-gemv`, the fp8 pair kernel; see `2026-10-17-hip-fp8-gemv.md`), tatooine R9700 #0 (gfx1201, ROCm 7.2.4 dev
image, `HIP_VISIBLE_DEVICES=0`, graphs on, `--profile-every 0`). Both trees are rsynced without `.git` into their own directory and
cargo target volume (`~/intarsiaml-normgemv`, `~/intarsiaml-normgemv-base`), so binaries report `unstamped`. Scripts:
`scripts/hip-decode/` (`build.sh`, `burst.sh`, `remote.sh`, `mutate.py`, `analyze.py`); raw logs of every burst, the profile tables,
the bench JSONs and the qualify reports are in `docs/results/2026-10-17-hip-norm-gemv/`.

## 1. Result

Two fused decode ops fold a norm into the first launch that reads it, in the prologue of the GEMV: every block recomputes the
row's norm itself (the hidden row is 2048 bf16, 4 KB, read from L2) in the order of the standalone kernel, so both ops are
**bitwise** the chains they replace. 99 of the 677 launches of a decode step go away.

| per decode step (rocprofv3 trace, 157 steady steps of a 160-token run) | launches | kernel time | step wall | idle inside step |
|---|---|---|---|---|
| base `dbdd40e` | 677 | 6.287 ms | 8.418 ms | 2.131 ms (25.3 %) |
| + norm into the router / b+a pairs (`profile-norm1.txt`) | 608 | 6.272 ms | 8.204 ms | 1.932 ms |
| **+ gated norm into out_proj (this branch, `profile-norm2.txt`)** | **578** | **6.237 ms** | **8.068 ms** | **1.830 ms (22.7 %)** |
| `INTARSIA_NO_FUSION=1`, base and branch | 897 and 897 | 6.624 / 6.621 ms | 9.413 / 9.420 ms | |

End to end, `bench speed --sizes 128,1024,4096 --gen 128 --reps 5`, `intarsia serve` per run, interleaved A1 B1 A2 B2 (A = the
`dbdd40e` build, B = this branch; `remote.sh bench35@base:A1 bench35:B1 bench35@base:A2 bench35:B2`), decode tok/s (prefill tok/s
in brackets):

| 35B | 128 | 1024 | 4096 |
|---|---|---|---|
| A1 base | 136.7 (1168) | 135.9 (2118) | 132.7 (2038) |
| B1 branch | 140.0 (1184) | 139.2 (2037) | 135.1 (2081) |
| A2 base | 136.4 (1241) | 135.8 (2064) | 132.2 (2078) |
| B2 branch | 139.9 (1153) | 139.2 (2105) | 135.7 (2050) |
| **base mean** | **136.6 (1205)** | **135.9 (2091)** | **132.4 (2058)** |
| **branch mean** | **140.0 (1168)** | **139.2 (2071)** | **135.4 (2065)** |
| decode change | **+2.5 %** | **+2.5 %** | **+2.3 %** |

The in-run spread of the decode figure is 0.02 to 0.26 tok/s, so the gain is far outside it and repeats in both rounds. Prefill is
not touched (the fused kernels exist for one decode row only; the 128-token column varies 1153 to 1241 between identical runs).

| 8B (dense bf16, no GDN, no MoE: not touched) | 128 | 1024 | 4096 |
|---|---|---|---|
| A1 / A2 base decode | 38.0 / 38.0 | 37.4 / 37.4 | 36.2 / 36.2 |
| B1 / B2 branch decode | 38.0 / 38.0 | 37.4 / 37.4 | 36.2 / 36.2 |
| base prefill (A1, A2) | 2059, 2104 | 3405, 3418 | 3080, 3076 |
| branch prefill (B1, B2) | 2079, 2082 | 3409, 3409 | 3087, 2994 |

`intarsia qualify --quick-only` on this branch (`qualify-*-quick-report.md`): **35B PASS** (7 of 7 stages: 40 kernels, 639 cases
admitted; acc-short prefill top-1 98.54 % KL 6.56e-3, decode 99.38 % KL 9.96e-3; bench 140.1 tok/s at 128 tokens), **8B PASS**
(40 kernels, 652 cases; acc-short prefill 100 % KL 2.34e-4, decode 100 % KL 3.63e-4; bench 37.9 tok/s at 128 tokens, 92 % of its
byte floor).

**The gain is 2.5 %, not the 4 to 5 % the previous page estimated.** The estimate used the trace's 3.0 us gap between launches. The
trace itself stretches the step (8.4 ms traced against 7.3 ms in the unprofiled run: 136.6 tok/s), and the bench says what a
removed launch is worth outside it: 7.321 ms -> 7.143 ms per token is 178 us for 99 launches, **1.8 us each**. The traced step
shrinks by 350 us, 3.5 us per launch, which is not the number to quote for end-to-end gain.

## 2. The adjacencies

Every residual-add + norm in the decode step, with the launches that read its output (35B, one decode token, 677 launches before):

| norm (launches / step) | first consumer in program order | other readers of the normed row | fused |
|---|---|---|---|
| `rmsnorm_add` before the MoE block (40) | bf16 pair router (N 256) + shared-expert gate (N 1), `gemv_pair_kernel`, 33 blocks, 2.5 us | MoE experts gate_up, shared glu (nvfp4), shared-gate logit | **yes, 40** into that pair |
| `rmsnorm_add` before a GDN block (29) + plain `rmsnorm` at layer 0 (1) | fp8 pair qkv + z (12288 x 2048, 1536 blocks, 44 us); then bf16 pair b + a (N 32 + 32, 8 blocks, 2.5 us) | the other of the two pairs | **yes, 29** into the b + a pair, which now runs first (the gates do not depend on qkv / z) |
| `rmsnorm_add` before an attention block (10) | fp8 q (8192 x 2048, 512 blocks, 31 us); then fp8 k + v pair (128 blocks, 5.7 us) | the other | no |
| final `rmsnorm_add` before lm_head (1) | nvfp4 lm_head, f32 logits, 460 us | none | no |
| `gated_rmsnorm` after the GDN delta rule (30) | fp8 out_proj (2048 x 4096, 128 blocks, 17 us) | none | **yes, 30** into out_proj |
| `qk_norm` (20) | rope, not a GEMV | | no |

Why those: the router + gate pair and the b + a pair are tiny (8 to 33 blocks), so every block recomputing a 2048-element norm
adds almost nothing, whereas folding into the 1536-block qkv + z pair would repeat the norm 1536 times for no gain (the pair
that runs first is not the one that should carry the norm). A norm that feeds several GEMVs needs one designated block to store
the normed row for the others and the new residual stream. The gated norm has exactly one reader (`out_proj`), which stages its
activations in LDS block by block anyway: the norm replaces the global load of that staging.

## 3. What was built

* **Ops** (`crates/intarsia-core`): `Op::RmsNormAddMatmulPair` (x, res, norm w, w0, w1 -> normed, sum, out0, out1 = `RmsNormAdd`
  then `MatmulPair` of the normed row) and `Op::GatedRmsnormMatmulFp8` (core, z, norm w, fp8 w, scales -> out = `MatmulFp8` of the
  `GatedRmsnorm` rows read as one row). Defined as exactly those compositions (`fused_ref.rs`, run by the cpu `ref_*` and `fast`
  kernels), admission cases (`quant/cases.rs`, in the `kernels test` sweep: 40 kernels, 639 cases on the 35B). Both are optional
  like the other fused ops: a backend without a kernel (CUDA, Metal, the cpu fast path), tensor parallelism, `INTARSIA_NO_FUSION=1`,
  or a shape the kernel predicate refuses leaves the unfused chain, and `fused_ok` / `Registry::select` is the only gate.
* **Model**: there is no graph rewrite pass in this repo (checked: nothing like it in `intarsia-core` or `intarsia-models`; the CUDA
  fusions of `2026-10-14-cuda-decode-fusion.md` are `try_fused` calls in the block code as well). A norm that precedes a block is
  handed to it as a `NormFold` (`quant/moe.rs`): `MoeWeights::forward_folded` and `gdn_block` try the fused op at their first pair and
  otherwise run `RmsNormAdd` themselves before anything reads the normed row. `run_layers` flips the residual buffers exactly as
  before. Attention, FFN (the 8B) and the layer-0 plain norm are unchanged.
* **Kernels** (`kernels/hip/matmul.hip`, `fp8.hip`): `rmsnorm_add_matmul_pair_bf16_gemv` is `rmsnorm_kernel<true, true>` statement
  for statement in each block's prologue (same 8-element chunks per thread, bf16 rounding of the sum before it is squared, same
  `ik_block_sum` over 256 threads: the block size the standalone launcher picks for any K >= 2048, and for a narrower row the extra
  threads add exact zeros), the normalised row in LDS (K <= 8192), then `gemv_body<1, 1, 8>` reading x from LDS; block 0 alone stores
  `sum` and `normed` (different buffers from the `x` / `res` every block reads; the launcher refuses overlapping ranges). The first
  batch of weight loads is issued before the prologue, so their DRAM latency hides behind the norm. `gated_rmsnorm_matmul_fp8_gemv`
  computes each 8-element chunk of the gated norm in the staging loop of `fp8_gemv_body` (`PRO == 2`): the 16 chunks of a head are
  16 lanes, and the xor butterfly starting at offset 8 sums them in the order the 32-lane reduction of `gated_rmsnorm_kernel` does
  (its first step adds exact zeros); `INTARSIA_HIP_FP8_ROWS` selects the rows per warp as for the single v2 GEMV.
  Registered in `quantmoe.rs` (priority 10, deterministic, `TOL`), dispatched in `device.rs`, declared in `kernels/abi/ik_abi.h`.
* Per kernel, from the trace: the fused pair is 4.0 us (69 launches) against 2.5 us + 2.1 us for the pair and the norm; the gated
  out_proj is 19.4 us (30) against 17.1 us + 2.2 us. So the kernel time is level or a little lower (6.287 -> 6.237 ms) and the whole
  saving is the launches.
* Scripts: `remote.sh` gets a `nofusion` stage and `PROF_PRE` / `GEN_ENV` hooks (the container forwards only `INTARSIA_HIP_*`
  variables, so the stage sets `INTARSIA_NO_FUSION` with `env` inside it), 13 new mutations in `mutate.py` and the default list of
  `remote.sh mutate`, `analyze.py` knows the new fp8 kernel.

## 4. Accuracy and proofs

All outputs of the fused kernels are bit-identical to the unfused launches, so the gates are unmoved by construction; measured:

| check | gate / base value | this branch |
|---|---|---|
| `accuracy --backend hip --mode both`, 35B, golden `qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors` (sha `b6b51e1b1699c66c`) | prefill top-1 98.40 %, KL21 7.071e-3; decode 98.75 %, 1.242e-2 | **identical**: prefill 98.40 %, 7.071e-3; decode 98.75 %, 1.242e-2; non-finite 0; golden PASS |
| qualify `acc-short` (35B, short golden) | none recorded | prefill 98.54 % / 6.56e-3; decode 99.38 % / 9.96e-3; PASS |
| `qwen3-8b-fp32` golden | gate: top-1 >= 99.5 %, KL < 1e-3 | prefill 100 % / 2.335e-4; decode 100 % / 3.626e-4; PASS |
| `qwen3-8b-long-fp32` golden | same gate | prefill 99.58 % / 3.966e-4; decode 100 % / 3.573e-4; PASS |

The 8B model has no GDN or MoE block, so neither new op can run there; its numbers are a check that nothing else moved, and I did not
compare them against a base run of the same goldens.

* **Graph vs eager**, 35B, 256 greedy tokens (`remote.sh graphproof`): the 256 ids are identical (255 commas in each file; graph
  140.18 tok/s with 256 replays and 1 capture, eager 136.96 tok/s with 0 replays). The 578 launches of the step are one captured
  graph (`steps=257 graph replays=256 (captures 1)`).
* **`INTARSIA_NO_FUSION=1`** (`remote.sh nofusion`): the 256 ids equal the fused run's (`gen35-nofusion.ids`, `cmp` clean against
  `gen35-graph.ids` and `gen35-eager.ids`), and the step is **897 launches in both trees**, the base's own unfused launch count, with
  neither new kernel in the trace (`profile-nofusion.txt`, `profile-nofusion-base.txt`).
* **Kernel tests** (`crates/intarsia-backend-hip/tests/fused_decode.rs`, bitwise against the launches they replace, all four
  outputs of the norm + pair op): `rmsnorm_add_matmul_pair_is_bitwise_norm_then_pair` (10 shapes x `w_offset` 0 / 1: the model's
  2048 with 32 + 32 and 256 + 1, a row narrower than the block, K = 8, a segment shorter than a block, K = 4104 and 8192 where a
  thread holds several vectors, a 4096-block grid), `gated_rmsnorm_matmul_fp8_is_bitwise_norm_then_gemv` (9 shapes x W8A8 / W8A16 x
  `w_offset`), and two many-draw tests (2500 re-drawn activations each) for the order of the sum of squares. The cpu reference
  (`near_reference`) is checked too. On the tatooine tree: `fused_decode` 8/8 (was 6), `fp8_pair_shapes` 1/1, `decode_v2` 5/5,
  `admission` 6/6.
* **Mutations** (`mutate.py`, `remote.sh mutate`, each on a copy of the tree; the intended `fused_decode` test must fail):

| mutation | what it breaks | caught by |
|---|---|---|
| `norm-skip-sum`, `norm-skip-normed` | the designated block skips its store of the new residual stream / of the normed row | both norm + pair tests |
| `norm-sum-round` | the sum is squared unrounded | both norm + pair tests |
| `norm-woff` | the zero-centred `+1` on the weight is dropped | both norm + pair tests |
| `norm-pair-block` | the second matrix's blocks start one block late | the shape test |
| `norm-sync` | no barrier between writing and reading the LDS row | the shape test (a race: caught on the one run made) |
| `norm-order` | the 8 squares of a chunk are added in the opposite order | the 2500-draw test only |
| `gn-round`, `gn-woff`, `gn-group`, `gn-weight`, `gn-silu` | no bf16 rounding of the normalised value; no `+1`; two heads share one sum; wrong weight chunk; silu on the wrong operand | both gated tests |
| `gn-order` | the head's sum of squares is added in the opposite order | the 2500-draw test only |
| `fp8-pair-scale`, `pair-block`, `fp8-scale`, `fp8-convert` | the earlier mutations of code this change shares (rerun on this tree) | as on the previous page |

  **17 of 17 fail the intended test.** Two of them survived the first version of the test: `norm-order` and `gn-order` change
  the last bit of the sum of squares, which flips a bf16 output element about once in 2^16, and 160 and 120 draws did not show it
  (the first assumption, "an order mutation is visible in a bitwise test", was wrong). The tests now re-draw 2500 activations
  against fixed weights (5 s) and both mutations fail them; the first `norm-order` (swapping neighbouring lanes before the
  butterfly) was an equivalent mutant, because the xor tree is invariant under that permutation, and was replaced by the reversed
  chunk order.
* After the mutation stage every source was touched and the tree rebuilt (27 s, `sync_build.sh`-style: `find kernels crates -type f
  -exec touch {} +` on tatooine, then `build.sh`), and the clean tree was re-tested (`fused_decode` 8/8) before any later number was
  taken. The rsync of the next sync would have restored the Mac mtimes, older than the mutated builds in the shared cargo volume, and
  cargo would have kept the mutated binaries: the step is needed, not optional.
* Mac: `INTARSIA_CPU_FREE_MEM_GB=8 cargo test --workspace` (83 test binaries, 0 failures), `cargo check --workspace`,
  `cargo check -p intarsia-cli --features hip`: all pass. The CUDA crate compiles (its Rust side; the `.cu` side is untouched and
  was not compiled here): its kernel-coverage test lists the two new ops as HIP-only, like `QkNormRopeKv` for Metal.

## 5. Radiance downtime

Every GPU burst: `fleetq acquire --host tatooine --takes-down radiance --mem 30 --wait`, `FLEETQ_JOB=<id>` on every ssh, `burst.sh`
(stop, stages, restart, poll `/v1/models`), `fleetq release` straight after it printed "radiance answers". Builds, rsyncs and result
pulls ran in `--mem 0` slots (no radiance impact). Times are tatooine's clock.

| job | what | stopped | answered | down |
|---|---|---|---|---|
| 2313 | fused tests, profile of the first fusion, base profile | 12:37:15 | 12:38:20 | 1:05 |
| 2315 | fused tests with the gated kernel, profile | 12:42:43 | 12:43:37 | 0:54 |
| 2319 | 35B + 8B accuracy, graph proof, NO_FUSION proof | 12:45:09 | 12:46:39 | 1:30 |
| 2320 | mutations A (9) | 12:47:03 | 12:50:59 | 3:56 |
| 2322 | mutations B (9) | 12:51:32 | 12:55:27 | 3:55 |
| 2325 | clean-tree test, order mutations | 12:57:25 | 12:59:05 | 1:40 |
| 2327 | 35B and 8B A/B bench, 2 interleaved rounds | 13:00:00 | 13:10:55 | 10:55 |
| 2329 | `qualify --quick-only` 35B + 8B | 13:11:43 | 13:17:18 | 5:35 |

Total 29:30 against the 60 min cap, longest burst 10:55 against 15 min. `/v1/models` answered 15 s after each restart. No `--force`
or `--no-queue`; no lease held while coding.

## 6. Commits and what is not done

Commits on `perf/hip-norm-gemv` (from `dbdd40e`): the code, tests and scripts, then this page and its `2026-10-17-hip-norm-gemv/`
data. Not pushed.

* Not fused: the attention norm into the fp8 q or k + v launches (10 launches, about 18 us per step at 1.8 us each: needs a
  rmsnorm_add prologue in `fp8_gemv_body` and the same designated-writer rule), the final norm into lm_head (1), the qkv + z fp8 pair
  (a norm in 1536 blocks, no gain), the 8B's FFN pair (36 launches, a 3072-block grid at H = 4096: would only add work), and anything
  under tensor parallelism (fusion is off there as for the other fused ops).
* The end-to-end A/B measures both fusions together; the norm1-only point exists only as a profile (608 launches, 8.204 ms traced,
  `profile-norm1.txt`) and as one 160-token unprofiled run (136.4 tok/s, taken right after the radiance stop, so not comparable with
  the interleaved numbers). The split of the 2.5 % between the 69 pair launches and the 30 out_proj launches is therefore not measured
  beyond the traces: kernel time per fused launch is level with the parts it replaces in both, and the gap saving is per launch.
* `norm-sync` was caught on the single run made; a missing barrier is a race and a pass on another run is possible, so that one
  mutation is evidence, not proof. The remaining 16 are deterministic.
* The kernels are written for hidden 2048 (the 35B); the predicates admit any K % 8 == 0, K <= 8192 for the norm + pair and
  head dim 128, K * 2 <= 48 KB for the gated one, and the tests cover the other shapes, but only the 35B was run end to end.
* Single-GPU numbers only; TP = 2 was not run.
