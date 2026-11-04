# DSv4.1 CUDA prefill: interleaved softmax (E5) and in-place q / o rope (E3) wired into the engine

Date 2026-11-04 (early morning EST), branch `dsv41/attn-wire` (worktree `.worktrees/dsv41-attn-wire`, from `6f56a2c2d` plus `git merge origin/main` at local ref `3c57d50a2`). Local commits only: nothing is merged or pushed. Follows [the tfblock bench doc](2026-11-03-dsv41-attn-tfblock.md) (fleetq #8282: E2 negative, E5 and E3 `ip` bench-proven at kernel level). E2 (`ATTN_TFBLOCK`, `attn_sparse_mma_tfb`) is not brought in and nothing of it is left in the tree.

## Status

Both changes are in and bit-identical through the engine on the 3-layer cache on spark2; no throughput was measured in the engine (the gain is the bench's, extrapolated). What did not run is listed under "Still needs a window".

| check | result |
|---|---|
| `cargo test --workspace --no-fail-fast` on endor, final run on the last commit (fleetq "workspace (final)") | 1231 passed, 0 failed, 29 ignored |
| first run (#8344), before the last test edits | 1229 passed, 2 failed, 29 ignored; both failures were mine and are fixed (below) |
| CI cuda-cfg check (`INTARSIA_CUDA_CHECK_ONLY=1 cargo check --locked -p intarsia-backend-cuda --all-targets` and `-p intarsia-cli --features cuda`, #8346) | pass |
| spark2, one job (#8347, 05:24 to 05:37 EST, `--quiet gpu,cpu --mem 20 --expect 60`): nvcc build of the whole tree | pass, 5 m 12 s for the CLI at `MAX_JOBS=4` |
| `dsv41_dpos` (CUDA): in-place rope bitwise equal to the copy rope, host and device position | 5 passed |
| `dsv41` (CUDA) kernels against the cpu reference | 4 passed, 1 ignored (bench) |
| `intarsia kernels test --backend cuda --quick` | 106 kernels, 918 cases, 0 fail (`rope_tail` 12 cases, `attn_sparse` 18 cases, all ADMITTED) |
| engine output hash, 3-layer cache, CUDA: `device_position_passes_equal_host_passes` | equal across the head build with `ROPE_INPLACE=1`, with `=0`, and with main's attention kernel (see below) |

`--expect` was 60, not the 30 in the brief: `build.rs` recompiles every `.cu` when any kernel file changes, and phase B (below) is a second full nvcc pass. The job took 13 minutes.

## What changed

- `kernels/cuda/dsv41_attn.cu` (E5): `softmax_rows4` is the softmax of `attn_sparse_mma`, unconditional. The `ATTN_SM4` `#ifdef` and the per-row loop are gone. Dependents of the old form: none. The SIMT `attn_sparse` kernel (`ik_warp_sum` / `ik_warp_max`, one row at a time) has its own softmax and is untouched; `attn_sparse_mma` is the only user of `softmax_rows4`. No `ATTN_TFBLOCK`, `attn_sparse_mma_tfb`, `kTf*` or `attn_mma_smem_tfb` remain.
- `kernels/cuda/dsv41_misc.cu` (E3): `rope_tail_ip_kernel` (one thread per (re, im) pair, rope dims only, per-element arithmetic of `rope_tail_kernel` verbatim), taken in `ik_rope_tail` when `x->data == out->data && x->dtype == out->dtype`. The `ROPE_INPLACE_SHFL` form is not brought in. Unconditional (no `#ifdef`), but only taken when a caller aliases.
- `crates/intarsia-core/src/dsv41/graph.rs`: `GraphTp::rope_inplace` (default false) and `rope_inplace_for(backend)`: true only for the backend named `cuda`, unless `INTARSIA_DSV41_ROPE_INPLACE=0`. With it the `rope(q)` and `rope_inverse(o)` of `attention()` and `dspark_attention()` write `ws:q` / `ws:o` instead of `ws:q_rope` / `ws:o_derot`. The other ropes (`kv`, `index_q`, `index_k`, `latent`) are untouched.
- `Dsv41Runtime::new` sets `opts.tp.rope_inplace` from its backend's name; `dsv41::plan::plan_json` does the same, so `plan show` describes what the CUDA runtime runs. The cpu reference keeps separate buffers (it would work in place too, `dsv41_ref.rs` reads the whole input first, but that was not tested and the brief says to leave other backends alone). HIP and Metal do not run this graph at all (`UNIMPLEMENTED`, no DSv4.1 kernels), so their behaviour is unchanged.
- Docs: `docs/tuning.md` row for `INTARSIA_DSV41_ROPE_INPLACE` (the env-var doc test requires it), `ops.rs` table row, the "no step's output aliases its input" comment in `graph.rs`, the CUDA row of "Defaults in force" and a `LOG.md` row.

## The unrotated-read question (E3)

Question: does anything read the unrotated `q` or `o` after the aliasing? Answer: no. Checked by reading, and then by a test that fails if it changes.

- `ws:q` is written by one step per attention, the `wq_b` matmul (lane 0). Its only reader before the rope is the rope itself (`graph.rs`, `attention()` and `dspark_attention()`: the `Arg` returned by `b.matmul` is passed to `b.rope` and never used again). `qr`, the input of `wq_b`, is what the indexer's `wq_b` reads, not `q`. After the rope the first and only reader is `AttnSparse` input 0, in the same layer.
- `ws:o` is written by `AttnSparse` only. Its only reader before the rope is the rope (`rope_inverse(o)`); the rope's output (now the same buffer) goes to `act_quant(o)` (FP8 checkpoints only) or to `wo_a` (`Transpose01` group-major on the EXL3 checkpoint, a grouped matmul on the dense one). Nothing else names `o`.
- The names `"q"`, `"o"`, `"q_rope"`, `"o_derot"` occur nowhere else in `crates/`, `src/`, `tools/` or `scripts/` as buffer names. `INTARSIA_DSV41_DUMP` is keyed by step name and writes each op's outputs after it ran, so a dump of `rope(q)` is the same bytes as before. `scripts/dsv41-tp2/densecrit.py` matches the step name `rope_tail`, not the buffer. The workspace is allocated per `Buf::Ws` name from the largest shape that uses the name; `q` and `q_rope` have the same shape, so sizes are unchanged and the `q_rope` / `o_derot` buffers simply stop existing on CUDA.
- Rank 1 and TP: the heads are `n_heads / world` per rank and the ropes are local; the only collectives of a layer are on the `wo_b` output (`blk`) and the MoE partial, neither of which names `q` or `o`. Split attention (`nsplit > 1`, `attn_sparse_merge`): the partials go to scratch and only the merge writes `out`; `q` is read by the split kernel, rotated. The SIMT `attn_sparse` reads `q` the same way. Nothing in the world-1 header fixture differs from a rank's graph in these steps; TP=2 itself was not run (it needs a window), so that part rests on this reading.
- Lanes: the producer and the rope are lane 0 steps; lane forks join before the next lane-0 step, so no side stream reads `q` or `o`.
- Replay: in-place rotation is not idempotent. Every pass (eager, device-position, CUDA-graph replay, verify with rollback, draft) runs `wq_b` / `AttnSparse` before its rope, so the rope always sees a fresh input. A replay of the rope alone would double-rotate; no code path does that, and the engine hashes below include 25 draft, 23 verify and 31 decode graph replays.
- `dpos`: `rope_tail_ip_kernel` has the same `dpos ? *dpos / ratio * ratio : pos0` line as the copy kernel. The bench did not exercise it. `dsv41_dpos` now does (see below) and so does the engine run with replayed CUDA graphs.
- The test `in_place_rope_leaves_no_reader_of_the_unrotated_rows` (`crates/intarsia-models/tests/dsv41.rs`) builds the graph with and without the flag for decode, 4, 16, 512 and 2048 rows, the device-position verify / absorb pass graphs and the draft graphs, and asserts: same steps, lanes, positions, extras, shapes and dtypes; the only changed buffers are the two ropes' outputs and their next readers; the producer of the buffer is `wq_b` (or `AttnSparse`) on lane 0; no step between the producer and the rope touches it; the first step after the rope is `AttnSparse` (for `q`) or `act_quant(o)` / `wo_a` (for `o`); every step still passes `phantom_call()` (`OpCall::check`, so no generic alias rejection); the graph without the flag has no step that aliases an input.

## Checks

### Kernel level, on spark2 (CUDA)

- `crates/intarsia-backend-cuda/tests/dsv41_dpos.rs` gained `check::rope_in_place`: the same rope launched with `x` as its own output, for bf16 and f32, T 1, 4, 5, ratio and stride (1, 1), (4, 4), (2, 2), forward and inverse, 8 positions including 4093 and 30000, head widths 512, 128 and 64, host position and device position, compared bitwise with the copy kernel's output (the whole row, so the unrotated 448 dims are checked to be left alone). Passed.
- `dsv41` against the cpu reference and `kernels test --quick` pass with E5 in the kernel (`attn_sparse` 18 cases, max_abs 3.9e-3, rel_l2 2.1e-3, as the tolerance expects; this sweep judges by tolerance, not bits).

### Engine, bit identity (spark2, CUDA, 3-layer cache `l3-r0of1`)

`device_position_passes_equal_host_passes` now prints an FNV-1a hash of every output bit (`RUNTIME output hash`). Same binary, `INTARSIA_DSV41_ROPE_INPLACE` 1 and 0, and a second binary whose `dsv41_attn.cu` is `origin/main`'s (that is E5 off). Everything else identical.

| run | ROPE_INPLACE=1 (head) | ROPE_INPLACE=0 (head) | ROPE_INPLACE=0, main's attention kernel |
|---|---|---|---|
| spec, replayed CUDA graphs, 140 prompt, 48 tokens, 52 passes, 5 042 063 values | `e5278faf840fd764` | `e5278faf840fd764` | `e5278faf840fd764` |
| host positions, 110 prompt, 48 tokens, 30 passes, 3 878 400 values | `b9622ff2274210b3` | `b9622ff2274210b3` | `b9622ff2274210b3` |

"Base" here is not a literal `origin/main` build: it is this branch's Rust with the alias switched off (main's graph, main's copy rope kernel path) and, in the third column, main's `dsv41_attn.cu` in place of E5. Building a second full tree was not worth a second hour. In each run the device-position passes are also bitwise equal to the host passes (the test's own assertion). `few_layer_runtime_matches_the_official_layer_outputs` on CUDA against the official fixtures prints the same nine error figures for all three binaries (`max_abs/max|ref|` between 3.5e-3 and 2.3e-2, `rel_rms` up to 3.5e-2 at 3 layers in bf16; these are the engine's known distance from the fp32 reference, not a change).

### `plan show` (Mia header fixture, `--backend cuda`)

Text render: identical with and without the flag (md5 equal). JSON (`raw/plan-cuda-ip0.log` against `raw/plan-cuda-ip1.log`): 160 lines differ, exactly 4 per layer times 40 layers, and nothing else:

| step (decode, per layer) | before | after | why |
|---|---|---|---|
| `rope(q)` outputs | `ws:q_rope` | `ws:q` | the alias |
| `sparse_attn` input 0 | `ws:q_rope` | `ws:q` | reads the rotated `q` |
| `rope_inverse(o)` outputs | `ws:o_derot` | `ws:o` | the alias |
| `wo_a(group-major)` input (`act_quant(o)` on FP8) | `ws:o_derot` | `ws:o` | reads the de-rotated `o` |

Steps per decode step (1179), prefill rows/steps/op counts, kernel picks and the `ops` list are unchanged. The cpu backend's plan (`raw/plan-cpu.log`) still has `q_rope` / `o_derot`.

### On endor

- `cargo check -p intarsia-core -p intarsia-models --tests` (#8342): pass.
- `cargo test --workspace --no-fail-fast` (#8344, first run): 1229 passed, 2 failed, 29 ignored. The failures: `every_env_var_in_the_code_has_a_row_in_docs_tuning_md` (the `tuning.md` row for the new variable was added after that run started; passes now), and my new graph test, whose first draft used a 4-row draft graph (a block is 5 rows) and, later, dpos probe positions the runtime itself skips. Both fixed (the graph test also skips `phantom_call` for device-position steps, whose `act_qdq` output is the whole slot by design); the final full run is 1231 passed, 0 failed, 29 ignored.
- CI cuda-cfg check (#8346): pass.

## Still needs a window (not run, by the brief)

- `scripts/dsv41-tp2/run-full.sh`, golden, and any TP=2 job; vLLM stays up. The engine claim here is the 3-layer, single-rank, short-prompt one.
- A prefill measurement through the engine. The bench figures are kernel level (E5 +7.6 % CSA T=2048, E3 q + o^-1 pair 1.165 -> 0.062 ms); the 88 ms per 4K request is an extrapolation and was not measured.
- TP=2 and the 40-layer, long-context (128K, `nsplit > 1` at long Nc) runs.
- A literal `origin/main` binary for the A/B (this doc used the env switch and main's attention file).
- The `kernels test --quick -m <dsv41 dir>` form does not work for DSv4.1 (`config.json missing 'num_experts'`: the sweep reads Qwen-style configs); the default `--quick` sweep covers the DSv4.1 ops at its own shapes. Not a regression.

## Reproduce

`raw/attnwire-spark.sh` (driver; `killmine.sh` and `memwatch.sh` are its helpers): copies the tree to `~/intarsia-dsv41-attnwire` on spark2 (target seeded from `~/intarsia-dsv41-tp2/src/target`), `MAX_JOBS=4`, a memory watch that kills only this job's processes below 10 GB available (lowest seen 50 GB), fetches the logs to `raw/` and removes the directory. Its exit code was 1 only because of the two `kernels test -m` lines above; their logs were deleted. Logs in `raw/`.
