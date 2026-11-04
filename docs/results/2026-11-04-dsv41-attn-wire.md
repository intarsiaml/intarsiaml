# DSv4.1 CUDA prefill: interleaved softmax (E5) and in-place q / o rope (E3) wired into the engine

Date 2026-11-04 (early morning EST), branch `dsv41/attn-wire` (worktree `.worktrees/dsv41-attn-wire`, from `6f56a2c2d` plus `git merge origin/main` at local ref `3c57d50a2`). Local commits only: nothing is merged or pushed. Follows [the tfblock bench doc](2026-11-03-dsv41-attn-tfblock.md) (fleetq #8282: E2 negative, E5 and E3 `ip` bench-proven at kernel level). E2 (`ATTN_TFBLOCK`, `attn_sparse_mma_tfb`) is not brought in and nothing of it is left in the tree.

## Status

Both changes are in. Through the engine they are bit-identical to a literal `origin/main` build: golden 640/640 and the 32K and 65K `longp` prompts at TP=2 on the full 40-layer model (section "The TP=2 window"). The engine gain is about 2 to 3 % of prefill, not the kernel-level figures; decode is unchanged. What did not run is listed under "Not run".

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
- The first-pass version of the test `in_place_rope_leaves_no_reader_of_the_unrotated_rows` (rewritten, see "Review fixes") built the graph with and without the flag for decode, 4, 16, 512 and 2048 rows, the device-position verify / absorb pass graphs and the draft graphs, and asserts: same steps, lanes, positions, extras, shapes and dtypes; the only changed buffers are the two ropes' outputs and their next readers; the producer of the buffer is `wq_b` (or `AttnSparse`) on lane 0; no step between the producer and the rope touches it; the first step after the rope is `AttnSparse` (for `q`) or `act_quant(o)` / `wo_a` (for `o`); every step still passes `phantom_call()` (`OpCall::check`, so no generic alias rejection); the graph without the flag has no step that aliases an input.

## Checks

### Kernel level, on spark2 (CUDA)

- `crates/intarsia-backend-cuda/tests/dsv41_dpos.rs` gained `check::rope_in_place`: the same rope launched with `x` as its own output, for bf16 and f32, T 1, 4, 5, ratio and stride (1, 1), (4, 4), (2, 2), forward and inverse, 8 positions including 4093 and 30000, head widths 512, 128 and 64, host position and device position, compared bitwise with the copy kernel's output (the whole row, so the unrotated 448 dims are checked to be left alone). Passed.
- `dsv41` against the cpu reference and `kernels test --quick` pass with E5 in the kernel (`attn_sparse` 18 cases, max_abs 3.9e-3, rel_l2 2.1e-3, as the tolerance expects; this sweep judges by tolerance, not bits, which is why the second pass added the bit guard below).

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

## Review fixes (second pass, same branch)

An Opus review found no output-changing defect by reading but called the guarding tests weak. Eight fixes, each checked below. Spark job: fleetq #8363 (`--quiet gpu,cpu --mem 20 --expect 60`, 9 minutes, logs in `raw-review/`, script `raw-review/attnwire-spark2.sh`). Endor: one slot (#8364, `--load cpu --expect 50`) for the targeted tests, the workspace run and the CI cuda-cfg check.

1. **The unrotated-read check now walks the whole graph.** `check_no_unrotated_read` follows `ws:q` and `ws:o` through every stage in execution order as `Unwritten` / `Raw` (after `wq_b` or the attention) / `Rotated` (after the in-place rope): the rope must find `Raw`, and any other step anywhere later that touches the buffer must find `Rotated` and be the one consumer (`AttnSparse` input 0; `act_quant(o)` or `wo_a`). `Extra`s that name `q`, `o`, `q_rope` or `o_derot` fail it. `check_base_buffers` does the same for the unflagged graph (`ws:q` is touched only by `wq_b` and `rope(q)`, `ws:o` only by the attention and `rope_inverse(o)`), and the buffer rename between the two graphs is now exactly `q_rope -> q`, `o_derot -> o`, so a later reader of the unrotated rows in the base graph cannot be mapped onto the rotated buffer. Proof it bites: `the_unrotated_read_check_fails_on_an_injected_late_reader` injects readers into copies of the graphs and requires the panic to carry the right message: a reader of `q` between `wq_b` and its rope ("unrotated rows"), a reader of `o` between the attention and its rope ("unrotated rows"), a stray reader of `q` at the start of the next layer and of `o` at the end of layer 9 ("is not the consumer"), late readers of the unrotated `q` and `o` in the base graph ("reads the unrotated"), and a rope applied twice ("not freshly produced"). The old first-toucher check, run on those graphs, would have accepted the layer 9 and base cases (it looked only at the first step after the rope, in its own stage).
2. **Configurations with the flag on.** `in_place_rope_in_the_fp8_dense_and_tp_configurations`: FP8 mode (`act_quant(o)` between the de-rotation and `wo_a`, 40 of them asserted), the dense grouped `wo_a` (the EXL3 slices hidden, `wo_a.weight` supplied), both, and a rank of TP=2 (a `Variant` shape wrapper shards the `wq_b` columns, `wo_b` rows, sinks, routed and shared experts and the head as the rank cache does), each through the same battery: decode and prefill-shaped host graphs, the device-position pass graphs (plain, verify, absorb) and the draft graphs, each against the unflagged graph. TP=2 here is the graph with `world = 2` and its collective `Extra`s; no TP run.
3. **Engine hash asserted in the test.** `device_position_passes_equal_host_passes` builds a second runtime with `INTARSIA_DSV41_ROPE_INPLACE=0` and asserts its host-position and device-position outputs are bitwise those of the in-place runtime (`rope_inplace_for` is no longer cached, so one process can build both). It still skips without `INTARSIA_DSV41_CACHE` (it is a cache-gated test, like its neighbours), so the evidence is the Spark run. **E5 bit guard:** `attn_sparse_mma_is_bitwise_the_8282_base_kernel` (`crates/intarsia-backend-cuda/tests/dsv41.rs`) regenerates the five attention shapes of #8282 with the bench's generators (`std::mt19937`, libstdc++'s float `uniform_real_distribution`, written out in Rust) and asserts the five recorded hashes of the unmodified kernel (`expected-hashes.txt`). On spark2 all five match (`4ee9a847ca24437f`, `aa254a578fedfa25`, `90ad1d7fcca0b866`, `21b89af74c9af5a2`, `e450cadc14aad612`), which also shows the Rust generators reproduce the bench inputs bit for bit. Negative control: with `INTARSIA_DSV41_ATTN_SIMT=1` (the SIMT kernel) it fails on all five shapes (`raw-review/cuda-guard-simt-control.log`). It is a CUDA test, so it runs on a Spark and skips elsewhere.
4. **Engine configuration.** The test keeps the prefill logits in the compared output and takes `INTARSIA_DSV41_DP_CHUNK`. Run on spark2 with chunk 2048, prompt 4200, cap 8192, 24 decode tokens and replayed CUDA graphs (17 passes, 2 197 760 values): in-place equals out-of-place, host equals device position. The default run (chunk 64, 140 prompt, speculation, graphs: 53 passes) also passes. **Not covered by that 3-layer world-1 run (the TP=2 window below covers the first two):** 32 heads per rank (the TP=2 head count: 64 heads over 2 ranks, and `kHeads = 32` in `attn_sparse_mma`; an earlier draft of this sentence said 16, which was wrong), the 40-layer stack, and more than about 2100 compressed rows (a 128K context) and `nsplit > 1` at long `Nc`.
5. **The dpos test now shows the in-place path ran.** Bits alone cannot (the copy kernel gives the same bits), so `kernels/cuda/dsv41_misc.cu` counts in-place launches (`ik_rope_inplace_launches`, declared in `ik_cuda_abi_dsv41.h`) and `check::rope_in_place` asserts the counter rises by one for every aliased launch (host and device position) and not for the out-of-place launch. The counter counts host launches; a replayed CUDA graph does not count again, so the engine runs prove the graph and bits, not the kernel choice per replay.
6. **Refusal.** `ik_rope_tail` returns `IK_ERR_INVALID_ARG` when `x` and `out` overlap and are not the same tensor (one pointer with two dtypes, or a partial overlap). Test `rope_refuses_aliases_other_than_the_same_tensor` (passes on spark2): f32 in / bf16 out at one pointer, a one-row-shifted overlap, both refused; disjoint halves and the same tensor run.
7. **Contract.** `docs/design/dsv41-ops.md`: the "outputs never alias inputs" rule now names the `RopeTail` exception and what it requires of the graph.
8. **Dead value.** `dsv41_engine.rs` builds `GraphTp` with `..GraphTp::default()`; `GraphTp::rope_inplace` documents that `Dsv41Runtime::new` sets it from the backend.

**Endor results (#8364, last commit before the docs row fix):**
- `cargo test -p intarsia-models --test dsv41`: 10 passed, 0 failed (includes the 3 new graph tests; the injection test caught each of its 7 injected faults with the intended message).
- `cargo test --workspace --no-fail-fast`: 1232 passed, 1 failed, 29 ignored. The one failure was `every_env_var_in_the_code_has_a_row_in_docs_tuning_md` (my new test variable `INTARSIA_DSV41_DP_CHUNK` had no `docs/tuning.md` row). Row added; `cargo test -p intarsia-cli --test docs_env_vars` then passed (2 of 2). The full workspace was not rerun after that one-line docs edit.
- CI cuda-cfg check (`INTARSIA_CUDA_CHECK_ONLY=1`: `-p intarsia-backend-cuda --all-targets`, `-p intarsia-cli --features cuda`, `-p intarsia-models --tests`): pass.

Two things to know, as the review asked:

- **The graph fingerprint omits `rope_inplace`** (`dsv41_engine::fingerprint`: source, world, layers, cap, chunk, act, spec). Two ranks with different `INTARSIA_DSV41_ROPE_INPLACE` settings are therefore not refused at the handshake. That is harmless only while the change is bitwise (it is: both settings give the same bits, above). The same is already true of `INTARSIA_DSV41_HC_FUSED` and `INTARSIA_DSV41_LANES`. If the in-place rope ever stops being bitwise, add it to the fingerprint.
- **This branch does not contain `dsv41/replay`**, so the unrotated-read check says nothing about replay graphs. Whichever of the two merges second must re-run `in_place_rope_leaves_no_reader_of_the_unrotated_rows` (and the injection test) against the replay graphs; a replay that re-runs a rope without its producer would double-rotate (the "rope applied twice" injection is that case).

## The TP=2 window (2026-11-04 09:47 to 10:19 EST, fleetq #8402)

One exclusive job over both Sparks (`--takes-down spark1-vllm,spark2-vllm`, `scripts/dsv41-tp2/run-full.sh` through `raw-window/window.sh`), after the merge of `origin/main` (PRs #296, #297, #298; chunk default 2048) into the branch, head `91cb61fc8`. Binaries built in a `--load cpu` job on spark2 (#8398, `raw-window/build-both.sh`): `intarsia.base` is the literal `origin/main` at `809674eb9` (git archive, nothing of this branch), `intarsia` is the branch. Rank caches `r0of2` / `r1of2`, cap 65536, engram hot 1024 MiB, rank 1 spin 100 ms, `earlyoom -m 2,1` on both. Eight configs, each a fresh start of both ranks, alternating base and head: ab0 base, ab1 head, ab2 base, ab3 head at chunk 2048, then ab4 to ab7 the same at chunk 512. ab0 and ab1 also ran golden and `longp`. The vLLM containers were stopped at 09:47:26 and were up again on both Sparks at 10:19:12 (docker "Up 3 minutes", fleetq `up`, restored by the script's trap; no failure path was taken).

### Bit identity (full 40-layer model, TP=2, chunk 2048)

- **Golden**, 10 prompts x 64 greedy tokens, base against head (`golden_tf.py compare`, `raw-window/golden-compare-base-vs-head.txt`): **640/640 identical, 0 prompts diverged.** No first differing token exists.
- **Long context** (`longp`: 16 greedy tokens with top-5 logprobs; `cmp_longp.py`, `raw-window/longp-compare-base-vs-head.txt`): the 32768-token prompt and the 65000-token prompt are **bit-identical** in text, tokens and all top-5 logprobs (16/16, max abs diff 0). Wall: 36.3 s and 73.3 s (base), same for head within noise.
- Both golden and `longp` ran at chunk 2048 only, the shipped default; chunk 512 was timed but not compared for tokens.

### Prefill and TTFT through the engine (localeval `speed`, `--gen 128 --reps 3`, medians of the 6 requests of the two rounds)

localeval builds each prompt to about the size asked, so the token count differs by a few tokens between configs (4075 to 4097 at "4096", 33277 to 33323 at "32768"); prefill tok/s is the fairer figure, TTFT is listed too.

| chunk | prompt | base TTFT s | head TTFT s | base prefill tok/s | head prefill tok/s | gain (tok/s) |
|---|---|---|---|---|---|---|
| 2048 | 4K | 4.18 | 4.05 | 980 | 1009 | +3.0 % |
| 2048 | 32K | 34.41 | 33.44 | 968 | 996 | +2.9 % |
| 512 | 4K | 5.00 | 4.86 | 820 | 839 | +2.3 % |
| 512 | 32K | 41.45 | 40.58 | 803 | 820 | +2.1 % |

Every one of the 24 head requests is faster than every base request of the same cell (the rounds do not overlap; e.g. 32K at 2048: base 34.32 to 34.49 s, head 33.35 to 33.51 s). Decode is unchanged within noise (46.9 to 49.2 tok/s for both, medians 47.6 / 46.9 at 4K and 48.2 / 47.4 at 32K for chunk 2048). **What the engine shows:** about 130 ms less TTFT for a 4K request at chunk 2048 and 140 ms at 512 (E5 and E3 together; the window does not split them). The kernel-level E5 gain (+7.6 % of the attention kernel) is a small part of a request, and the 88 ms per 4K request extrapolated for E3 alone cannot be told apart from the total here. The gain is real and repeatable, but it is 2 to 3 %, not more. Chunk 2048 over chunk 512 is the larger lever (+19 % prefill at 4K for the base, +20 % for the head). Raw: `raw-window/ab*/le-speed-*.log`.

### Memory (1 s `memfine` on both ranks, `raw-window/*-memfine-094726.log`)

| config | spark1 min MemAvailable GiB (after load) | spark2 min (after load) |
|---|---|---|
| ab0 base 2048 (golden, longp 65K, speed) | 7.34 (7.48) | 7.18 (7.18) |
| ab1 head 2048 (same steps) | 6.84 (7.15) | 6.73 (6.97) |
| ab2 base 2048 (speed) | 6.85 (7.42) | 6.56 (7.27) |
| ab3 head 2048 (speed) | 6.85 (7.50) | 6.51 (7.39) |
| ab4 / ab6 base 512 | 6.81 / 6.82 (8.66 / 8.65) | 6.63 / 6.37 (8.52 / 8.52) |
| ab5 / ab7 head 512 | 6.77 / 6.80 (8.70 / 8.68) | 6.60 / 6.61 (8.56 / 8.55) |

In every config the lowest value (6.4 to 7.3 GiB) is in the weight load, not in a prefill; "after load" is from 45 s on, which includes the longp 65K run in ab0 and ab1. At chunk 2048 the steady minimum is 6.97 to 7.50 GiB for both binaries, at chunk 512 it is 8.5 GiB. Head against base: ab0 to ab1 is 0.33 / 0.21 GiB lower, ab2 to ab3 is 0.08 / 0.12 GiB higher, so there is no systematic change in what the box has available. What does change is the engine's workspace, as it should with the aliasing: **1.136 GiB -> 1.011 GiB at chunk 2048 and 0.304 -> 0.273 GiB at chunk 512** per rank (`q_rope` and `o_derot` no longer exist; the "memory" line in each `serve-*.log`). The watcher (kill line 5 GiB, unchanged) never fired. The lowest margin above the line was 1.37 GiB (spark2, base, chunk 512, during load) and 1.97 GiB for the head at 2048 steady state on spark2. The 0.6 GiB margin of last night's chunk 2048 run was not reproduced in this window (why was not investigated).

## Not run

- Golden and `longp` at chunk 512 (timing only at 512), and golden against TensorFold's reference tokens (this window compares branch to base, as asked).
- Contexts over 128K, and `nsplit > 1` at the long `Nc` of such contexts (the 65000-token prompt is the longest).
- The `kernels test --quick -m <dsv41 dir>` form does not work for DSv4.1 (`config.json missing 'num_experts'`: the sweep reads Qwen-style configs); the default `--quick` sweep covers the DSv4.1 ops at its own shapes. Not a regression.
- A split of the 2 to 3 % between E5 and E3 (would need a head build with one of them off; `INTARSIA_DSV41_ROPE_INPLACE=0` switches E3 off at run time, E5 has no switch).

## Reproduce

`raw/attnwire-spark.sh` (driver; `killmine.sh` and `memwatch.sh` are its helpers): copies the tree to `~/intarsia-dsv41-attnwire` on spark2 (target seeded from `~/intarsia-dsv41-tp2/src/target`), `MAX_JOBS=4`, a memory watch that kills only this job's processes below 10 GB available (lowest seen 50 GB), fetches the logs to `raw/` and removes the directory. Its exit code was 1 only because of the two `kernels test -m` lines above; their logs were deleted. Logs in `raw/`.
