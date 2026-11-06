# CUDA packed prefill: a 16..31-row segment loses its solo bits (lane pf, cuda-pack-flip)

Lane `lane-pf/cuda-pack-flip`, 2026-11-05. Scripts `scripts/cpf/` (`scripts/pf/` belongs to an older lane). Raw logs: `2026-11-05-cuda-pack-flip/raw/`.
Lane pb (`origin/lane-pb/packed-bitid`, `docs/results/2026-11-05-pb-packed-bitid.md`) found the same root cause independently, with a minimum of 32.
pb closed without a merge. Its macro trailing-chunk probe is copied here (section 5).

## 1. Finding

On main, the batched-MTP lane saw something in its 6-prompt greedy gate (`docs/results/2026-11-05-batched-mtp.md` on `origin/lane-bm/batched-mtp`).
The gate is `scripts/bm/ids.py`, 35B NVFP4 tp1 on spark1, `serve --max-batch 16`.
In 2 of 4 runs, sequence 2 differed from the solo spec-off run from token 2 on. With `INTARSIA_PACK_PREFILL=0` every run matched.

The six prompts are 15, 103, 17, 201, 15 and 82 tokens long.
- A packed pass runs each prompt's body: the prompt minus its last token, which runs alone with the head.
- So the bodies are 14, 102, 16, 200, 14 and 81 rows.
- Before this lane, a body could pack from 16 rows (`Model::PACK_MIN_ROWS`).
- The two 14-row bodies could never pack.
- The 16-row body of the 17-token prompt (sequence 2) was the only one in 16..32 rows. It was also the only sequence that ever diverged.

## 2. Reproduction: the served schedule, now logged

New `TraceEvent::Pack { iter, segs: [(request, body rows)] }` (`crates/intarsia-core/src/scheduler.rs`, `prefill_packed`).
It is printed under `INTARSIA_CB_TRACE=1` and records which prompts pack together and with how many rows.

`INTARSIA_PACK_MIN_ROWS=16` brings back main's rule, so the dev binary reproduces main's behavior. The runs below are on spark1, fleetq #8669, `raw/ids/`, compared with the solo spec-off server.

| run | rule | packed passes (trace) | result |
|---|---|---|---|
| old16-r1 | 16 | `[(2, 16), (3, 81)]` | sequence 2 differs at token 2 |
| old16-r2 | 16 | none | ALL |
| old16-r3 | 16 | `[(3, 81), (4, 102)]` | ALL |
| old16-r4 | 16 | none | ALL |
| old16-r5 | 16 | `[(3, 81), (4, 16)]` | sequence 2 differs at token 2 |
| old16-r6 | 16 | `[(1, 16), (2, 81), (3, 102)]` | sequence 2 differs at token 2 |
| fix-r1..r8 | 33 | r4: `[(2, 200), (3, 102)]`; others none | ALL, 8 of 8 |

The 16-row body packed in 3 of 6 runs, and all 3 diverged. Packs without it (r3) matched.
- Which prompts pack together depends on arrival order. That is why main failed only intermittently.
- The divergent ids are the same in every failing run, both here and in the bm lane's raw data: `[7054, 11, 10033, 3069, 220]` against solo `[7054, 11, 3300, 279, 9944]`.

## 3. Deterministic test and smallest failing shape

`real_35b_forced_packs_are_bitwise_solo_prefill` (`crates/intarsia-models/tests/batch_cuda_real.rs`, `#[ignore]`, 35B) calls `Model::forward_packed` with forced packs. It compares each segment with that prompt's solo prefill, bit for bit:
- the state right after the body: every recurrent slot, and the K / V rows of every attention slot (70 blocks);
- the logits of the last prompt token;
- 8 greedy decode steps, logits and ids.

The packs:
- the gate's pack: bodies 102 + 16 + 200 + 81, and 16 + 81;
- one corpus body of 15, 16, 17, 20, 24, 31, 32, 33, 34, 40 or 64 rows, placed first or second, beside a 102-row or a 300-row body;
- two equal bodies.

| arm | packs run / refused | segments differing |
|---|---|---|
| fix (minimum 33) | 20 / 37 | 0: every pack is bitwise solo; every pack with a body under 33 is refused |
| `INTARSIA_PACK_MIN_ROWS=16` | 52 / 5 | 32: every segment of 16, 17, 20, 24 and 31 rows, in every position and with every partner; the gate pack reproduces the served divergence exactly (ids differ at 2); 32-row and longer segments are bitwise |

Smallest failing shape: two 16-row bodies packed together (32 rows).
- The state differs in 68 or 69 of the 70 blocks, and the logits differ from the first row.
- Greedy ids diverge only sometimes: at step 2 for the gate prompt, at step 5 or 6 for some corpus bodies. That makes the served symptom intermittent.

Raw: `raw/model/forced-fix.log`, `raw/model/forced-old16.log`.

## 4. Root cause

The packed pass runs every op that `Backend::pack_split_gemms` does not split per prompt once over the packed total. On CUDA, only the cuBLAS matmul family is split (`model.rs` `macro_split_ops`). The registry therefore picks the MoE experts kernel by the packed total.

The CUDA MoE experts kernel switches at 32 tokens (`crates/intarsia-backend-cuda/src/quantmoe.rs:95` `MMA_MIN_TOKENS`):
- below 32: `moe_experts_bf16`, warp-per-row;
- from 32: `moe_experts_mma_bf16`, tensor core;
- from 768: `moe_experts_w4a16_bf16`, bitwise mma.

The warp-per-row and tensor-core kernels do not give a row the same bits. So a body of 16..31 rows, packed into a total of 32 or more, ran the tensor-core kernel. Alone it runs the warp kernel.

Kernel-level proof: new `crates/intarsia-backend-cuda/tests/packed_rows.rs`, the CUDA twin of HIP's `packed_rows`.
- It runs the kernel the registry selects for the packed total and the kernel it selects for the segment alone, and compares the segment's rows byte for byte.
- Segments: 15..40, 255/256/257, 767/768 and 1000 rows.
- Each segment runs first and after a partner of 16, 300 or 1100 rows.
- Raw: `raw/kern/packed_rows.log`, #8669.

| op (35B shape) | segments that differ |
|---|---|
| MoE experts (256 experts, top-8, H 2048, I 512) | 15..31 rows whenever the total is 32 or more: 5 to 10 rows of each segment, both `->mma` and `->w4a16`. 32 and up: identical |
| `matmul_nvfp4`, shared expert gate/up (N 512, K 2048) and down (N 2048, K 512) | none. gemv / mma / gemm / gemm_reg are bitwise across their switches at these shapes |
| MoE top-k | none |

**Fix.**
- New hook `Backend::pack_min_rows` (default `Model::PACK_MIN_ROWS` = 16; `crates/intarsia-core/src/backend.rs`).
- CUDA returns `quantmoe::pack_min_rows()` = max(`MMA_MIN_TOKENS` 32, `GEMM_MIN_TOKENS` 33) = 33. The NVFP4 GEMM switch at 33 measured bitwise, so the extra row is margin. The coordinator chose 33 over pb's 32.
- `Model::pack_min_rows` reads the hook, or `INTARSIA_PACK_MIN_ROWS` if set. It is used in three places:
  - `forward_packed`, which refuses a shorter segment;
  - the batched engine's `pack_min_rows`, so the scheduler runs such a chunk alone;
  - `macro_plan`, which keeps a shorter trailing chunk out of a macro pass.
- HIP and Metal stay at 16, so nothing changes there.

## 5. Macro pass: lane pb's trailing-chunk probe

The macro pass also used `PACK_MIN_ROWS` for its trailing chunk. On main, a 16..32-row trailing chunk could join a macro pass, where the MoE runs at 4k+ rows.

Probe: `real_35b_macro_trailing_chunk_probe`, from lane pb, copied with `kv_fraction` 0.3.
- Prompts are two 2048-row chunks plus a trailing body of 16, 20, 31, 32 or 40 rows.
- Macro 8192 is compared with the plain schedule: last-row logits plus 8 greedy steps, bit for bit.

| arm | t = 16 / 20 / 31 / 32 / 40 |
|---|---|
| fix (33) | equal, equal, equal, equal, equal (1 macro pass each) |
| `INTARSIA_PACK_MIN_ROWS=16` (main's rule) | equal, equal, equal, equal, equal (1 macro pass each) |

So this probe found no case where main was wrong: the macro pass and the plain schedule gave the same bits for these lengths under both rules.

The kernel sweep would predict a difference at t = 16..31 under rule 16, and I have not explained why there is none. The plain schedule's chunking of the 2048·2 + t body may not run the tail as a separate pass (not verified).

The fix makes the rule consistent whatever the explanation, and it cannot cost speed: a trailing chunk of under 33 rows now runs as its own pass.

Raw: `raw/model/macro-probe-{fix,old16}.log`.

## 6. Speed at 16 streams

localeval speed, spark1, the same dev binary in both arms (A = `INTARSIA_PACK_MIN_ROWS=16`, B = fix), `serve --max-batch 16`, 2 ABBA rounds. Raw: `raw/speed/`.

| arm | round 1 | round 2 | TTFT median | pack passes / packed chunks per server |
|---|---|---|---|---|
| old rule (16) | 259.9 tok/s | 258.8 | 1.58 / 1.59 s | 5 / 14 |
| fix (33) | 260.6 | 260.6 | 1.60 / 1.59 s | 5 / 14 |

- These are 1k-token prompts, 128 generated tokens, 16 streams.
- Run ids: `20261105-210804-adhoc-pf-old16-r1`, `20261105-210908-adhoc-pf-fix-r1`, `20261105-211005-adhoc-pf-fix-r2`, `20261105-211109-adhoc-pf-old16-r2`.

The difference is +0.4 %, which is noise. Packing is identical in both arms (same pass and chunk counts), as expected: 1k bodies are far above 33 rows.

**Not measured:** 16 streams of prompts in the affected 17..33-token range.
- localeval's chat prompt has about 29 tokens of overhead. Size 24 came out as 53 prompt tokens (body 52), so localeval cannot produce such prompts. The job (#8697) was cancelled before it ran.
- The cost applies only to raw-completions prompts of 17..33 tokens. Each such prompt now runs a pass of 16..32 rows on its own instead of inside a pack. That is one extra short forward pass per such prompt, and only when several of them arrive together.

## 7. HIP

Not run, and not affected. HIP's registry picks the same kernel at every row count from 16 up for the 35B's packed ops:
- `matmul_nvfp4_mma` from 9 rows;
- `moe_experts_bf16_v2`, whose internal 15/16 switch is why HIP's minimum is 16;
- `moe_topk`;
- `matmul_bf16`.

HIP's `tests/packed_rows.rs` covers exactly these with a 16-row and a 17-row segment. The new hook keeps HIP at 16, so this lane changes nothing on HIP and radiance was not taken down.

## 8. Gates

| gate | where | result |
|---|---|---|
| forced-pack test, fix | spark1 #8669 | pass: 20 packs bitwise solo, 37 refused |
| CUDA `packed_rows` (3 tests) | spark1 #8669 | pass: no segment of 33 or more rows differs |
| 6-prompt greedy ids against solo spec-off, fix, 8 repeats | spark1 #8669 | 8 of 8 ALL. Only r4 packed (200 + 102): with the 16-row body no longer packing, these six prompts rarely pack at all. The forced-pack test is the deterministic gate |
| macro trailing-chunk probe (pb) | spark1 #8669 | pass under both rules |
| CUDA `plan show` (35B, `--max-seq 8192 --mem-budget 40 --json`), dev against `~/intarsiaml-bm/bin/main` | spark1 #8669 | byte-identical |
| CUDA `kernels test --quick` | spark1 #8669 | 106 kernels, 918 cases, 0 not admitted or failed |
| `cargo test --workspace --no-fail-fast` | endor #8670 | 1324 passed, 0 failed, 31 ignored (231 test binaries) |
| Metal no-op against main 5390e65b4 (plan show, 256 greedy ids, 8B + 35B; kernels test; goldens) | endor #8670 | identical; kernels test passes; goldens pass (section 9) |
| local: `cargo test -p intarsia-core --test scheduler`, `-p intarsia-cli --test docs_env_vars` | endor (Mac) | pass |

## 9. Metal no-op and workspace tests

These ran on endor, fleetq #8670, comparing the branch with main 5390e65b4. Raw logs: `raw/metal/`.
- **Greedy ids:** 256 each, Qwen3-8B and the 35B, short and ~4k prompts. All identical.
- **`plan show`:** 8B and 35B, json and text. Byte-identical.
- **`kernels test --backend metal`:** 49 kernels, 980 cases, exit 0.
- **Goldens** `qwen3-8b-fp32` and `-long` (bf16 class): PASS.
- **`cargo test --workspace --no-fail-fast`:** 1324 passed, 0 failed, 31 ignored, 231 binaries.
