# Lane s27d: Qwen3.8-27B single-stream decode on CUDA (spark1, GB10) against TensorFold

Date 2026-10-27 (EDT). Branch `lane-s27d/cuda-27b-decode`, PR #199, base `origin/main` `6bcfe52ce`. Box spark1 (spark-4e77, GB10), model
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (the checkpoint of `recipes/qwen3_5_vl.toml`; copied from tatooine to `~/intarsiaml-s27d/models/`,
fleetq #6061), `intarsia serve --backend cuda --max-seq 20480 --mem-budget 45`, W4A16 (the CUDA default), single sequence, tp=1. Raw logs:
`2026-10-27-s27d-cuda-27b-decode/raw/`; scripts: `scripts/s27d/`. Every GPU window ran in a fleetq quiet slot
(`--quiet gpu,cpu`); spark1-vllm (flight, guidance, capcom) stayed up and idle (the vLLM `generation_tokens_total` counter did not move
across the six gate rounds, `raw/gate6/load.txt`; the first window (4k numbers), the nsys run and the 16k run have no before/after snapshot, the counter read flat at 06:32).

## 0. Result

**The 27B on CUDA decoded without speculation (13.6 tok/s) because its recipe had no CUDA MTP default. Turning on the recipe's own MTP head
(`--spec mtp --spec-k 2`) with the 65536-row draft head by default on CUDA tp=1 takes 1k decode from 13.6 to 22.7 tok/s (x1.67, six
interleaved rounds, every round >= x1.59), prefill unchanged (-0.2 %).** The change is two recipe keys (`[tuning.cuda] mtp_default_k = 2`,
`mtp_draft_vocab = 65536` in `recipes/qwen3_5_vl.toml`); greedy ids are identical to `--spec off` (section 5). It does not close the gap
to TensorFold's 83 tok/s, and no recipe key, kernel or graph change in this lane can: that 83 is TensorFold's two-Spark (TP2) vendor
figure; the same method on one Spark is 31.7 to 39.7 tok/s (section 1), and one Spark streams 18 GB of weights per token.

**The task's premise "MTP on (default)" was not true for this model on CUDA**: the recipe only carried `[tuning.hip]` MTP keys, so the
base build ran plain decode (the 35B has had a CUDA MTP default since lane cmtp/mpf, the 27B had none).

### Baseline vs branch, decode tok/s (prefill tok/s in brackets)

| context | base `origin/main` default (= `--spec off`) | MTP k=2, full head (`--spec mtp`, n=1) | **branch default (MTP k=2, draft head 65536)** | TensorFold 1 Spark (localeval) | TensorFold TP2 (localeval) |
|---|---|---|---|---|---|
| 1k (990 tok) | 13.6 (64) | 19.5 (64) | **22.7** (64) | 31.7 (n/a) | 52.0 |
| 4k (4090 tok) | 13.5 (64), n=1 | 19.8 (64), n=1 | **23.7** (64), n=1 (explicit flags, same config) | 39.7 (578) | 50.9 (779) |
| 16k (16.6k tok) | 12.8 (65), n=1 | not run | **20.6** (65), n=1 | 37.1 (582) | 54.7 (727) |

- Base and branch at 1k are medians of 6 interleaved rounds (`raw/gate6/`, paired ratios 1.709, 1.689, 1.652, 1.684, 1.645, 1.591;
  base 13.59-13.65, branch 21.63-23.21). The 4k row is one round of the first window (`raw/base-w1/`, explicit flags). 16k is a single
  sample per arm with no warmup (`raw/dec16k/`, a 16k prompt prefills in 4.3 min, so localeval's warmup plus 3 reps was out of budget).
- MTP off / on: "base default" is the MTP-off column (the branch's `--spec off` is the same code path: the diff touches recipe keys only).
- Prefill is 64 tok/s at every size on both arms (the CUDA MXFP4 matmul has no tensor-core GEMM and re-reads the weights per 8 rows, see
  section 6); TensorFold prefills the same model at 578-582 tok/s on one Spark.
- k and draft head (w2, 2 rounds, 1k): k=1 21.6, **k=2 23.1**, k=3 21.6 tok/s (dev rows of `raw/w2/`); k=2 is best, as on HIP. Draft head: full
  head 19.5 vs 65536 rows 23.4 (+20 %, w1, one round).

### Gap to TensorFold (decode, same localeval method, same box type)

| | before (base) | after (branch) |
|---|---|---|
| vs TensorFold 1 Spark, 1k (31.7) | 0.43x | **0.72x** |
| vs TensorFold 1 Spark, 4k (39.7) | 0.34x | **0.60x** |
| vs TensorFold TP2, 4k (50.9) | 0.27x | 0.47x |
| vs the Phase-4 gate figure 83 (TensorFold TP2 vendor "code sampled" 82.4; vendor 1 Spark is 49.7) | 0.16x | 0.27x |

Sources: `/opt/code/model-benchmarks/data/records/localeval.jsonl` runs `20260926-143519-spark2-tensorfold-1spark`,
`20260926-152841-spark-tensorfold-tp2-tp2` (TensorFold 0.3.0 CUDA, MLX-4bit 27B + DFlash2, unanimis `af855bd8`: vendor protocol 1 Spark
49.7 / 45.8 tok/s code / chat, TP2 76.5-82.4 code). Not re-run. localeval's nonce-prefix filler prompts accept fewer draft tokens than the
vendor's code prompts, which is why TensorFold's own localeval numbers sit below its vendor ones; our MTP numbers use the same localeval prompts.
**The Phase-4 gate of docs/ARCHITECTURE.md (83 tok/s) compares a two-box figure with a one-box run**; on one GB10 it is out of reach
for any engine that reads 18 GB per pass: the plain pass is 72.7 ms (13.7 tok/s) and MTP k=2 yields 2.3-2.5 tokens per 102 ms pass.

## 1. Profile (4k context, MTP k=2, draft head 65536; nsys, `raw/prof-mtp2dv/`)

`scripts/s27d/prof-decode.sh`: 4210-token prompt, 192 tokens, 76 verify passes, 2.51 tokens per pass (acceptance 76 %), 24.5 tok/s; decode
window 7.80 s = 102.6 ms per pass, kernel time 97.0 % of it, idle 3 %. Per pass:

| sink | ms/pass | share | note |
|---|---:|---:|---|
| `mxfp4_gemv_kernel` (27B body, 7-8 calls per layer, M=3 verify rows) | 68.7 | 67 % | 12.9 GB per pass: **188 GB/s** |
| lm_head verify (bf16 `gemv_kernel`, 248320 rows x 5120, M=3) | 10.9 | 11 % | 2.54 GB: 233 GB/s |
| 2 draft steps: 65536-row head (2.8 ms each) + MTP layer (bf16) | ~10.9 | 11 % | draft head 0.67 GB: 239 GB/s |
| decode attention (`splitkv` + combine) | 3.6 | 3.5 % | |
| GatedDeltaNet (`gdn_delta_split`, conv, conv state) | 2.5 | 2.4 % | |
| norms, rope, kv append, glue | ~1.5 | 1.5 % | |
| launch gaps | 3.1 | 3 % | CUDA graphs already replay (76 replays, 4 captures) |

Reading: plain decode (M=1) is at the memory roofline already (72.7 ms for 18.0 GB = 248 GB/s including the lm_head; the MXFP4 GEMV streams M=1 at
255-263 GB/s in the microbench), so speculation is the only lever that moves tokens per second. Inside a pass the MXFP4 GEMV drops to 188 GB/s at the verify's
M=3 (bf16 GEMVs hold 233-239 GB/s); at most ~12 ms of 102 would come back if M=3 ran at the M=1 time. Section 6 has what was and was not tried.

## 2. What changed

- `recipes/qwen3_5_vl.toml`: new `[tuning.cuda]` with `mtp_default_k = 2` and `mtp_draft_vocab = 65536`; the comment above `[tuning.hip]` no longer
  says CUDA is unmeasured.
- Tests: `crates/intarsia-models/tests/tp_mtp.rs` (`only_the_measured_recipe_backend_pairs_default_to_mtp` now lists `qwen3_5_vl cuda tp1 2`),
  `crates/intarsia-models/tests/mtp.rs` (27B draft vocab on CUDA is 65536, `--mtp-draft-vocab 0` turns it off, Metal/CPU stay unset),
  `crates/intarsia-models/tests/quark_scheme.rs` (the test that appended its own `[tuning.cuda]` table now inserts its key into the recipe's).
- `docs/tuning.md`, `docs/recipes.md`, `docs/decisions/LOG.md` rows. Scripts `scripts/s27d/`.
- Rollback: `--spec off` or `INTARSIA_SPEC_DEFAULT=0` (MTP default); `--mtp-draft-vocab 0` (full draft head). `serve --max-batch` > 1, a prefix store
  and a checkpoint without MTP head keep plain decode (the shared `mtp_default` rules, unchanged).

## 3. Gates

| gate | result | evidence |
|---|---|---|
| decode >= +5 % paired, >= 6 interleaved rounds, median | **PASS**, x1.668 (rounds x1.591-x1.709), 1k, base `origin/main` vs branch binary, default flags | section 0, `raw/gate6/agg.txt` |
| prefill >= 0.99x | **PASS**, -0.2 % (paired -0.2, -0.0, +0.5, -0.1, -0.3, -0.0 %); 64 tok/s on both arms | `raw/gate6/agg.txt` |
| greedy ids identical, 8B and 27B probe sets | **PASS**: 27B, `--spec off` vs the recipe default (no flag; the log line `the default of recipe qwen3_5_vl on cuda at tp=1` and a `spec:` stats line are required): short-0, short-1, short-2, 4k-0 (3870 tokens), 256 ids each, identical; 8B (`Qwen/Qwen3-8B`, base vs branch binary): short-0, short-1, 4k-0, 256 ids and first-token top-20 logits identical. **Trimmed set** (4 + 3 prompts, no 16k) | `raw/ids8/`, `raw/ids27/` |
| KL / MMLU (only if numerics change) | not applicable: the verify rows are bitwise the sequential steps, the drafter and its vocabulary only move acceptance; ids gate stands in | section 5 |
| kernel admission, 3 planted mutants (new kernel only) | not applicable: no kernel added or changed | section 2 |
| `plan show` byte-identical on 8B, 35B, Gemma-4 (CUDA) | **PASS**, json and text for all three, base vs branch binaries | `raw/plan/`; the 27B plan differs only by the MTP head's bf16 ops (`add_bf16`, `matmul_bf16`, `matmul_bf16_gemv`, `logits_bf16_gemv_rows`, `matmul_pair_bf16_gemv`), as intended |
| Mac `cargo test --workspace` | **PASS**, 1039 passed, 0 failed (`--no-fail-fast`) | `raw/cargo-test-workspace-mac.log` |
| CI green | **PASS**, all six checks (cargo check linux, cuda cfgs, hip cfgs, macOS metal; cargo test linux; reproducible build) on the merged head | PR #199 |
| new behaviour behind an INTARSIA_* knob | **PASS for the default path** (`INTARSIA_SPEC_DEFAULT=0` = base behaviour exactly; `--spec off` too). **Partial for the draft head**: a user who passes `--spec mtp` explicitly on the CUDA 27B now also gets the 65536-row head (before: full head); there is no env var for that, rollback is the flag `--mtp-draft-vocab 0` only (same as the existing HIP key) | section 2 |

## 4. Measurement notes

- The ids, `plan show` and speed gates ran on the dev tree = `6bcfe52ce` plus the recipe change. The merge of #197/#198 that followed touches HIP tp>1 (`tp_partial`) and `qualify`, not CUDA tp=1; CI and the Mac test run cover the merged head.
- Not covered: per-category decode (on HIP the 65536 head lost 16 % on Chinese text; CUDA unmeasured, break-even is about 1.4 tokens per pass and k=2 at 51 % acceptance gives ~1.77), a 16k ids prompt, more than one 4k/16k decode sample.
- TensorFold's 83 is described in docs/ARCHITECTURE.md as "27B EXL3 + DFlash2"; the only TensorFold 27B records here (TensorFold 0.3.0 CUDA, MLX-4bit 27B + DFlash2) are 1 Spark and TP2, and no EXL3 27B record exists in model-benchmarks.

- The shared vLLM (`guidance`) carried ~78 tok/s of unrelated traffic for ~20 minutes at the start (fleetq quiet jobs waited for it); every timed
  window began after that stopped and the counter was read before and after each arm (`raw/gate6/load.txt`, delta 0 everywhere).
- Branch decode varies across rounds (23.2, 23.0, 22.5, 22.9, 22.4, 21.6) while base is flat (13.59-13.65). MTP throughput depends on acceptance and on
  clocks; no pattern with arm order or outside load was found (vLLM counter flat), and the gate margin is large either way.
- First `serve` load of a new binary re-runs kernel admission (170 s once, cached after).
- localeval `speed --reps 1` was used for rounds 3-6 (one warmup plus one rep per size); rounds 1-2 used 2 reps.

## 5. Why ids are expected and checked identical

`--spec mtp` verifies the drafts with the main model's own rows: on CUDA the MXFP4 matmul keeps a fixed per-row reduction order (independent
of M), decode attention runs per row for MTP (`INTARSIA_CUDA_VERIFY_ATTN_PER_ROW`) and GatedDeltaNet rows run the sequential recurrence, so a
verify row is bitwise the plain decode step (docs/results/2026-10-26-cmtp-cuda-mtp-default.md). The draft head's vocabulary and the drafter's
numerics only decide how many drafts are accepted. The measured check is the ids gate of section 3.

Acceptance in the ids runs: 69.8 %, 62.8 %, 76.2 % (short), 80.5 % (4k); 2.26-2.60 tokens per pass.

## 6. Not done, and why

- **MXFP4 GEMV at M = 2..4 (largest remaining in-pass decode lever, bitwise)**. Standalone microbench (`scripts/s27d/mxfp4bench/`, `raw/microbench/bench.txt`, the
  27B's six shapes, weights rotated through 4 copies, 40 launches): the current kernel streams M=1 at 255-263 GB/s (the roofline; the plain
  decode is bandwidth-bound and untouchable), but M=3 at 198-223 GB/s and M=4 at 157-190 GB/s (M=3 costs 1.15-1.3x the M=1 time on the
  same bytes, the profile above shows 188 GB/s in the engine): each extra verify row re-converts the 32 activations per group and runs 32 FMAs,
  and the kernel is ALU/issue-bound for rows >= 2, not latency-bound. Batching the lane's weight loads (B = 2 or 4 groups in flight, arithmetic order unchanged, bitwise equal in
  every case) did not help (within +-8 % of the baseline, sometimes slower), so memory-level parallelism is not the limit. A kernel that stages the activation rows once per block
  (shared memory, f32) and reuses the dequantised weights across rows could bring M=3 to the M=1 time: at most ~12 ms of the 102 ms pass (~12 % decode), bitwise
  possible (same per-row fma chain). Not built: it needs `kernels test` admission with planted mutants and its own paired window, which did not fit the budget.
- **CUDA MXFP4 prefill** (user-visible and 9x behind TensorFold): 64 tok/s, 64 s to first token at 4k, 4.3 min at 16k. `mxfp4.cu` is the M1-drill GEMV
  ("correct, not fast: there is no MXFP4 tensor-core GEMM here"); a dequantise-to-bf16 tile GEMM on the `nvfp4_mma` pattern is the fix. Not the decode gap,
  but it set the cost of every window here (a 6-round 1k/4k/16k A/B does not fit in a window) and it bounds MMLU-style numerics gates on this model on CUDA.
- **DFlash on CUDA** (planned, not built): TensorFold gets its 46-50 tok/s on one Spark from block drafting (DFlash2); MTP k=2 tops out near
  2.5 tokens per pass. An NVFP4 DFlash2 draft model is already on spark1/spark2 (`~/models/Qwen3.8-27B-DFlash2-NVFP4-*`).
- **Draft cost**: the two draft steps are 11 % of a pass (bf16 MTP layer + 65536-row head). Quantising the drafter's weights is output-invariant (only
  acceptance moves) and could save ~5 ms per pass; not tried.
- **lm_head verify** reads 2.54 GB bf16 per pass (11 %); quantising it changes numerics (KL/MMLU gate; MMLU on this model needs the slow prefill).
- 16k and 4k decode have one sample per arm (see section 0; 16k: ttft 256.1 s base vs 255.7 s branch, decode 12.83 vs 20.55, x1.60); 16k MTP with the full draft head was not run.
- The ids gate used a trimmed set (section 3 / 5), not the 18-prompt sets of lanes cmtp/mpf, for the same prefill-cost reason.

## 7. Fleet log

Owner `claude-f31ccf91`, spark1 only (vLLM `flight/guidance/capcom` up the whole time, nothing taken down): copy of the checkpoint #6060/#6061
(net,disk), builds #6066 (base), #6110 (dev), quiet GPU windows #6081 (w1), #6093 (nsys), #6111 (w2), #6132 (w3), #6146 (w4), #6154 (w5, 16k single-sample decode; #6147 was killed after its client crashed on a missing usage field and the script was fixed); shared
functional smokes #6075, #6077, #6078. Trees `~/intarsiaml-s27d/trees/{base,dev}`, model `~/intarsiaml-s27d/models/Qwen3.8-27B-MXFP4`,
worktree `.worktrees/s27d`: kept (see the report).
