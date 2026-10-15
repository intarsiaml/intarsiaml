# Balanced prefill chunking (attention-only states): measured on Metal, a win only for tiny tails

Date: 2026-10-15 (11:20-13:40 EDT), endor (Apple M5 Max), Qwen3-8B BF16, branch `metal-prefill-balance` off `spike/metal` c556008
(worktree `/opt/code/intarsiaml-prefill-balance`). Code commit `5e3ac63`. **Not merged into spike/metal, nothing pushed.**
fleetq: endor exclusive slot **#2013** (all GPU measurements and gates below, released 13:38); spark1 CPU-only checks **#2005**, **#2016**.
Raw data and scripts: `docs/results/raw/2026-10-15-metal-prefill-balance/`.

## Verdict

The change does what was asked and is correct (tests, greedy ids, qualify all pass), but **the measured speed effect is small and
mixed**: -1.5 to -2.6 % TTFT for prompts whose old tail is tiny (9-24 rows), no change where there was no tail, and **+5 % at ~4.3k
and +1.8 % at 16k**, where the old tail was 175-290 rows and the old scheme was already efficient. If this is merged as written, 16k
and 4.3k prefill get slower. Recommendation in "Follow-up" below (limit the rebalance to small tails); not implemented or measured here.

## Contamination note (read first)

Before taking a slot I ran `cargo test --workspace` on endor four times (about 11:20-11:35 EDT, iterating on test expectations).
The workspace includes `intarsia-backend-metal`, so its GPU test binaries ran for seconds each. That overlapped fleetq **#2001**
(claude-31e10c55, "metal spec G2G3: gate sweep", exclusive, RUNNING at 11:30). #2001 was a gate sweep, not a timing run, but anyone
reading its 11:20-11:35 numbers should know. I also launched `localeval speed` against a dead URL to read its help; it queued a
fleetq job **#2012** that I cancelled within about two minutes (it never ran). Everything reported below ran inside #2013.

## What changed

`crates/intarsia-core/src/engine.rs`, `generate_one`: the tokens left after prefix reuse used to be cut into full
`model.prefill_chunk()` chunks plus a remainder (4097 rows = 2048 + 2048 + 1). New `pub fn balanced_chunk_len(remaining, max)`
gives the next chunk size: `remaining` if it fits, else `ceil(remaining / ceil(remaining / max))`. Calling it on what is left gives the
fewest chunks that fit, sizes differing by at most one, non-increasing (4097 -> 1366, 1366, 1365). The maximum chunk size
(`--prefill-chunk`, recipe `[tuning.<backend>] prefill_chunk`, `LoadOptions::prefill_chunk`) and workspace sizing are unchanged. The chunk starts wherever prefix reuse
ended: the KV cache is contiguous and no layout alignment is required, so nothing is rounded.

- **Recurrent (GDN / Mamba) states keep the fixed grid.** Their snapshots sit on chunk boundaries, and the existing invariant is that
  a run restored from a snapshot is bit-identical to a from-scratch run. Balanced sizes depend on the prompt length, so that would
  break (I saw it: `tests/graphs.rs` restored-vs-scratch logits differed by 3.6e-6, and the restore point moved 16 -> 15 in
  `tests/hybrid.rs`). Balancing is therefore `!st.has_recurrent()` only; the tests for the recurrent paths are unchanged. The test
  helper `prefill_like_the_engine` in `crates/intarsia-models/tests/graphs.rs` mirrors the gate.
- **Spec arithmetic.** The task's example "4100 -> 2050 + 2050" exceeds the 2048 maximum, so it cannot be a plan. Also, the serve path
  reuses the 3 chat-template tokens, so a 4100-token prompt computes 4097 rows: three chunks (1366/1366/1365), not two. Any prompt
  over `k*max` rows needs `k+1` chunks; balancing only evens them out.
- Not touched: the teacher-forced splitter in `Model::forward` (`model.rs`, `ids.chunks(max_tokens)`) used by `accuracy` and the
  perplexity paths. See "Accuracy".

## Tests

- New unit tests (`engine::tests`, `cargo test -p intarsia-core --lib engine::tests`, 5 tests): prompt at most max (1, 100, 2048, 0 rows),
  exactly `k*max`, `k*max+1` (4097, 2049, 257, 4107), every remaining length 1..5*max for max in 1, 2, 7, 64, 256, 1024, 2048 (sum
  right, fewest chunks, sizes within one, non-increasing, never below max/2 once split, i.e. no slivers), and prefix reuse (3, 1000,
  2048, 2049, 4099 reused tokens of a 4100-token prompt).
- `cargo test --workspace` on endor: exit 0, in #2013 (`gates/cargo-test-workspace.txt`). `cargo test -p intarsia-backend-metal --release
  -- --test-threads=1`: exit 0, all binaries ok (`gates/cargo-test-metal.txt`).

## CUDA / HIP: this is a behaviour change, not a no-op

`generate_one` is shared core code, so **dense (attention-only) models on CUDA and HIP get different chunk sizes too** (for example the
default CUDA/HIP chunk of 256: 257 rows are now 129 + 128, were 256 + 1). Hybrid/recurrent models (the 35B) are unchanged. I did not
run or measure anything on CUDA/HIP, and the CUDA/HIP goldens and speed baselines were not re-checked. The only proof is compile:

- spark1 (#2005 and, on the final tree, #2016; `--mem 8`, CPU only, no model or GPU use, vLLM left alone, scratch dir created and deleted):
  `cargo check -p intarsia-core -p intarsia-models --tests`, `cargo check -p intarsia-cli --features cuda --tests` (nvcc 13.0 present):
  both Finished, no errors.
- `cargo check -p intarsia-cli --features hip --tests`: Finished, **but spark1 has no ROCm, so the HIP crate built its stub** and Rust
  behind `cfg(ik_hip)` was not type-checked. The change itself is in the backend-independent `intarsia-core`, which was checked.

## Speed (Metal, `intarsia serve`, localeval speed)

Method: two release binaries, `--features metal`, both default (recipe chunk 2048): base = c556008 (`git archive` into scratchpad), new =
`5e3ac63` (stamped `head=5e3ac63 code-tree=4616acdf2fbd dirty=no`); md5s in `binaries-md5.txt`. One fresh `serve --max-seq 20480` per
pass, `localeval speed --url ... --thinking off`, passes interleaved ABBA BAAB (8 passes per run), macmon at 100 ms, each rep's GPU MHz
averaged over its prefill window (`scripts/an.py`). Rows computed per request = prompt tokens - 3 (the chat-template prefix is reused).
Run 1 (`perf/`): sizes 1024, 2112, 4096, 4108, 16384, 3 reps. Run 2 (`perf2/`): sizes 4096, 4108, 4140, 4300, 16384, 6 reps (24 reps per
cell). The GPU clock moved between 1280 and 1590 MHz (thermal/power, as in the p3-noise doc), so the table gives the all-rep median, the
median of reps at or above 1450 MHz, and the median scaled to 1600 MHz (`ttft * MHz / 1600`). Old tail = rows mod 2048.

Run 2 (`perf2-summary.md`), TTFT seconds:

| localeval size (prompt tok) | old tail rows | base, all | new, all | base, >=1450 MHz | new, >=1450 MHz | base, scaled 1600 | new, scaled 1600 | scaled delta |
|---|---|---|---|---|---|---|---|---|
| 4096 (4084-4092) | none (2 chunks) | 1.120 | 1.160 | 1.085 (n=14) | 1.090 (n=11) | 1.041 | 1.039 | -0.002 (0 %) |
| 4108 (4108-4115) | 9-16 | 1.160 | 1.155 | 1.160 (n=19) | 1.130 (n=16) | 1.091 | 1.075 | -0.016 (-1.5 %) |
| 4140 (4115-4123) | 16-24 | 1.195 | 1.190 | 1.150 (n=12) | 1.140 (n=12) | 1.097 | 1.082 | -0.015 (-1.4 %) |
| 4300 (4274-4302) | 175-203 | 1.260 | 1.315 | 1.180 (n=9) | 1.240 (n=11) | 1.116 | 1.173 | **+0.057 (+5.1 %)** |
| 16384 (16651-16677) | 264-290 (9th chunk) | 6.405 | 6.485 | - | - | 5.513 | 5.612 | **+0.099 (+1.8 %)** |

Run 1 (`perf-summary.md`, 12 reps per cell, scaled to 1600 MHz): 1024: 0.227 vs 0.231; 2112: 0.490 vs 0.493; 4096: 1.042 vs 1.049; 4108: 1.100
vs 1.089 (-1.0 %); 16384: 5.519 vs 5.617 (**+1.8 %**, same as run 2). The 16k and 4.3k regressions reproduce in both runs and per pass
(run 2 16k at clocks above 1380 MHz: base 6.38, 6.29, 6.35 s; new 6.43, 6.46, 6.44 s).

How to read it: per-rep spread is about +-0.1 s at 4k, so the 4108 and 4140 gains (15-30 ms) are within noise on their own; they are the
right sign in both runs and in the filtered and scaled views, but I would not call them established. The p3-noise doc estimated ~55 ms
(5 %) for removing a 2-63 row tail; I get at most half of that. The 4300 and 16k losses are larger than the noise and reproduce. A
plausible cause (not measured, no kernel profile was taken): a chunk of ~1440-1850 rows runs less efficiently than a full 2048-row one,
and balancing costs the 4300/16k prompts that efficiency to avoid a 175-290 row tail that the old scheme already handled fine (p3-noise
sweep: tails of 64-1999 rows cost nothing extra, 16-63 about +50 ms).

## Accuracy and greedy ids

- **Greedy ids, base vs new through `generate` (the only check that exercises the new planner)**: prompts short, 4k (4062 tokens), and
  exact lengths w4086, w4107, w4110, w4120, each at 48 and 256 tokens: **12 of 12 identical** (`gates/identity/summary.txt`). Chunking
  has been bitwise-invariant on Metal for M >= 2 since the NAX_MIN_M fix, so identical ids are the expected result.
- `intarsia accuracy` (bf16 gate class), base and new alike: short prefill top-1 100.00 % KL 2.832e-4, decode 100.00 % KL 3.313e-4;
  long prefill 99.72 % KL 3.699e-4, decode 100.00 % KL 3.371e-4; all PASS. **These cannot see this change**: they go through
  `Model::forward`'s own fixed splitter, not `generate_one`, so equal numbers are by construction (`gates/accuracy/`).
- Fresh `qualify --backend metal --bandwidth-gbs 614` with the new binary (`gates/qualify/`): **PASS** (11 PASS, `ab`/`tp` SKIP). It
  includes acc-4k (prefill 99.28 % KL 2.44e-4, decode 98.96 % KL 2.57e-4, baseline and envelope PASS) and its **prefix-reuse gate
  through `generate`** (turn 2 reuses turn 1 vs a cold run, past 2 chunks of 2048): PASS. That is the real-weights test of reuse plus the
  new planner. The qualify bench figures (decode 36.5 tok/s at 128) are qualify output, not benchmark data.

## Limits

- Metal, Qwen3-8B BF16 only. No CUDA/HIP run (see above); no hybrid/recurrent model is affected by construction.
- The speed deltas for tiny tails are near the noise floor of this machine (GPU clock swings of +-10 %); 8 passes x 24 reps per cell.
- localeval sizes map to prompts of 4108-4115 tokens, i.e. old tails of 9-16 rows; the 1-8 row tails and exact 1-row cases were not
  measured (the p3-noise sweep did, before the NAX fix).
- Attention-only models take no snapshots, so shifting chunk boundaries with the prompt length affects nothing but the chunk sizes.

## Follow-up (not done)

The data suggest limiting the rebalance to prompts whose old tail is small (for example under 64 rows, where the p3-noise sweep shows
the penalty) and keeping full chunks otherwise; that would keep the -15 to -30 ms at 4.1k and the unchanged 4.3k and 16k. That is a
different planner than the one specified, so I have not written or measured it. Decision for the coordinator.
