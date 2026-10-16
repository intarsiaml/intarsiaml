# Metal continuous batching: decode-time prefill chunk 256 vs 320

Date 2026-10-16. Machine endor (Apple M5 Max), Qwen3-8B BF16 (`~/models/Qwen3-8B`), `--max-seq 8192`, `serve --max-batch 8`. Branch `metal-cb-chunk320` off `spike/metal` `6d99a6f`; commits `152b744` (flag), `93282f7` (A/B tool), later commits only add scripts and raw output. Not pushed, not merged. **The default stays 256**; this is the measurement for a later decision.

**Status: done.** Every item in the task ran. Nothing is partial.

## Result

* **320 is not a win on this machine, on these measurements.** The 4k prompt reaches its first token only 2 to 4% sooner (paired median delta -44 ms, -2.2%, 8 of 10 rounds faster; medians of the rounds 1993 vs 1918 ms, -3.8%), while the decoding streams stall 20% longer at p50 and 26% longer at p95 per iteration that carries a chunk (p50 122 to 146 ms, p95 139 to 176 ms). Aggregate throughput is unchanged (162.6 vs 162.3 tok/s, paired mean -0.3). No GPU-busy gain (0.840 vs 0.854, within the round-to-round spread).
* The reason is in the per-chunk cost: a prefill chunk adds 84.7 ms to an iteration at 256 rows and 107.8 ms at 320 rows (median over rounds of stall p50 minus no-prefill ITL p50), i.e. 0.331 and 0.337 ms per row. The flat T=128..320 region found for prefill attention alone (`/opt/code/intarsiaml-cbgap/docs/results/2026-10-16-metal-prefill-cbgap.md`) does not carry over to the whole iteration; the weight GEMMs scale with the row count, so 320 rows cost 1.25x of 256 rows and fewer chunks buy back only about 3% (13 chunks of 320 vs 16 of 256 for a 4k prompt, minus the longer chunks).
* **Identity holds at both sizes.** 0 mismatches in 203/203 requests at B=8, with reuse off and on, for `--busy-chunk 320` and for 256 (control): four runs, 812 requests, all pass.
* **Serving-path invariance 320 vs 2048 holds: bitwise equal.** 39 prompts (the 33 of the chunk-golden set plus 320, 321, 640, 641, 961 and 3841), 32 sampled-step logits rows each, token ids and every row hash identical, `Model::forward` last-row and all-rows hashes identical. So the existing 2048 goldens cover 320 (and 256, also re-verified).
* Checks: `cargo check --workspace --all-targets` pass, `cargo test -p intarsia-core` pass (66 unit + integration), `cargo test -p intarsia-cli` pass (79), Metal tests and `cargo test --workspace --release --no-fail-fast` (real model) pass.

## 1. The flag

`intarsia serve --busy-chunk N` (new, `crates/intarsia-cli/src/cmd/serve.rs`): prefill chunk of a prompt that starts prefilling while other sequences decode. Default 256 (`BUSY_PREFILL_CHUNK`, now exported as `intarsia_core::engine::BUSY_PREFILL_CHUNK`), range 16..=2048 (clap range, `u16`), capped at the model chunk (`--prefill-chunk`, 2048 on Metal) the same way as before (`min(busy, solo)`). A prompt that starts while nothing decodes still uses the model chunk. Serve-only, like `--max-batch` and `--spec-off-at`; no effect without `--max-batch`. The scheduler chunk rule is untouched (`next_prefill_chunk`, no 1-row tail on Metal). No CUDA/HIP change: the field is in `BatchedOptions` (the Metal-only batched engine) and `EngineOptions`; the default path is byte-for-byte the previous 256.

Plumbing: `ServeArgs.busy_chunk` -> `ModelArgs.busy_chunk` (skip) -> `EngineOptions.busy_chunk` -> `BatchedOptions.busy_chunk` -> `SchedulerConfig.busy_chunk`. Tests: `engine::batch::tests::busy_chunk_reaches_the_scheduler_config` (default 256, 320 reaches `SchedulerConfig`, capped at the model chunk, validates) and `common::tests::busy_chunk_is_a_serve_flag_with_a_checked_range` (default 256, 320 parses, 15 and 2049 rejected, other commands refuse the flag). End to end: every batched server in the runs below logged `PrefillStart { chunk: 320, decoding: 7 }` for the arriving 4k prompt (`ab/analysis.json`: `big_prefill_chunk`), and the identity runs report `usage.prompt_tokens_details.prefill_chunk` 320 for 181 requests (2 ran alone at 2048). `serve --help` shows the flag with its range and `[default: 256]`.

## 2. A/B: 7 streams at 1k context plus one 4k prompt

Tool: `tools/cbchunkab.py` (scenario + analysis), `tools/cbchunkab_slot.sh` (slot driver). Each round, per arm: fresh `serve --max-batch 8 --busy-chunk C` with `INTARSIA_CB_TRACE=1`; a warm-up pass of the same shape (7 streams of 250 tokens plus a 4k prompt, discarded) and, with no idle gap, the measured pass: 7 streams (about 1k-token corpus prompts, 400 tokens, `ignore_eos`, temperature 0), and once all 7 are past 40 tokens plus 1 s the 4k prompt arrives (16 tokens). Prompts are seeded by round, so both arms of a round run identical prompts (stream prompts 851-1401 tokens, 4k prompt 3803-5082 tokens depending on the round; the harness has no tokenizer, it sizes by characters). Fresh server per arm with the arm order alternating (odd rounds 256 first, even rounds 320 first). 10 rounds per arm, one exclusive fleetq slot (#2181, 30 min), macmon every 100 ms for the whole slot.

Definitions. *Prefill-carrying gap*: a client-side gap between two consecutive tokens of one of the 7 streams that contains a `Chunk` trace event of the big prompt (ms-resolution server wall clock, 5 ms slack); this is the decode-row stall. *No-prefill ITL*: gaps of the 7 streams before the 4k send and after its last token, while the batch is 7 rows. *Throughput*: tokens delivered from the 4k send to the last completion (the 7 streams and the 4k prompt) over that time. *GPU busy*: macmon `gpu_usage` fraction, mean of the samples between the first chunk and the 4k first token.

| metric | 256 | 320 | 320 vs 256 |
|---|---|---|---|
| 4k TTFT, ms, median of rounds | 1993 | 1918 | -3.8% |
| 4k TTFT, ms, mean +- sd (rounds differ in prompt length) | 2063 +- 259 | 2011 +- 243 | |
| 4k TTFT paired, 320 minus 256 | | | mean -51 ms (-2.4%), median -44 ms (-2.2%), 8/10 rounds faster |
| 4k prefill wall (first chunk to first token), median ms | 1934 | 1846 | -4.6% (paired median -59 ms, -3.2%) |
| prefill chunks of the 4k prompt (median) | 16 | 13 | -19% |
| **decode-row stall, p50 ms** (n = 1161 / 937 gaps) | **121.9** | **146.3** | **+20%** |
| decode-row stall, p95 ms | 139.3 | 175.5 | +26% (paired p95 delta +31 ms, 320 worse in 10/10 rounds) |
| decode-row stall, max ms | 242.4 | 265.9 | +10% (round 8 outlier in both arms; paired max delta +29 ms, worse in 10/10) |
| decode ITL without prefill, p50 / p95 / max ms (reference, n = 20468 / 20982) | 37.2 / 40.4 / 110.2 | 37.1 / 41.6 / 111.0 | same |
| server Decode-to-Decode gap, chunk-classified, p50 ms (cross-check only, see below) | 122 | 147 | |
| decode tokens/s delivered by the 7 streams during the 4k prefill | 60.3 | 51.4 | -15% |
| aggregate throughput, tok/s, mean +- sd | 162.6 +- 5.2 | 162.3 +- 6.9 | -0.3 paired mean (320 higher in 6/10 rounds) |
| time from the 4k send to the last completion, s | 13.84 | 13.98 | +1.0% (paired mean +0.14 s, 320 shorter in 5/10) |
| GPU busy during the prefill | 0.840 | 0.854 | +1.6% (paired +0.014, 320 higher in 6/10) |
| GPU clock during the prefill, MHz mean | 1360 | 1382 | |
| GPU clock over the scenario, MHz mean / min sample | 1434 / 1093 | 1439 / 981 | |

How to read it. The no-prefill ITL is identical (37 ms, 27 tok/s per stream at 7 rows), so the arms differ only through the chunk. Decode-row stall is the iteration time: decode step plus one chunk. At 256 a chunk adds 84.7 ms to the 37 ms decode step (median over rounds), at 320 it adds 107.8 ms; per row that is 0.331 vs 0.337 ms, so there is no sub-linear region at the whole-iteration level. Fewer iterations (13 vs 16) cost 25% more per iteration, and the net gain in time to first token is 2 to 3%, inside the run-to-run spread of single rounds (-150 to +20 ms). The throughput cannot separate the arms: both do the same work and the difference of a few chunk times is below the noise of a 14 s window.

Noise notes. The two slowest rounds for both arms (2 and 3) coincide with GPU clock dips (`gpu_mhz` 1137-1309 in those pairs vs 1340-1450 elsewhere), so the pairing, not the absolute values, is the comparison; leaving rounds 2 and 3 out gives a mean paired TTFT delta of -49 ms. The "ITL max" of about 110 ms in the no-prefill reference is the same in both arms (stream start and end effects). The stall max includes one 242/266 ms outlier in round 8 (both arms). Whole-scenario GPU busy is 0.886 vs 0.889. macmon samples at 100 ms, 15-25 samples fall in each prefill window, so per-round GPU busy is a coarse number; read the means.

Idle effect: each measured pass runs directly after a warm-up pass of the same shape, no gap above a few hundred ms, so the weight re-wire after idle (about 170 ms) is outside the numbers. TTFT is measured per round inside one server and the arms are paired by round, as asked; TTFT is not compared across servers by `localeval`.

Raw: `docs/results/raw/2026-10-16-metal-cb-chunk320/ab/` (`round-*.json` with every token timestamp, `server-*.log` with the scheduler trace, `macmon.jsonl`, `analysis.json`, `summary.md` with the per-round table, `slot.log`). The pilot (1 round, #2173) is in `pilot/`.

## 3. Identity gates

### 3.1 Batched vs solo at B=8 (`tools/cbidentity.py`, slot #2182)

203-case set (`tools/prompts/cb_identity_cases.jsonl`: raw and chat, greedy and sampled, EOS, `min_tokens`, stop strings, `ignore_eos`, 20 cancellations, prompts 127-4500 tokens, three golden 4k prompts). References are solo servers (`--max-batch 1`) at `--prefill-chunk` 2048, 320 and 256, every case alone after a flush (cold prefill). Each batched request is compared with the reference whose chunk equals the `prefill_chunk` the batched server reports for it (`compare --ref ref2048 --ref-alt ref320 ref256`). Batched server: `--max-batch 8 --busy-chunk C`, reuse off (`INTARSIA_CB_NO_REUSE=1`) and on, 8 clients in flight (`--b 8`), streams mixed (113 streamed, 90 non-streamed, plus the 20 cancellations, always streamed), `--arrival-scale 0.25`, token ids compared.

| busy chunk | reuse | requests | pass | mismatch | chunk of the batched requests (2048 / C) | ids compared | cancels cut short | `cached` differs from the cold reference |
|---|---|---|---|---|---|---|---|---|
| 320 | off | 203 | **203** | **0** | 2 / 181 | 201 | 20/20 | 0 |
| 320 | on | 203 | **203** | **0** | 2 / 181 | 201 | 20/20 | 21 (all pass; the 17-token chat header) |
| 256 (control) | off | 203 | **203** | **0** | 2 / 181 | 201 | 20/20 | 0 |
| 256 (control) | on | 203 | **203** | **0** | 2 / 181 | 201 | 20/20 | 25 (all pass) |

(20 requests per run, the cancellation cases, report no chunk; 2 requests ran with nothing decoding and used 2048.) Server side after the 320/reuse-off run: decode passes of 1..8 rows 12 / 19 / 28 / 92 / 208 / 297 / 376 / 476, 912 prefill chunks, 20 cancelled. Both stream modes separately, other B values and arrival patterns were not repeated here (the task asked for B=8; the M1 gate covered B=2..8 at 256). The harness' own plan-difference warning (`max_batch`, `ops`) appears in each compare and is expected: reference servers are `--max-batch 1`. Raw: `raw/2026-10-16-metal-cb-chunk320/identity/`.

### 3.2 Serving-path invariance, 320 vs 2048 (chunk-golden harness, slot #2177)

The harness of `docs/results/2026-10-15-metal-chunk-golden.md` (`chunkgolden`, `Engine::generate` = `Seq::prefill_step`, greedy 32 tokens, `ignore_eos`, every sampled-step logits row hashed bitwise (FNV-1a 64 over the f32 bits), plus `Model::forward` hashes), built from this branch, run at `--prefill-chunk` 2048, 320 and 256. Prompts: the 33 of the earlier set (lengths 2, 3, 255, 256, 257, 258, 511, 512, 513, 514, 1000, 2047, 2048, 2049, 2050, 4095, 4096, 4097, 4098, 4200 and 13 golden prompts) plus 320, 321, 640, 641, 961, 3841: 39 prompts in all (321 and 641 are the L = k x 320 + 1 cases that use the no-1-row-tail split at 320; 3841 = 12 x 320 + 1).

| comparison | prompts | serving-path ids | serving-path logits rows bitwise | `Model::forward` last row | `Model::forward` all rows |
|---|---|---|---|---|---|
| 320 vs 2048 | 39 | 39/39 same | 39/39 identical (32 rows each) | 39/39 identical | 39/39 identical |
| 256 vs 2048 (control, reproduces the earlier result) | 39 | same | identical | identical | identical |
| 320 vs 256 | 39 | same | identical | identical | identical |

So 320 is bitwise equal to 2048 on the serving path for every prompt tried, including the lengths that put a chunk boundary at 320, 640, 3840 and one row after it. **The existing 2048 goldens cover 320.** (As before this covers the prompts tried, not a proof for every length.) Raw: `raw/2026-10-16-metal-cb-chunk320/golden/` (`compare-*.md` per-prompt tables, `chunk*.jsonl`).

## 4. Other checks

* `cargo check --workspace --all-targets`: pass. `cargo test -p intarsia-core`: pass (66 unit tests, includes the new one). `cargo test -p intarsia-cli`: pass (79, includes the new one). Release build with `--features metal`: pass.
* In slot #2177 (`INTARSIA_TEST_REAL_MODEL=1`): `cargo test -p intarsia-backend-metal --release` exit 0 and `cargo test --workspace --release --no-fail-fast` exit 0 (81 result lines ok, 0 failed). Logs in `golden/cargo-test-*.log`.

## fleetq jobs (all `--host endor --exclusive --mem 24`, all released by completion, none of mine left)

| job | what | exit |
|---|---|---|
| #2173 | pilot, 1 round | 0 |
| #2177 | chunk-golden 320/256/2048 + Metal and workspace tests | 0 |
| #2180 | identity gate, first submission, withdrawn while queued (runner killed, then `fleetq release` after the runner was dead) and resubmitted behind the A/B | cancelled |
| #2181 | A/B, 10 rounds per arm | 0 |
| #2182 | identity gate | 0 |

Binary: `intarsia-chunk320` (sha256 prefix `650d6c264d482493`, release `--features metal`, built from `152b744`; no `crates/` change since).

## Read of the trade-off

On this machine the only thing 320 buys is about 45-60 ms off a 4k prompt's roughly 2 s time to first token (2 to 3%), and it pays for it with 25 ms more stall per iteration for every decoding stream during the whole prefill (p50 122 to 146 ms, p95 139 to 176 ms). The design's M2 gate (running streams' p95 gap during a 4k arrival, 150 ms) is met at 256 and missed at 320 on the client-side pooled stall p95 (139 vs 176 ms; per-round p95 122-146 ms at 256 and 149-187 ms at 320). The server-side tails are not usable for this (see Limits). The motivating observation (per-layer prefill attention flat from T=128 to 320) is real but is a small slice of an iteration: the weight GEMMs dominate and grow with the rows, so the per-row cost of a chunk is flat (0.33 ms) and a bigger chunk only moves the same work into fewer, longer iterations. Aggregate throughput does not change. Keep 256. The knob can stay as an operator setting because 320 is exact: bitwise invariant against 2048 and clean on the identity gate.

## Limits

* One machine state per round; clock dips hit rounds 2 and 3 in both arms (paired deltas used). The 4k prompt length varies with the round (3803-5082 tokens), identical across the two arms of a round.
* The scenario has 7 streams at about 1k context; stall scales with batch and context, and the 320 vs 256 gap would change with them. Not varied here.
* The stall metric is client-side: gaps between the 7 streams' tokens, classified with the server's ms-resolution `Chunk` trace times (5 ms slack). That is the primary metric. The server-side Decode-to-Decode gaps are only a p50 cross-check (122 and 147 ms agree with the client 121.9 and 146.3); their tails do not (`analysis.json` `srv_stall_ms` p95 206 / 229 ms, `srv_itl_noprefill_ms` has 2-5 gaps of 110-180 ms per round). Cause found in round 256-1: the `Decode` trace line is printed before the decode compute and the chunk is enqueued asynchronously, so its cost lands in the next `Decode` gap, and a `Chunk` event printed in the same millisecond as the next `Decode` is classified into the wrong gap; the last chunk's gap (about 226 ms, chunk plus head and first token) also has no later `Chunk` event and falls in the 'no chunk' class. Those server-side tails are an artefact of the classifier and are not used.
* Identity: one stream mode (mixed) at B=8 per config; the cases never used 320 with speculation (`--spec ngram` was not part of this task).
* The harness `tools/cbchunkab.py` ran on this scenario only; no selftest beyond a mock-server smoke run.

## Files

`crates/intarsia-cli/src/cmd/serve.rs`, `crates/intarsia-cli/src/common.rs`, `crates/intarsia-models/src/load.rs`, `crates/intarsia-core/src/engine.rs`, `crates/intarsia-core/src/engine/batch.rs` (flag); `tools/cbchunkab.py`, `tools/cbchunkab_slot.sh`; `docs/results/raw/2026-10-16-metal-cb-chunk320/{idgate_slot.sh,golden_slot.sh,guard.sh,compare_chunks.py,ab/,pilot/,identity/,golden/}`.
