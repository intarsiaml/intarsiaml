# CHUNK4K: Metal 35B prefill chunk 2048 -> 4096 (2026-10-29)

Lane `chunk4k`, branch `lane-chunk4k/metal-prefill-chunk`, base `main` `492a8b35e` (#254), merged with `b26979329` (#255, CUDA-only)
before the measurements. endor (Apple M5 Max, 128 GB, macOS 27.0), Metal, Qwen3.6-35B-A3B NVFP4
(`~/Models/Qwen3.6-35B-A3B-NVFP4`). Raw data, logs and the scripts: [raw/](2026-10-29-chunk4k-metal/raw/). Every GPU run was in
a fleetq slot on endor (jobs at the end).

**Change:** `[tuning.metal] prefill_chunk = 4096` in `recipes/qwen3_5_moe.toml` (was 2048), and the four Metal 35B accuracy
files that carry `prefill_chunk` re-recorded at 4096. Nothing else in the engine changed. CUDA, HIP and the 8B recipes
(`qwen3.toml`: 2048 on Metal) are untouched. Source of the idea: MPRE section 4 (`2026-10-29-mpre-metal-prefill.md`), where a
third arm at `--prefill-chunk 4096` gave +6.2 % at 4k and 16k.

**Result:** greedy ids are bitwise identical (12 of 12 comparisons). Served 35B prefill at 4k is about +7.5 % (best 2048 start
5451 against best 4096 start 5860; paired median +11 %, every one of the 6 paired rounds at least +7.3 %) and at 16k about +2 to
+6 % and noisy (best against best +2 %, paired median +5.5 %, one of 6 rounds -1.9 %). The 6-round medians in the table (+17 % at
4k, +9 % at 16k) are larger because the 2048 arm is slow in some server starts. Decode (1 stream at 1k / 4k / 16k) and the 1 / 4 / 8 /
16-stream burst are inside noise. The ids gate and the re-record ran on `492a8b35e` plus this change; #255 touches no Metal code, and the
A/B and the final tests ran on the merged tree.

## 1. Served A/B, 2048 vs 4096

One binary (`.lane/intarsia-metal`, built from the merged branch), arms differ only by the flag: `4096` is the recipe default (no
flag), `2048` is `--prefill-chunk 2048`. A fresh server per arm and per round (model load 2-3 s), arm order rotated each round
(r1 4096 first, r2 2048 first, ...), localeval inside one fleetq slot. Prefill tok/s is localeval's TTFT-based figure; the
"16k" prompt is 16656 tokens (localeval's calibration), "4k" is 4090, "1k" is 991.

Single stream (default serve, no `--max-batch`, `--max-seq 20480`, `--sizes 1024,4096,16384 --gen 256 --reps 3`): each cell is the
median of 3 reps inside one server; the planned 3 rounds were followed by 3 extra rounds (job #7127) because the 2048 arm's 4k
number moved between server starts (4511 / 5107 / 5451 tok/s, tight inside each start), while the 4096 arm did not (5859 / 5699 /
5858).

| size | metric | 2048, rounds 1-6 | 4096, rounds 1-6 | median 2048 | median 4096 | change | paired changes per round |
|---|---|---|---|---:|---:|---:|---|
| 1k | prefill tok/s | 4582 / 4583 / 4593 / 4599 / 4547 / 4534 | 4590 / 4496 / 4570 / 4585 / 4580 / 4542 | 4582 | 4575 | -0.2 % | +0.2 -1.9 -0.5 -0.3 +0.7 +0.2 |
| 1k | decode tok/s | 124.3 / 124.8 / 124.2 / 123.9 / 123.1 / 124.3 | 124.3 / 121.1 / 124.5 / 124.8 / 124.5 / 123.8 | 124.3 | 124.4 | +0.1 % | 0.0 -3.0 +0.2 +0.7 +1.1 -0.4 |
| **4k** | **prefill tok/s** | 4511 / 5107 / 5451 / 5448 / 4742 / 4756 | 5859 / 5699 / 5858 / 5844 / 5592 / 5270 | 4932 | 5772 | **+17.0 %** | +29.9 +11.6 +7.5 +7.3 +17.9 +10.8 |
| 4k | decode tok/s | 119.6 / 119.5 / 121.3 / 120.2 / 119.1 / 120.0 | 121.1 / 120.1 / 121.4 / 121.7 / 116.7 / 119.0 | 119.8 | 120.6 | +0.7 % | +1.3 +0.5 +0.1 +1.2 -2.0 -0.8 |
| **16k** | **prefill tok/s** | 3702 / 4256 / 4228 / 4361 / 3768 / 3800 | 4378 / 4380 / 4453 / 4279 / 4418 / 4020 | 4014 | 4379 | **+9.1 %** | +18.3 +2.9 +5.3 -1.9 +17.3 +5.8 |
| 16k | decode tok/s | 109.0 / 108.3 / 109.2 / 108.8 / 108.6 / 108.7 | 108.7 / 109.0 / 109.1 / 108.3 / 108.2 / 108.4 | 108.8 | 108.6 | -0.2 % | -0.3 +0.6 -0.1 -0.5 -0.4 -0.3 |

Planned 3 rounds only: 4k prefill 5107 -> 5858 (+14.7 %), 16k 4228 -> 4380 (+3.6 %), decode within +-1.3 %
(`raw/ab/analysis-planned-3-rounds.txt`, `raw/ab/analysis-single-all-rounds.txt`). The best 2048 4k start (5451) against the best 4096
start (5860) is +7.5 %, which is the figure to compare with MPRE's +6.2 %: the 2048 arm is slow in some server starts for a reason
this lane did not find, so the medians above overstate the steady gain at 4k. The 16k prefill is noisier (reps inside a start
spread 4200-4640 on both arms); one of six paired rounds is negative (-1.9 %).

Burst (`--max-batch 16`, `--max-seq 8192`, 1024-token prompts, `--gen 512 --reps 1`, aggregate decode tok/s; the 1-stream row is
the same server's single request; 3 rounds, no extra rounds):

| streams | 2048 r1 / r2 / r3 | 4096 r1 / r2 / r3 | median 2048 | median 4096 | change |
|---:|---|---|---:|---:|---:|
| 1 | 122.9 / 124.9 / 124.5 | 124.6 / 123.6 / 124.1 | 124.5 | 124.1 | -0.3 % |
| 4 | 243.7 / 254.9 / 259.0 | 258.8 / 257.0 / 260.0 | 254.9 | 258.8 | +1.5 % |
| 8 | 306.8 / 334.8 / 336.0 | 329.1 / 324.7 / 331.8 | 334.8 | 329.1 | -1.7 % |
| **16** | 332.0 / 347.4 / 351.1 | 347.2 / 349.6 / 351.5 | 347.4 | 349.6 | +0.6 % |

The burst prompts are 1k tokens, one chunk under both settings, so this table is a chunk-neutral control. The 8-stream -1.7 % is
spread inside the 2048 arm itself (306.8 to 336.0; its r2 and r3 are higher than every 4096 round), within endor's decode noise
of about 5 %, so no cell is slower beyond noise. **The 16-stream number on current main (after #253 / #254 / #255) is about 347-350 tok/s aggregate**
(per-stream about 22-24 tok/s), which MDEC had not re-measured after #253; MDEC's table had 296-305 at `--max-batch 16` before.

## 2. Gates

| gate | result |
|---|---|
| greedy ids, 256 tokens, `--chat --no-think --ignore-eos --show-ids`, prompts `scripts/mpf/ids-prompts/{short-0, 4k-0, 4k-1, 16k-0}` (41 / 3870 / 3993 / 15481 prompt tokens; the 4k prompts are 2 chunks at 2048 and 1 at 4096, the 16k one is 8 against 4) | 4096 (recipe default, `--repeat 2`) == `--prefill-chunk 2048` == `--spec mtp --spec-k 2` on 4 of 4 prompts; the two repeats of the 4096 arm equal each other. 12 of 12 comparisons identical (`raw/ids/`, `raw/w1-job.log`) |
| `plan show` 4096 vs `--prefill-chunk 2048` (35B, `--max-seq` 8192 and 20480, `--mem-budget 40`) | only: `scratch_bytes` of three ops (`matmul_fp8_nax` 41.9 -> 50.3 MB, `moe_experts_bf16` 84.0 -> 167.9 MB, `gdn_delta_rule_bf16_pre` 34.1 -> 68.2 MB; total +83.95 MB), `prefill_chunk` 2048 -> 4096, `prefill_chunk_source` (an artifact: the old arm was forced with the flag), and the audit row list gained 2049, 4095 and 4096 (chunk-boundary admission cases) (`raw/plan/diff-*.txt`) |
| memory | process peak footprint of `generate` (`/usr/bin/time -l`): 4096 20.61-20.76 GB, 2048 20.51-20.76 GB across the four prompts, no systematic difference (the +84 MB scratch is below the spread); the loaded weights are 20.83 GB. No gate by unified-memory size added: the 35B needs at least a 24 GB-class Mac either way and the recipe only applies to it. See the one case where it showed, in the test row below |
| other recipes | `plan show` on the other local checkpoints: Qwen3-8B 2048 (recipe qwen3), Llama-3.1-8B 256, Qwen2.5-7B 256, Qwen3.8-27B-Quark 256, all unchanged; MiniCPM-V-4.7 and Kolibri-1 do not load on Metal on main (as in MPRE). Only the `qwen3_5_moe` recipe changed |
| `batch_recurrent` (`INTARSIA_REQUIRE_GPU=1`) | 14 passed. It uses synthetic fixtures, so it does not exercise the recipe chunk |
| `batch_metal_real` 35B (`INTARSIA_TEST_REAL_MODEL=1`) | `real_35b_batched_decode_is_bitwise_the_solo_step` and `real_35b_batched_engine_equals_the_local_engine` pass; `real_lfm2_batched_decode_is_bitwise_the_solo_step` fails only because `INTARSIA_LFM2_DIR` is unset (as in MPRE) |
| `batch_metal_real` `--ignored real_35b_batched_engine_long_prompts_equal_the_local_engine` (2.6k / 4k / 16k prompts, 2 and 4 slots) | passes in its default invocation after the follow-up (section 5): 10 of 10 requests IDENTICAL to the LocalEngine run, 0 DIFFERS, with the recipe's busy chunk 2048 against the solo 4096. In the first PR commit it failed in one process (`prompt of 16384 tokens does not fit the context window of 16377 tokens` in the fourth engine); the cause was not the scratch (section 5) |
| prefix reuse equals cold | the 35B prefix-reuse gate is `qualify`'s `acc-4k`, which runs `accuracy --mode all` at the plan chunk (4096) and `--mode reuse` at chunk 1024. At 4096 `--mode all`: continue case on chat_reuse_4k oracle-relative PASS (warm/cold 0.71x), restore / aligned bit-identical; long_docs_4k and long_code_4k are skipped there because their 4064 tokens cannot cross a 4096 boundary (judged 1, skipped 2, as on HIP). The chunk-1024 `--mode reuse` run judges all three and FAILS on `long_code_4k continue` (streams diverge at token 3, "the golden's prompt token ids differ from the ones it was made for", no committed oracle applies): **this failure is not caused by the change**, the same command with a binary built from `main`'s recipe (2048) fails identically (`raw/control/`), and it never reads the recipe chunk. `qualify` therefore reports `acc-4k` FAIL both before and after. The PRC probe tests (`prefix_reuse_real`, `reuse_rule_resumes_with_cold_bits` ...) need a dense checkpoint and the 8B recipe is unchanged, so they were not re-run |
| `cargo test --workspace --release --locked` (endor, after merging #255, no GPU-required env) | **1105 passed, 0 failed, 28 ignored** (`raw/tests/post-merge/ws.log`). A first run in the same job as the real-model tests left `INTARSIA_REQUIRE_GPU=1` exported and reported 15 failures that are all the CUDA / real-model tests refusing to skip (`gpu_tests_were_not_built`, `real_*` on a Mac); that log is kept as `raw/tests/w3-ws-REQUIRE_GPU-env-leak.log` and is a script error, not a result |

## 3. Re-recorded baselines

The project's own recording tool, the gc2 procedure (`2026-10-22-gc2-metal-35b-chunk2048/scripts/run.sh`): the old files moved
aside, `qualify --backend metal ... --skip kernels --skip gate --skip bench --skip bench-long --skip ab --skip tp` (q-0) records at
the recipe chunk, the outputs copied into `golden/baselines/`, then a q-1 verdict run against them. Changed files (all
`prefill_chunk` 2048 -> 4096 in their fingerprint, plus the numbers recorded by the current build):

* `golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.metal.baseline.json` (short golden, `acc-short`)
* `golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.metal.envelope.json` (noise envelope)
* `golden/baselines/qwen36-35b-a3b-nvfp4-vllm-4k-top20.metal.baseline.json` (4k golden, `acc-4k`)
* `golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.metal.selfgolden.safetensors` (CPU-oracle self-golden)

Why: each of them records `prefill_chunk` and the accuracy tool refuses to use it at another chunk (q-0 printed `MISSING-BASELINE:
recorded at prefill chunk 2048, this run uses 4096`). `git status golden/baselines` shows only these four; the CUDA, HIP and
HIP-TP2 files are untouched. Caveat: the old files were recorded on 2026-10-22, before MPRE's NAX kernels and MDEC; the new ones
are recorded by the current build, so their diff also contains those kernel changes, which cannot be separated from the chunk.
The `acc-long` golden has no recorded baseline file on Metal (it is judged by the KL floor).

q-1 verdict against the new baselines (`raw/qualify/q-1-verdict.log`): `kernels` PASS (48 kernels, 924 cases, which include the new
2049 / 4095 / 4096-row cases), `gate` PASS, `acc-short` PASS (prefill top-1 97.85 % KL 7.93e-3; decode 97.50 % / 1.24e-2),
`acc-oracle` PASS (prefill top-1 96.04 % KL 1.07e-2), `bench` PASS (decode 123.7 tok/s), `acc-long` PASS (prefill 94.20 % / 4.19e-2;
decode 98.44 % / 9.54e-3), `acc-4k` FAIL for the pre-existing reason in the gate table (its 4096 `--mode all` part passes: prefill KL
1.254e-2 over scored rows against the 3.0e-2 floor, per-prompt gate ok, prefix-reuse gate PASS, golden gate PASS). The q-0 record run: 235 s, q-1: 298 s.

## 4. Caveats

* **Decode stall beside a long prefill**: addressed by `busy_chunk = 2048` (section 5); the stall itself was derived from the prefill rate, not measured (no ITL harness).
* The 2048 arm's 4k prefill varies between server starts (4511-5451 tok/s) and the 4096 arm's does not, so the +17 % median is
  partly a variance effect; the best-against-best figure is +7.5 %.
* Prefix reuse at 4096: the 4064-token documents of the 4k golden cannot be judged by `accuracy --mode all` (see gates).
* `localeval` logs each run to unanimis (title `chunk4k-r*-...`).

## fleetq jobs (endor)

| job | what | flags |
|---|---|---|
| #7113, #7115, #7117, #7123 | builds (CLI, workspace tests, CLI again, post-merge) | `--load cpu` |
| #7114 | `plan show` 2048 vs 4096 | default |
| #7116 | first window-1 attempt: the CLI binary had been overwritten by the test build and had no Metal backend, goldens were not linked; nothing recorded (baselines restored with `git checkout`) | `--quiet gpu,cpu` |
| #7118 | window 1: plan show on all models, greedy-ids gate, q-0 record, q-1 verdict (11 min) | `--quiet gpu,cpu` |
| #7120, #7121 | `acc-4k` reuse control: main-recipe binary vs branch binary (#7120 was a no-op: `qualify` has no `--prefill-chunk`) | `--quiet gpu,cpu` |
| #7122 | `batch_recurrent`, `batch_metal_real` 35B (one failure: LFM2 dir; long-prompt memory failure), first workspace run (env leak) | default |
| #7124 | post-merge: long-prompt identity 3 cases, workspace tests 1105 / 0 | default |
| #7125 | A/B attempt 1: `serve` subcommand missing in the script, servers exited at once, no data | `--quiet gpu,cpu` |
| #7126 | window 2: served A/B, 3 rounds, single stream and burst (10 min) | `--quiet gpu,cpu` |
| #7127 | 3 extra single-stream rounds (4.3 min) | `--quiet gpu,cpu` |
| #7129 | `docs_env_vars` test | `--load cpu` |
| #7130 | final binary (recipe comment edited after the re-record) rebuilt; `qualify` acc-short grades against the installed baseline: PASS (97.85 % / 7.93e-3, not MISSING-BASELINE) | `--quiet gpu,cpu` |

## 5. Follow-up: busy chunk 2048 and the long-prompt test (coordinator request)

**Busy chunk.** `[tuning.metal] busy_chunk = 2048`. Where it is used: `Scheduler::prefill_plan` (`crates/intarsia-core/src/scheduler.rs`,
the `chunk` of a request's first prefill step: `if n_decoding > 0 { cfg.busy_chunk } else { cfg.solo_chunk }`; the same rule in
`pack_body_rows`) takes the busy chunk only when a stream is decoding at that moment, and the solo chunk otherwise. The recipe value
reaches the scheduler through `load.rs` (`recipe.resolve_busy_chunk(backend)` into `BatchedOptions::busy_chunk`, clamped to the solo
chunk by `scheduler_config`). Nothing else reads it: the single-sequence engine (default serve, no `--max-batch`) and an idle batching
server prefill at 4096. A prompt that starts prefilling beside a decoder keeps 2048-row steps for its whole prefill (the chunk is
fixed on its first step).

| check | result |
|---|---|
| server log at `--max-batch 4` | `prefill chunk 4096 (recipe qwen3_5_moe, metal)` and `continuous batching: up to 4 sequences (--max-batch), busy prefill chunk 2048` (`raw/followup/w5-job.log`) |
| single-stream 4k prefill (default serve, `--sizes 1024,4096 --gen 128 --reps 3`, order default / `--prefill-chunk 2048` / default) | 5844, 5381, 5548 tok/s: +8.6 % and +3.1 % over 2048 (the 2048-start variance of section 1 again); 1k 4568 / 4585 / 4579 (flat), decode 124.3 / 124.9 / 124.4 and 120.9 / 121.2 / 118.8 (flat) |
| greedy ids (256 tokens, `generate`, short-0 and 4k-0) | identical to the saved 2048 runs. `generate` is the single-sequence path, so it does not exercise `busy_chunk`; the busy path is checked by the next row |
| batched engine vs LocalEngine with busy 2048 and solo 4096 (long-prompt test, 2.6k / 4k / 16k prompts, 2 and 4 slots, 3 cases) | 10 of 10 requests IDENTICAL, so on Metal the busy schedule differs from the solo one without changing a bit (mixed iterations 2 / 12 / 12 against 1 / 6 / 6 at busy 4096: the long prefills now take twice as many, half-size, steps beside the decoder) |
| `plan show` (35B, `--max-seq 20480`) against the commit without `busy_chunk` | no diff (0 lines): `plan show` reports only the solo `prefill_chunk`; the busy chunk is an engine option and shows in the serve log line above |

The decode stall itself (the worst step of a decoding neighbour while a long prompt prefills) was not measured: there is no ITL
harness in this lane. It is back to the 2048-row step it was before this PR, by construction.

**Long-prompt test.** The shortfall was not the larger scratch and not a race. Instrumenting the test (`drop_and_settle`: drop the
engine, wait for `MetalBackend::info().free_mem` to return) showed that the free memory never came back: 110094 MiB at the start,
87755 after the reference engine was dropped, 65416 after the 2-slot engine, 43077 after the next one, exactly 22339 MiB (the
weights) per engine (`raw/followup/w4-long-default-first-failure-with-settle-helper.log`). The cause is `host::HostCache` in
`crates/intarsia-backend-metal`: it maps every model file's mapping address to a whole-file no-copy buffer plus an `Arc` of the
mapping, and nothing ever removed an entry, so a backend that loads a second model in the same process kept the first one's weights
allocated, and the next engine's free-memory plan started from that much less (at the 2048 chunk the fourth engine still fit by a
few tokens; at 4096 its larger workspace pushed it 7 tokens under 16384). Fix: `HostCache::prune()` drops the entries whose mapping
nothing else holds (`Arc::strong_count == 1`; every `Buffer` handed out keeps its own `Arc`), called on a cache miss and at the start
of `MetalBackend::info()`, which is what the memory plan reads. A process with one model is unaffected. After the fix `free_mem` is
back at 110094 MiB 0.1 s after each drop and the test passes in its default invocation (66.8 s, 10 of 10 IDENTICAL,
`raw/followup/w7-long-default-final.log`); the test now calls `drop_and_settle` between engines (it reports the free memory and
waits at most 10 s). `INTARSIA_LONGENG_CASE` is removed, with its `docs/tuning.md` row. Other checks: `cargo test -p
intarsia-backend-metal` 166 passed, 0 failed, 1 ignored; `docs_env_vars` 2 passed. The full workspace run was not repeated after this
fix (it predates it, 1105 / 0); the change is 12 lines in one backend.

follow-up fleetq jobs: #7131 (first test run with the settle helper, still failing), #7132, #7137, #7138 (builds; #7137 failed to
compile: a misplaced `impl`), #7133 (busy-chunk checks, long test, ids, serve), #7139 (long test default invocation after the
prune, backend-metal tests), #7145 (final long test, `docs_env_vars`). GPU time of the follow-up: about 13 min.
