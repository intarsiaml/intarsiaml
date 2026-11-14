# mtp-sampled: sampled requests draft with MTP in the batching engine, seed-exact

Lane `mtp-sampled`, branch `mtp-sampled` from origin/main `48c7e9f39`, 2026-11-13. Source: candidate 2 of `docs/research/2026-11-13-tensorfold-1.2.8-review.md` (TensorFold PR #2455, commit `18f556378`). Model under test: Qwen3.6-35B-A3B NVFP4 on spark1 (GB10, CUDA), fleetq owner `claude-mtp-sampled`.

## Problem

With `--max-batch` > 1 and `--spec mtp`, only greedy requests drafted (`Seq::build`: `spec_on` needs `temperature <= 0`). The 35B's `generation_config` default is temperature 1.0, so default client traffic never drafted on a batched server (`docs/results/2026-11-12-ab35b-cutover-followup.md`, "Batched B never drafts on sampled requests"). The single-sequence engine has sampled speculation (`INTARSIA_SPEC_SAMPLED=match`), but its draws go through a different sampling routine (`Dist::pick` on a f64 uniform), so a seed gives different text with it on.

## Design

The rule is TensorFold #2455's: drafts are the MTP head's greedy chain, and a draft is kept only where it equals the target's own keyed draw for that position. Here the "keyed draw" is literally the plain step's draw:

- A sampled request under `--spec mtp` in the batching engine gets `spec_keyed` (`Seq::enable_keyed`, called by `BatchedEngine::submit` after `Seq::build`). Same exclusions as greedy drafting: no `trace_logits`, no constraint, and (in `plan_step`, unchanged) no draft inside a `min_tokens` window that masks stop ids.
- The draft step (`Seq::keyed_draft`, `crates/intarsia-core/src/engine/keyed.rs`) runs `Model::mtp_draft` (greedy chain, as for a greedy request) and verifies `[tok, d_1..d_k]`. The verify rows come back as logits: `Model::forward_verify_logits` for a solo pass on the sequence's own slot, the new `Model::forward_verify_batch_logits` when the batching engine verifies several sequences in one ragged pass (`INTARSIA_BATCHED_VERIFY=1`; the pass returns logits rows for the sampled sequences and greedy tokens for the others).
- Acceptance (`Seq::keyed_accept`): row `i` holds the logits of generated token `generated + 1 + i` given that `d_1..d_i` were kept. The sequence calls `Seq::token_from(StepOut::Logits(row), generated + 1 + i)`, the function a plain step's output goes through (temperature, top-k, top-p, one `SplitMix64` draw), emits that token, and goes on to the next row only if the token equals the next draft. Rows behind the first mismatch are never drawn from and are rolled back (`spec_cut`), exactly as after a greedy verify pass.

Why this is exact for a seed, not only in law:

1. A verify row's logits are bitwise the logits of the single decode step at that position (`Model::forward_verify` contract; on CUDA `--spec mtp` runs verify attention per row for that reason, `INTARSIA_CUDA_VERIFY_ATTN_PER_ROW`). The batched plain step's row is bitwise the solo one (the batching engine's identity gates).
2. The request's RNG yields one value per generated token, in order: a plain step takes one draw (`Sampler::sample` or `Sampler::sample_topk`), a verify row that decides a token takes one draw, a row behind a mismatch takes none. So token `n` is always drawn with the `n`-th value of the request's generator, from the same logits bits, through the same sampler. The draw depends on (seed, request, position) only.
3. A plain batched step on CUDA asks the device for the top-k candidates (`HeadMode::LastTopK`) while a verify row is sampled from the whole row on the host; `Sampler::sample_topk` returns the token `Sampler::sample` returns for the same draw (existing test `topk_candidates_sample_like_the_whole_row`).

So which steps drafted (the scheduler's batch-size gate, the drafter gates, the other sequences, arrival order) changes speed only. Greedy requests are untouched: they never get `spec_keyed`.

Scheduler interface (`crates/intarsia-core/src/scheduler.rs`): `SeqStep::Verify(ids, want)` and `BatchModel::verify_step(&[(slot, ids, want)])` carry an optional head mode per sequence; the pass answers `Verified::Greedy(tokens)` or `Verified::Rows(outputs)`. No kernel change. The verify logits downloads now use the pooled buffers of a decode step's logits row (fresh megabyte allocations page-fault on GB10).

Knob: `INTARSIA_BATCHED_MTP_SAMPLED` (`docs/tuning.md`), `0` = sampled requests take plain steps in the batching engine (main). The startup line `batched MTP: sampled requests (temperature > 0) ...` names the setting.

Not touched: `model/mtp.rs`, `prefix_store.rs`, the store + spec refusal in `engine/batch.rs` (lane `mtp-store`). `engine.rs` gets one field, its initializer, one `prepare_with` arm and `mod keyed;`; everything else is in the new `engine/keyed.rs`.

## Gates

### CPU (this machine, debug build)

| Gate | Result |
|---|---|
| `cargo test -p intarsia-core --lib` | pass, 201 passed, 2 ignored |
| `cargo test -p intarsia-core --test scheduler` | pass, 45 passed (new: `seeded_sampled_drafting_equals_drafting_off_in_any_batch`) |
| `cargo test -p intarsia-models --test batch_engine` | pass, 9 passed |
| `cargo test -p intarsia-cli --test docs_env_vars` | pass, 2 passed (after the `docs/tuning.md` row) |
| `cargo test -p intarsia-models --test mtp batched_` and `sampled` (not in the lane's list; where the real engine's batched MTP tests live) | pass, 7 + 2 passed (new: `batched_mtp_sampled_requests_equal_spec_off_for_a_seed`, `batched_verify_logits_rows_are_the_decode_steps_rows`) |

What the new tests cover:

- `tests/scheduler.rs`, `seeded_sampled_drafting_equals_drafting_off_in_any_batch` (mock model and sequence, now with a drafter and `verify_step`): 8 workloads of 16 requests (79 seeded sampled, 49 greedy; `min_tokens`, stop sequences, EOS), each run with drafting off (which equals every request's solo reference) and then in 14 drafting configurations: `--max-batch` 1, 2, 4, the drafting gate at widths 2, 3, 5, solo and batched verify passes, draft depths 1 to 4 with one draft in three wrong, and the same requests arriving in reverse order. Every request's events are identical to the drafting-off run. Coverage in one run: 26,894 of 54,566 drafts accepted, 2,844 verify passes holding greedy and sampled sequences together, 2,173 holding two or more sampled ones. Negative control: a sequence that spends a draw on rows behind its first rejected draft changes 67 of the 79 sampled requests' outputs.
- `tests/mtp.rs`, `batched_mtp_sampled_requests_equal_spec_off_for_a_seed` (the real `Seq` and `Model` on the hybrid CPU fixture: GatedDeltaNet + MoE + gated attention + MTP head): 4 seeded sampled requests (temperature 1.0 and 0.6, with and without top-k / top-p, one with stop ids behind `min_tokens`) next to 2 greedy ones equal their spec-off single-sequence runs (ids, finish reason, counts) with 4 and 2 slots, drafting at every width or only at B <= 2 or 1, solo and batched verify, k = 1..3, scripted drafts (every pass accepts all, some or none) and the head's own, the sampled requests alone, pairs, and the reverse arrival order. With `mtp_sampled` off the sampled requests give the same ids from 0 verify passes.
- `tests/mtp.rs`, `batched_verify_logits_rows_are_the_decode_steps_rows`: the rows `Model::forward_verify_batch_logits` returns are bitwise the logits of plain decode steps fed the same tokens, for a lone sequence (the solo pass) and for 2 to 3 sequences in one ragged pass, with rows returned only for the sequences that asked.

### CUDA, real model (spark1, fleetq #9281, 13 min, `--quiet gpu`, vLLM up)

Build `mtp-sampled-ad2606054` (fleetq #9280; an earlier build of commit `b6ea108c4`, #9277, was not measured). `scripts/mtp-sampled/gates.sh`: five servers one after the other (`serve --backend cuda --max-seq 32768 --mem-budget 40 --max-batch 4 --profile-every 0`), each answering the same 8 prompts x 8 cases through `/v1/completions` with `return_token_ids`, `ignore_eos`, fixed seeds, 120 to 160 tokens each (1160 tokens a case). Cases: temperature 0.6 and 1.0 with top_k 20 / top_p 0.95 (the 35B's `generation_config`) at 1, 2 and 4 concurrent requests (every other group arrives staggered by 0.35 s, so batches form differently), temperature 1.0 over the whole vocabulary (top_k 0, top_p 1) at 2, and 4 greedy + 4 sampled requests at 4.

| Arm | Server | Requests identical to `ref` at 1 stream |
|---|---|---|
| `ref` | `--spec off` | 64 of 64 (its own 2- and 4-stream cases equal its 1-stream case) |
| `off` | `INTARSIA_BATCHED_MTP=4 --spec-off-at 5`, `INTARSIA_BATCHED_MTP_SAMPLED=0` (main's behaviour) | 64 of 64 |
| `on` | `INTARSIA_BATCHED_MTP=4 --spec-off-at 5`, sampled drafting on, solo verify passes | 64 of 64 |
| `onv` | the same with `INTARSIA_BATCHED_VERIFY=1` (ragged verify pass, logits rows) | 64 of 64 |
| `default` | shipped defaults (MTP while one sequence decodes) | 64 of 64 |

**Pass: 320 of 320 request runs give the ids of the spec-off server's single-stream run**, at B = 1, 2 and 4, drafting on or off (`raw/ids-compare.txt`, `raw/ids-*.json`). The `onv` arm drafted in every step at every width (451 passes at temperature 0.6 and 481 at 1.0 whether 1, 2 or 4 streams ran), so the batched logits path is covered at B = 2 and 4, not only B = 1.

Acceptance (drafts accepted / drafted, k = 2; `raw/gates.log`):

| Sampling | Verify passes | Accepted / drafted | Tokens per pass |
|---|---|---|---|
| temperature 0.6, top_k 20, top_p 0.95 | 451 | 695 / 901 = **77.1 %** | 2.54 |
| temperature 1.0, top_k 20, top_p 0.95 | 481 | 667 / 959 = **69.6 %** | 2.39 |
| temperature 1.0, whole vocabulary | 488 | 651 / 973 = 66.9 % | 2.33 |
| greedy requests in the mixed case, for reference (`off` arm: only they draft) | 20 | 29 / 39 = 74.4 % | 2.45 |

Wall time of the 1-stream cases (8 requests one after the other, prefill included): 14.3 s plain (`ref`, `off`) against 11.7 s with sampled drafting (`on`, `onv`, `default`) at both temperatures. Memory: lowest MemAvailable during any arm 56.3 GB, 0 watchdog kills, earlyoom active.

## Paired speed A/B (spark1, fleetq #9283, 23 min, `--quiet gpu,cpu`, vLLM up and idle)

`scripts/mtp-sampled/ab.sh`. **A**: `INTARSIA_BATCHED_MTP_SAMPLED=0` (main: sampled requests take plain steps). **B**: sampled drafting on. Both `serve --backend cuda --max-seq 32768 --mem-budget 40 --profile-every 0 --max-batch 2` (and 4), shipped defaults otherwise (MTP k = 2 while one sequence decodes), prefix store off. Order A B B A per `--max-batch`, a fresh server per visit (10 visits with the two extra arms below). Workload (`scripts/mtp-sampled/speed.py`, localeval 0.4.0's streaming `Client` and timing definitions): 8 chat prompts of about 25 prompt tokens, temperature 1.0 (top_k 20 / top_p 0.95 from the checkpoint's `generation_config`), a fixed seed per prompt, thinking at the server default, exactly 384 tokens each (`ignore_eos` + `min_tokens`), run in groups of 1, 2 or 4 concurrent streams after one unmeasured warm-up request. Decode tok/s = (completion tokens - 1) / decode window per stream; combined = the sum over a group's streams, median over groups. The reply text of every request is identical on all four visits (sha256, 8 of 8 in every row), so the arms are compared on the same token streams. spark1's vLLM reported 0 running requests before and after every visit (20 of 20 checks).

| `--max-batch` | streams | A per stream, tok/s (two visits) | B per stream (two visits) | **B / A** | A combined | B combined | ITL p99 ms, localeval definition, A / B (two visits each) | chunk gap p99 ms, A / B |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 84.44 / 83.81 | 109.29 / 109.53 | **1.301** | 84.44 / 83.81 | 109.29 / 109.53 | 111, 101 / 101, 147 | 80.5, 80.5 / 83.8, 90.1 |
| 2 | 2 | 68.97 / 67.92 | 68.03 / 68.56 | 0.998 | 138.15 / 135.90 | 135.92 / 137.52 | 151, 139 / 157, 103 | 86.3, 85.9 / 86.8, 83.7 |
| 4 | 1 | 84.77 / 84.99 | 108.69 / 107.08 | **1.271** | 84.77 / 84.99 | 108.69 / 107.08 | 97, 92 / 149, 108 | 78.9, 80.1 / 91.0, 86.1 |
| 4 | 2 | 68.80 / 69.10 | 68.14 / 67.40 | 0.983 | 137.73 / 138.13 | 136.33 / 134.61 | 156, 107 / 155, 122 | 86.2, 82.6 / 86.6, 82.7 |
| 4 | 4 | 49.62 / 49.82 | 49.76 / 48.74 | 0.991 | 198.80 / 199.44 | 199.08 / 194.94 | 170, 164 / 187, 155 | 95.0, 92.0 / 101.9, 88.6 |

- **One stream on a batched server: +30.1 % (`--max-batch 2`) and +27.1 % (`--max-batch 4`)**, 84 -> 108 to 110 tok/s. Acceptance at temperature 1.0: 1822 of 2471 drafts = **73.7 %**, 2.47 tokens per verify pass, 1237 passes for 8 x 384 tokens (identical on all four B visits, as the token streams are). Per-request range in B is 99 to 116 tok/s, it follows the text.
- **Two and four concurrent streams: unchanged** (0.998, 0.983, 0.991). With the shipped `batched_mtp = 1` nothing drafts while two or more sequences decode (4 and 2 passes in those rows, at group ends), so there is no mechanism for a difference beyond those tails; the A visits alone spread 67.9 to 69.1 at 2 streams. The one cost that exists there is the MTP head catching up over its queued rows when a group's last stream is left alone for its final tokens.
- **ITL p99**: no clear change. By localeval's definition (chunk gap divided by the chunk's tokens) the visits spread 92 to 187 ms in both arms, larger than any A/B difference. The raw gap between chunks at p99 is 79 to 95 ms in A and 83 to 102 ms in B (1 stream: 80 -> 84 to 91 ms). The median chunk gap in B at 1 stream is 0 ms: a verified run arrives as a burst, as with greedy MTP. TTFT is unchanged (0.072 s at 1 stream in both arms).

Extra arms, 2 streams on `--max-batch 2`, one visit each (does sampled drafting pay while two sequences decode?):

| Arm | per stream tok/s (min-max) | combined | vs A (68.45 / 137.0) | MTP passes, acceptance | chunk gap p99 ms (max) |
|---|---|---|---|---|---|
| `mb2-N2`: `INTARSIA_BATCHED_MTP=2`, solo verify passes | 66.75 (65.85-67.95) | 133.89 | 0.975 | 88, 165 / 176 (the busy gate stops most drafting) | 84.9 (108) |
| `mb2-V2`: `INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1` | 69.42 (63.11-71.83) | 139.01 | 1.014 | 1237, 73.7 % | 97.2 (191) |

Neither is a win worth a default: solo verify passes beside a batched step cost 2.5 %, the batched verify pass gains 1.4 % (one visit, inside the noise) with a wider per-stream range and a longer gap tail. This matches the greedy result of `docs/results/2026-11-05-batched-mtp.md` (N >= 2 does not pay on CUDA); `batched_mtp = 1` stays.

## Decision

**Default on.** Both gates pass: the seeded ids are identical with drafting on and off (CPU tests; 320 of 320 request runs on the 35B at B = 1, 2, 4), and the A/B shows +27 to +30 % decode for a lone sampled stream with no loss at 2 or 4 streams. `INTARSIA_BATCHED_MTP_SAMPLED=0` restores main's behaviour (sampled requests take plain steps). Nothing else changes default: `batched_mtp` stays 1, `INTARSIA_BATCHED_VERIFY` stays opt-in.

## Deviations and limits

- **Speed harness**: `localeval speed` is greedy only (no temperature option), so the A/B uses a small script on localeval's own streaming `Client` (same TTFT / decode / ITL definitions), as `scripts/ab35b/sampled.py` did. It keeps the chunk arrival times localeval measures to report the raw chunk gap. The runs are not logged to unanimis by localeval; the raw files are in `docs/results/2026-11-13-mtp-sampled/raw/`.
- **Scheduler trait changed** (`BatchModel::verify_step`, `SeqStep::Verify`, `BatchSeq::accept_verify`): needed so one ragged verify pass can serve greedy and sampled sequences. One implementation (`SlotModel`) and the test mock were adapted.
- **`engine.rs` touched** (the other lane's file): one field, its initializer, one `match` arm, one `mod` line. `model.rs`: `forward_verify_batch` split into a shared body with an optional logits download, and the verify logits download moved to the pooled buffers (also used by the single-sequence `INTARSIA_SPEC_SAMPLED` pass; values unchanged). Whichever of this lane and `mtp-store` lands second rebases.
- **Identity holds per request given its prefill**: a request's bits also depend on its prefill chunk schedule (busy chunk vs solo chunk, a reused prefix), which follows from timing and existed before this lane. The CUDA gate uses prompts of one chunk; batched MTP runs without slot prefix reuse. Long prompts that start prefill while others decode were not part of the ids gate.
- **Not measured**: HIP (the 27B on tatooine: out of the lane's fleet scope; the code path is backend-neutral and HIP has no batched MTP default), Metal, temperature 0.6 speed (acceptance only: 77.1 %), prompts beyond one chunk, k other than 2, the prefix store with MTP (lane `mtp-store`).
- **Acceptance differs from the single-sequence figure** quoted in the brief (57 to 64 %): here 69.6 to 73.7 % at temperature 1.0 with top_k 20 / top_p 0.95 on short chat and completion prompts; 66.9 % over the whole vocabulary.
- The 80 ms chunk-gap p99 of plain decode (about 12 ms median) is main's behaviour, present in arm A; this lane did not look into it.
- The advisor tool named in the user's global instructions is not available to this sub-agent; no advisor call was made.

## Fleet

| fleetq | What | Host | Load | Time |
|---|---|---|---|---|
| #9277 | build `mtp-sampled-b6ea108c4` (superseded, not measured) | spark1 | cpu, 34 GB | 5 min 49 s |
| #9280 | build `mtp-sampled-ad2606054` (the measured build) | spark1 | cpu, 34 GB | about 5.5 min |
| #9281 | seeded ids gate, 5 servers | spark1 | quiet gpu, 50 GB | 13 min |
| #9283 | paired speed A/B, 10 servers | spark1 | quiet gpu,cpu, 50 GB | 23 min |

spark1's vLLM (34 GB) stayed up throughout; lowest MemAvailable with this lane's server loaded 56.3 GB; earlyoom active; 0 watchdog kills. Remote scratch `~/intarsia-mtp-sampled` removed at the end (fleetq #9288, `--load none`; #9287 was the same command refused by its own process check, a `pgrep -f` that matched itself). No slot held.
