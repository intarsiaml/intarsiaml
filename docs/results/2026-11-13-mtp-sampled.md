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
