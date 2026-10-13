# Prefix reuse: the Qwen3-8B chunk-16 divergence and the 35B `continue` failure (2026-10-13)

Head `7e65770` on `arch/plan`. Question: is the `intarsia accuracy --mode reuse` failure on Qwen3-8B (HIP, tatooine,
`--prefill-chunk 16`, short golden, `science` / `continue` diverging at token 9; log in
`qualify/run2-prefix-reuse-fail/`) an engine bug, or noise that the check is stricter than?

## Verdict

**Not an engine bug. A numerical tie, flipped by the last bit of the KV rows.** The engine's reuse bookkeeping (cached
length, truncation, positions, chunk boundaries, last-token row) is exact; the warm and cold runs differ only in
rounding. Rounding is unavoidable here (the decode kernels write the reply's KV rows, a cold run prefills them with the
GEMM / prefill-attention kernels), and a greedy stream flips whenever two candidates are closer than that rounding. The
check was wrong: "N identical leading tokens" fails a correct engine at a rate set by how often the model has near-ties.
The gate now judges a divergence on the logits.

No engine code needed a fix. What changed is the reuse check, plus tooling and tests that make the claim checkable
(section 5).

## 1. Evidence on HIP (Qwen3-8B, tatooine, fleetq #1617)

`intarsia accuracy --mode reuse --reuse-diag` records the logits row of every sampling step of the warm and the cold
run (`GenRequest::trace_logits`) and replays both through `Engine::forward_script` (explicit prefill chunks, one-token
decode steps, `truncate`), all teacher-forced on the cold stream so that the rows share one context. Full output:
`prefix-reuse/hip/diag-c16.txt`. Chunk 16, 32 rows (generated tokens) per comparison.

| check (32 rows each) | `science continue` (split 39, cached 54 of 95) | `story continue` (split 19, cached 34 of 55) |
|---|---|---|
| warm run vs the same warm run again | 32/32 rows bitwise | 32/32 bitwise |
| cold run vs the same cold run again | 32/32 bitwise | 32/32 bitwise |
| engine warm (generate) vs scripted replay of the same operations | 32/32 bitwise | 32/32 bitwise |
| engine cold vs scripted cold replay | 32/32 bitwise | 32/32 bitwise |
| cold at chunk 16 vs chunk 8 / 15 / 32 | 32/32 bitwise (all three) | 32/32 bitwise (all three) |
| **warm vs cold** | 0 bitwise, KL mean 1.4e-4 max 1.0e-3, max abs logit 0.25, 1 argmax flip | 0 bitwise, KL mean 2.9e-4 max 3.3e-3, max abs logit 0.24, 1 flip |
| cold vs warm whose reply was **prefilled** instead of decoded | **32/32 bitwise** | not bitwise (KL 2.9e-4), 1 flip |
| warm vs warm with prefilled reply (decode-written KV only) | 0 bitwise, KL 1.4e-4, 1 flip | 0 bitwise, KL 2.7e-4, 0 flips |

What this shows:

* The HIP backend is deterministic run to run and the engine does exactly what its documented operation sequence says
  (warm = replay, bitwise). A stale tail, a wrong `pos0`, a wrong chunk boundary or the wrong logits row would show up
  as a mismatch against the replay or as a large KL. There is none. (The same on the Mac cpu backend: every row of every
  comparison is bitwise identical, `prefix-reuse/cpu-before/diag-short-c16.txt`.)
* The cold prefill is bitwise independent of the chunk size in these cases (8 / 15 / 32 against 16; one exception, chunk 15 in
  `science restore`, KL 9e-5, no flip), so "warm chunks the prompt
  differently from cold" cannot be the cause on this backend. For `science` the whole warm-vs-cold difference is the
  15 reply rows that the one-token decode path wrote: prefill them instead and warm equals cold bit for bit. For
  `story` the unaligned start of the warm prefill (row 34 in tiles of 16) adds a second, equally small difference.
* The size of the difference is the size of the golden gate's decode-against-prefill noise (teacher-forced decode KL
  3.6e-4 against the fp32 reference in `run2-prefix-reuse-fail/`), far below anything a state bug produces.
* The divergent rows are ties. `science`, token 9, tokens 4354 / 576: the warm run preferred 4354 by 0.0125 nats, the
  cold run preferred 576 by **0.0003 nats**. `story`, token 28, tokens 9324 / 5710: margins 0.125 and 0.051 nats, with
  the shared rows moving by up to 0.14 nats of log-probability elsewhere in the same comparison.
* At chunk 32 the same prompt diverges at the same token 9 over the same two tokens with the direction reversed
  (warm 576, cold 4354; `hip/gate-c32.txt`): a coin flip decided by rounding, not a systematic error.

Why it looked like a bug: 2 of the 7 `continue` cases at chunk 16 (and 1 of 3 at chunk 32) hit such a tie within 32
tokens, i.e. roughly one near-tie per 100 generated tokens (2 in the ~200 tokens compared before a divergence ended the comparison). Prompts with flat next-token distributions (a story, a
science explanation) have many; the chunk-256 long golden has one reusable prompt, so it never saw one.

## 2. The 35B hybrid (`long_docs_4k continue`, ADR-0007 / phase1b 7.3)

Not re-run (spark1 was off limits for this task). It has the same signature: only `continue` failed (the case that
reuses decode-written rows, and on the hybrid the recurrent state advanced by the decode kernel), the failure vanished
in later runs on the same code, and the cause was never found. The mechanism above explains all three observations, but
that is an inference. The new gate would print the margin and KL of that divergence, which settles it.

## 3. Quantifying the "chunk-matched cold run" idea (task step 3)

Re-chunking alone does not move the rows on this backend (cold 16 against 8 / 15 / 32: bitwise; the Mac cpu backend:
bitwise too), so a chunk-matched cold run is the same run, and the check would not get less strict from it. What
differs between warm and cold is which kernel wrote which KV rows, and no cold run can reproduce decode-written rows.
Only a *restore* at a chunk-aligned split has a cold twin that prefilled the same chunks; that case is now an exact check
(section 4). For `continue` a logit-level criterion is the only honest one.

## 4. The new check (`crates/intarsia-cli/src/cmd/accuracy.rs`)

* Identical warm and cold streams pass, as before.
* New case `aligned`: `restore` with the split rounded down to a multiple of the prefill chunk (attention-only models).
  Warm and cold then run identical prefill chunks. On a backend whose cold run repeats bit for bit (checked from the
  recorded logits only when the streams differ) the streams must be identical (`--reuse-exact`, default on).
* A diverging pair is judged on the logits (`judge_streams`, `judge_divergence`): both runs are repeated with logits
  recorded; the divergence passes when every shared row up to and including the divergent one agrees within
  `--reuse-logit-tol` (default 0.25 nats of log-probability over the top-5 tokens, raised to twice what re-chunking a cold run
  moves them) and within `--reuse-kl-tol` (KL(cold || warm) <= 1e-2). The note prints the tokens, both margins, the
  row distances and the legacy verdict. An engine that cannot record logits falls back to `--reuse-min-match`.
* `--reuse-prompt a,b` limits the run; `--reuse-diag` prints the table of section 1.
* The tolerances are calibrated on three ties of one model: shared rows moving by 0.09-0.14 nats at most (0.025-0.10 at the
  divergent row) and KL up to 3.3e-3, against defaults of 0.25 and 1e-2 (the golden gate's own worst single-position KL against the fp32 reference is 1.3e-2). A real
  reuse bug moves rows by orders of magnitude more, but the margin is judged, not measured on a bug. A divergence that
  does not reproduce when logits are recorded is accepted with a note ("the run is not repeatable"); on a deterministic
  backend that case cannot arise (section 1).

## 5. Tests

Engine tests (`crates/intarsia-models/tests/engine.rs`, tiny bf16 model on the cpu backend, chunk 8):

* `prefix_reuse_restore_on_a_chunk_boundary_is_bit_identical_to_cold`: splits 16 / 24 / 40; `cached == split`, same
  tokens, logits rows bitwise equal to a cold run.
* `prefix_reuse_continue_and_unaligned_restore_match_cold`: the *continue* case (reply rows written one token at a
  time, `cached == split + reply - 1`) and an unaligned restore against a cold run (< 1e-3).
* `scripted_replay_reproduces_a_warm_generate_bit_for_bit`: `generate`'s warm run equals the same operations issued
  through `Engine::forward_script`.

No bug was found, so there is no "fails before the fix" test. Instead the tests were mutation-checked against the
bookkeeping: prefill restarting one token early (`let mut pos = p.saturating_sub(1)` in `generate_one`) fails all three new
tests (and two older ones); `SequenceState::truncate` not truncating `tokens` fails both reuse-vs-cold tests. Both
mutations reverted. The criterion has unit tests (`a_flipped_near_tie_is_rounding_but_a_moved_distribution_is_not`,
`row_nu_looks_at_the_top_tokens_only`) in `accuracy.rs`; dropping the pre-divergence row check fails the first.

## 6. Before / after

"Before" is the old criterion (`agree >= 16` of 32 leading tokens); the `agree` column of the new logs is what it
evaluated, and the notes print "legacy criterion: FAIL" where it disagrees. cpu: Qwen3-Reranker-0.6B (path in
`qualify/mac-cpu-smoke/state.json`), goldens `qwen3-reranker-0.6b[-long]-fp32`. hip: Qwen3-8B, goldens
`qwen3-8b[-long]-fp32`. Case counts are rows of the table (`continue` + `restore` + `aligned`, as many as the split allows).

| backend | golden | chunk | cases run | before (old criterion) | after (new gate) |
|---|---|---|---|---|---|
| hip | short | 16 | 20 | **FAIL**: `science continue` 9/32 (`story continue` 28/32 passed) | PASS: 2 rounding ties accepted (`science` @9: nu 0.05, KL 6.5e-4; `story` @28: nu 0.10, KL 3.3e-3); 18 identical |
| hip | short | 32 | 9 | **FAIL**: `science continue` 9/32 (the same token pair, reversed) | PASS: 1 tie accepted (nu 0.025, KL 6.7e-4); 8 identical |
| hip | long | 64 | 6 | PASS (all 32/32) | PASS (all 32/32) |
| hip | long | 256 | 3 | PASS (all 32/32; one reusable prompt) | PASS (all 32/32) |
| cpu | short | 16 | 20 | PASS (all 32/32) | PASS (all 32/32) |
| cpu | short | 32 | 9 | PASS | PASS |
| cpu | short | 64, 256 | 0 | SKIPPED, counts as FAIL (no split reaches a chunk) | not re-run, unchanged |
| cpu | long | 16 | 26 | PASS | PASS |
| cpu | long | 64 | 6 | PASS | PASS |
| cpu | long | 256 | 3 | PASS | PASS |

The hip "before" for chunk 32 was not run with the old binary: the old verdict is read from the `agree` column of the
new run, which is the same computation (the streams of the first, untraced runs). The chunk 16 failure was
reproduced with the new binary too (`science` `continue` 9/32, the same divergence index as `run2-prefix-reuse-fail`), so the
failure is deterministic on HIP, not a flake. The cpu "before" logs are in `prefix-reuse/cpu-before/`, after in
`cpu-after/`, hip in `hip/`.

The CPU cannot reproduce the failure: its matmul and attention are batch-invariant, decode and prefill kernels
produce the same bits, so warm equals cold bitwise at every chunk size (no tie can flip). It exercises the
bookkeeping, not the numerics; the numerics need the GPU kernels, hence the tatooine burst.

## 7. Limits and what is not done

* One model, one backend: the tolerances are calibrated on Qwen3-8B / HIP with two observed ties. CUDA and the hybrid
  are not measured (section 2).
* `forward_script` replays are written for attention-only models: the hybrid path adds a snapshot boundary at
  `plen - 1` that `generate` takes and the replay does not mirror, so `--reuse-diag` on a hybrid would show a
  non-bitwise engine-against-replay line by construction. The verdict logic does not use the replay.
* The `qualify` stage list is unchanged (no chunk-16 stress stage; `docs/adding-a-model.md` points here).
* tatooine burst: fleetq #1617, `--takes-down radiance`, ~8 minutes (14:10:52 to 14:19:02 local time); radiance stopped for the
  runs and started again, `/v1/models` answered before release. The binary was built twice inside the slot (the first
  tolerance of 0.05 nats was too tight and rejected the two ties; the second run used the final defaults). Nothing
  touched spark1 or spark2.
