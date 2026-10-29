# MMTP: Metal MTP step without host round trips (35B, endor)

Lane `lane-mmtp/metal-gpu-accept`, off main `0d02d3f36`, merged with main `bbe668f1d` (#257, docs and scripts only: no Rust
change, so the measured binary is the merged code). Hardware: endor (M5 Max, 128 GB). Model: Qwen3.6-35B-A3B NVFP4
(hybrid: GatedDeltaNet + gated attention, MoE), `--spec mtp`, k = 2 (the default).

**Result.** Served MTP decode, median of 5 interleaved rounds x 3 reps (localeval): **+7.8 % at 1k, +5.4 % at 4k, +5.4 % at
16k**. Prefill unchanged (+0.1 % at 4k). Greedy ids are identical to `--spec off` and to main's MTP path on all 4 gate
prompts. GPU idle per verify pass (cb trace, 4k) goes from **1.12 ms to 0.31 ms**. The change sits behind
`INTARSIA_METAL_MTP_GPU_ACCEPT` (default on; `=0` is main's path).

## 1. What the 2.2 ms really was

MDEC (docs/results/2026-10-29-mdec-metal-decode.md, section 1) reported 2.2 ms of GPU idle per pass, 1.3 ms of it after
the verify pass's last buffer. Re-reading MDEC's own trace (`raw/w1/35-4k-mtpcb.jsonl.gz`) shows that 1.1 ms of it is one
gap: the 52 ms between the last decode buffer and the end of the run (output, shutdown), spread over 47 passes. Without
that last record the trace gives **1.12 ms idle per pass (6.7 % of 16.7 ms)**, with no single large gap. The idle is about
six gaps of 0.15 to 0.2 ms per pass, one per host round trip:

| gap (4k, MDEC trace without the last record) | per pass | idle per pass |
|---|---:|---:|
| head pass end -> its argmax read, chain copy, next id upload | 2 | 323 us |
| upload sync -> catch-up or verify embedding (t3) | 1.8 | 288 us |
| upload sync -> chain embedding (t1) | 1.1 | 179 us |
| verify end -> download, accept, rollback | 1 | 180 us |
| the rest | | ~150 us |

Each gap is the same pattern: about 70 to 90 us until the host opens the next encoder after the GPU finished (wake,
read), then about 60 us from commit to GPU start. On Metal `upload()` calls `synchronize()`, so every head pass's id
upload drains the queue too, not only the draft reads. The per-pass sequence on main was: verify read, upload of the
catch-up ids (drain), catch-up, argmax read (drain), upload of the chained id (drain), chained pass, argmax read (drain),
upload of the verify ids (drain), verify, read.

## 2. CUDA prior art

CUDA and HIP run draft + verify as one captured graph (`crates/intarsia-core/src/model/spec_graph.rs`): the catch-up and
chain read their ids from device tables, the verify reads the drafts where the head's argmax left them, and acceptance and
rollback stay on the host after the one download. Lane CMTP (#257) measured CUDA at 0.42 % idle per pass on that design.
Metal has no graph capture, but the same sequence of launches can be encoded eagerly: that is this lane.

## 3. The change

`crates/intarsia-core/src/model/spec_pipe.rs` (new), `crates/intarsia-core/src/engine.rs` (the `Seq::prepare` branch and
`Seq::next_draft_k`), `crates/intarsia-core/src/spec.rs` (`SpecGate::would_allow`), a `Model::set_spec_pipe` override field.
No backend or kernel change, no new op.

One step (`Model::spec_pipe_step`):

```text
  stage      ws.ids[0] = next, ws.ids[R ..] = catch-up ids       one upload, while the queue is empty
  catch-up   head over the n queued rows, ids from ws.ids[R ..]   argmax -> ws.ids[1]
  chain      k - 1 one-row head passes, each embedding ws.ids[i]  argmax -> ws.ids[1 + i]
  verify     main model over ws.ids[0 .. k + 1], row-exact kernels  -> verify_tok
  read       verify_tok and ws.ids (the drafts): the only wait
```

After the read the host accepts (`spec::accept`, unchanged) and rolls back (`SequenceState::spec_rollback`, unchanged: the
GDN rollback is the same GPU copy of the saved row, queued without a wait). Then, before the pass's tokens are emitted,
`Model::spec_pipe_predraft` encodes and commits the next pass's catch-up and chain (`Backend::fence`, no wait), so the GPU
drafts while the host emits tokens and encodes the next verify pass. The next pass's ids are staged right after the read,
while the queue is still empty, so that submission needs no drain either.

* **Not speculative.** The next draft is encoded after acceptance is known, so it is the draft the next step needs. It is
  only unused when the next step does not draft with the same `k` at the same length (a stop, the gate going plain, the
  end of the request); then it left nothing but head cache rows past `kv.len`, which the next head pass writes again (the
  host mirror of the head cache is updated only when a step takes the draft).
* **Exactness.** The same kernels on the same inputs in the same order as the eager step: the head passes as in
  `mtp_draft_rank`, the verify pass as in `forward_verify_inner` (`exact_rows`, `prepare_spec_rows`, `verify_head`, queue
  copy). Only the buffer the embeddings read their ids from and where command buffers end change. The drafts are bitwise
  the eager drafts (equal acceptance counts on every gate prompt, section 4).
* **Scope.** Single-sequence engine (`run_seq`) only: between two steps of a batched sequence other sequences use the
  workspace. Not under tensor parallelism, `--spec auto` / `dflash`, `INTARSIA_MTP_VERIFY_CHECK`, a plan recording or a
  profiled step. Batched MTP is not supported on Metal at all (`--spec mtp/auto with the batching engine (use --max-batch 1)`).
* **Not done:** encoding the next verify pass before acceptance resolves. Its position depends on the accepted count, which
  would need device-side positions and a split-KV grid sized for the maximum; the split count changes the reduction
  order, so it is a bitwise risk against `--spec off`. With 0.31 ms idle left (about 2 % of a pass) it is not worth it.

## 4. Gates

| gate | result |
|---|---|
| greedy ids, 256 tokens, 4 prompts (short 19 tok, 4k a 4096, 4k b 4171, 16k 16384), `generate` | `--spec off` == MTP `=0` == MTP `=1` on all 4 (md5 of the `ids:` lines: b4ca5abeaa, bda48660a1, 3c7cb036eb, 9687832874). Acceptance counts identical between `=0` and `=1` on every prompt (short 153/203, 4k a 155/198, 4k b 160/190, 16k 156/197) |
| main's MTP | main's own binary (`0d02d3f36`, `generate --spec mtp`) on the same 4 prompts: MAINMTP_RESULT (raw/diag3); its served cold and warm texts equal `=1`'s (raw/diag2). `=0` runs the eager branches of `Seq::prepare` as on main; the only change on that path is that `prepare` computes the plan first and clears the pipelined fields on non-pipelined branches |
| batched MTP == solo | not applicable: Metal refuses `--spec mtp` with `--max-batch` > 1 (raw/w3/srv-batch.err); the pipelined path is solo-only |
| prefix reuse == cold | **FAIL, on main as well, unchanged by this lane.** A served request resumed at 4095 of its 4105 tokens (the previous prompt + a suffix) gives different greedy text from the same request on a cold server, in every arm: `=1`, `=0`, main's binary and `--spec off` (raw/diag2). The warm text is identical across all four arms, so this lane does not change reuse; the cut itself is the issue (flagged as a separate task). Repeating the same prompt (warm A == cold A) is identical. REUSE_TESTS |
| `batch_metal_real` (35B, `INTARSIA_TEST_REAL_MODEL=1`) | 3 passed, 5 ignored |
| `batch_recurrent` | 14 passed |
| `kernels test --backend metal` (default and `--model` 35B) | exit 0 |
| `plan show --spec mtp` diff vs main | one line: the head's load time (0.4 s vs 0.3 s). No op or kernel change, as expected (no new op) |
| new tests (`crates/intarsia-models/tests/mtp.rs`, cpu) | `pipelined_mtp_step_keeps_output_and_drafts_of_the_eager_step` (k 1..3: output == MTP off, same passes / drafts / acceptances as the eager step), `pipelined_mtp_step_with_scripted_drafts_stays_identical` (all / some / no drafts accepted, max_tokens and stop ids inside accepted runs, follow-up requests reusing the prefix); 1533 of 1681 pipelined steps found a draft submitted ahead (scripted run, trace) |
| `cargo test --workspace --release` (Mac) | pass: 207 test binaries, 1107 passed, 0 failed, 28 ignored (#7216, commit `813c7f94a`; later commits change tests and docs only, `tests/mtp.rs` re-run after them: 25 passed, 0 failed, including the 4 pipelined tests) |

## 5. A/B (served, localeval)

`intarsia serve --backend metal --max-seq 20000 --spec mtp|off` (single sequence), `localeval speed --sizes
1024,4096,16384 --gen 256 --concurrency 1 --thinking off`, the arm order rotating by one per round. localeval puts a fresh
nonce first in every prompt, so the acceptance (and with it MTP tok/s) differs from call to call: endor's decode noise is
about 5 %, MTP's more. Medians are quoted.

**Final (w4, after the merge): 5 rounds x 3 reps, MTP `=1` vs `=0` (same binary).** Decode tok/s, median of each round's
3 reps, then the median of the rounds:

| context | `=0` (main's path) per round | median | `=1` per round | median | delta |
|---:|---|---:|---|---:|---:|
| 1k | 153.9 / 148.9 / 152.7 / 148.5 / 144.4 | 148.9 | 160.1 / 161.7 / 160.5 / 158.5 / 170.0 | 160.5 | **+7.8 %** |
| 4k | 144.7 / 142.9 / 145.0 / 139.0 / 148.5 | 144.7 | 156.2 / 152.9 / 152.5 / 149.7 / 150.9 | 152.5 | **+5.4 %** |
| 16k | 121.4 / 113.1 / 113.1 / 117.5 / 119.0 | 117.5 | 120.4 / 123.8 / 120.4 / 124.9 / 129.8 | 123.8 | **+5.4 %** |
| prefill 4k (tok/s) | 5827 / 5817 / 5827 / 5829 / 5837 | 5827 | 5839 / 5829 / 5838 / 5834 / 5832 | 5834 | +0.1 % |

At 1k and 4k every `=1` round is above every `=0` round.

**First pass (w2, before the merge, same code): 3 rounds x 1 rep, four arms.** MTP `=1` vs `=0`: 1k +3.0 %, 4k -0.8 %, 16k
+4.9 %, prefill 4k -1.7 %. With one rep per cell the nonce-driven acceptance spread dominates (4k: 144.2 / 145.9 / 144.1
vs 143.5 / 148.1 / 145.4); that is why w4 took 3 reps and 5 rounds. The host trace of one fixed 4k request (raw/diag1,
`INTARSIA_SPEC_TRACE=1`) gives the per-pass time directly: 14.63 ms per pipelined step vs 15.49 ms for the eager draft +
verify (-5.6 %), with 194 of 196 passes finding their draft submitted ahead.

**MTP off, no regression (w2, 3 rounds):** this branch vs main `0d02d3f36` binary: decode 1k -0.6 %, 4k +0.2 %, 16k -0.5 %,
prefill 4k +0.9 %; all inside noise (the off path runs no new code).

| arm (w2) | 1k decode | 4k decode | 16k decode | 4k prefill |
|---|---:|---:|---:|---:|
| off, main binary | 123.5 | 119.5 | 108.7 | 5262 |
| off, this branch | 122.8 | 119.7 | 108.2 | 5310 |

`generate` (one run each, 256 tokens, window 1): 4k a 165.6 -> 174.6 tok/s (+5.5 %), 4k b 171.8 -> 180.5 (+5.1 %), short
175.7 -> 185.9 (+5.8 %), 16k 133.9 -> 140.3 (+4.8 %); MTP off 120.6 / 120.8 / 126.6 / 109.1.

## 6. cb trace (4k a, 129 tokens, 52 passes, `INTARSIA_METAL_PROFILE=cb`)

| | period / pass | GPU busy / pass | idle / pass | command buffers / pass |
|---|---:|---:|---:|---:|
| `=0` | 15.68 ms | 14.56 ms | **1.12 ms** | 16.4 |
| `=1` | 16.19 ms | 15.87 ms | **0.31 ms** | 10.2 |

(Both without the run's last record, see section 1; `raw/scripts/passgaps.py`.) The remaining idle is one gap per pass:
verify end -> the predraft's first buffer (127 us on 0.7 of the passes, 72 us on the rest).

Under the cb profiler the pipelined run's verify buffers are about 10 % slower (1.6 vs 1.46 ms per 128-dispatch buffer, the
same 961 dispatches per pass), so its profiled period is longer. That is the profiler: the unprofiled runs are faster
(section 5), and the unprofiled per-pass host trace (14.63 vs 15.49 ms) matches the idle the cb trace removes (0.81 ms).
The cause of the profiler's slowdown was not found.

## 7. Runs and raw data

fleetq jobs: #7185 (build), #7186 (window 1: ids + cb), #7187 (main baseline build), #7197 (window 2: served A/B, 4 arms),
#7200 (test build), #7204 (served trace diagnostic), #7205 (build), #7206 (window 3: gates), #7212 (prefix reuse
diagnostic), #7215 (window 4: final A/B), #7216 (`cargo test --workspace --release`).

`raw/`: `w1` (generate logs with ids, cb traces), `w2` and `w4` (localeval outputs, server logs), `w3` (plan show, kernels
test, real-model tests, served gate texts), `diag1` (spec traces of one request per arm), `diag2` (prefix reuse texts per
arm), `scripts` (window scripts, trace analysers), `localeval-run-ids.txt` (result directories and unanimis records),
`ab-w2.md`, `ab-w4.md`.
