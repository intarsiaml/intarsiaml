# Structured output speed (lane so), 2026-10-22

Lane `so`, branch `lane-so/structured`, base `origin/main` `acba5987`. Measurement lane: the question was whether
constrained decoding (JSON schema through llguidance, J1) is slow per token, as finding O11 (1.7x llama.cpp) suggested.

## Headline

1. **There is no per-token structured-output gap to close.** On endor (Metal, Qwen3-8B BF16, exclusive slot) a
   constrained greedy step costs 27.5 ms against 26.9 ms for the unconstrained device-argmax step, +1.8 %
   on main (36.39 against 37.04 tok/s). The mask itself is off the critical path: the mask worker needs 28 us per step and the
   forward pass waits for it 0.5 us on average.
2. **O11's 1.7x is output length, not per-token cost.** The R9700 measurement behind O11
   (`docs/results/2026-10-22-fx6-standings.md` section 3) shows 68 completion tokens (median) for intarsia against 12
   for llama.cpp on the same `{"answer": "A".."D"}` schema, because that schema has no `additionalProperties: false`,
   so extra keys are allowed and the model writes a `reasoning` key. Closing that gap changes which tokens are
   allowed. It stays what O11 already says: an open decision for the user, option (c) first. Nothing here changes it.
3. **The four levers named in the brief are already in J1** (next-token mask computed on a helper thread while the
   forward pass runs; u32 bitset mask; logits copy only for constrained rows; llguidance's precomputed token trie).
   Two further micro-levers were tried (persistent mask worker instead of a thread per step; `fill` for all-zero mask
   words). Both are real but below measurement noise (section 3), so they are shipped as neutral, not as a speedup.
4. **Identity holds.** Allowed-token sets and greedy output are byte-identical to main (section 4).

## 1. Per-token breakdown (Metal, Qwen3-8B BF16, `--max-seq 4096`, fruits-40 schema, 177 tokens per request)

Slot: **exclusive** (fleetq #4197, endor, 10:36-10:48 EDT). One request each, streaming, greedy;
tok/s = (events - 1) / (last event - first event). Visit order: profile, base, head, head, base (interleaved A/B,
second round reversed). 2 reps per visit, so n = 4 per cell (fewer than the n = 5 asked for; the endor queue left a
12-minute window).

| arm | decode path | tok/s, median (min-max), n = 4 |
|---|---|---|
| unconstrained greedy, `ignore_eos`, same token count | device argmax, token feedback | base 37.04 (37.01-37.28), head 37.13 (36.91-37.30) |
| constrained greedy (`response_format` json_schema) | host logits + mask | base 36.39 (36.36-36.60), head 36.40 (36.38-36.65) |
| unconstrained sampled, temperature 0.7, `ignore_eos` | host logits, host sampler | base 26.25 (26.23-26.31), head 26.32 (26.29-26.35) |

Per-phase cost of one constrained step, from the opt-in timers (`INTARSIA_CONSTRAIN_PROFILE=1`, head binary, 300
constrained steps, same exclusive slot). Phases on the main thread are serial; the worker runs in parallel with the
forward pass:

| phase | us per step | where |
|---|---:|---|
| model step (derived: 1000 ms / 37.04 tok/s, device argmax) | 27 000 | GPU |
| mask compute (llguidance) | 27.6 | helper thread, overlapped |
| main thread waiting for the mask | 0.5 | serial |
| waking the worker | 1.4 | serial |
| commit (advance the grammar by the chosen token) | 5.4 | serial |
| mask apply (full-vocab `-inf` fill) | 39.7 | serial |
| sampler (greedy argmax over 151 936 floats) | 146.9 | serial |
| **serial CPU total** | **193.9** | 0.7 % of a step |
| host-logits path and no token feedback (derived, the rest) | about 290 | serial: 600 KB readback + GPU sync, and `Seq::feedback_ok` (T1) excludes constrained requests, so the greedy arm's on-device feedback is not available |

Constrained minus unconstrained is 27.48 - 27.00 = 0.48 ms per token (main), of which 0.19 ms is measured CPU work and about
0.29 ms is the host-logits path plus the missing token feedback (inferred by subtraction; not timed directly). Mask overlap holds: the wait of
0.5 us shows the worker is done long before the forward pass returns. The largest single serial item is the
argmax, 147 us; vectorising it would save at most 0.5 % of a step.

The "before" breakdown: main runs the same code as the head binary profiled above except that it spawns a thread per
step to compute the mask (about 17.5 us, CPU microbench below) where head wakes a persistent worker (1.4 us in situ).
Every other phase is the same code on both. No timed profile of the main binary itself exists (it has no timers).

CPU-only cross-check on the real Qwen3 vocabulary (`structured.rs::constrained_step_phases`, non-exclusive CPU slot,
so only the order of magnitude counts; random-logit walk, so its mask density differs from a real model's):
mask 2.3 us mean, commit 0.7 us, argmax 145 us, thread spawn + result channel per step 17.5 us against 4.3 us for
waking a persistent worker.

## 2. llama.cpp on the same schema and prompt

The harness is `docs/results/2026-10-22-so-structured-output/scripts/ssbench.py`, not localeval: `localeval speed`
never sends `response_format` (its `--structured` flag only shapes deepeval quality calls; see
`docs/results/2026-10-21-j1-structured.md`), and localeval forces a fixed output length, which a grammar that ends in
EOS cannot honour. O11's own metric (localeval MMLU with `--structured json_schema`, seconds per call) was not
re-run on Metal.

llama.cpp (BF16 GGUF, `-ngl 999 -fa on`, same exclusive slot #4197, 4 reps, `/v1/completions` with the same
`response_format`):

| arm | tokens | tok/s, median (min-max) |
|---|---:|---|
| `response_format` json_schema | **600 (cap)** | 35.06 (34.97-35.33) |
| greedy, `ignore_eos`, 600 tokens | 600 | 35.07 (34.71-35.23) |
| sampled, `ignore_eos`, 600 tokens | 600 | 35.12 (34.67-35.22) |

Read this with care. The constrained request ran to the 600-token cap where intarsia ends at 177 tokens (the grammar
closes after 40 fruits), so **llama.cpp's grammar was probably not applied on `/v1/completions`** and the 35.06 is
its unconstrained speed. What the table supports: intarsia constrained decode (36.4 tok/s) is not slower per token
than llama.cpp decoding without a grammar (35.1 tok/s), and intarsia's unconstrained greedy (37.1) is 5.7 % ahead of
llama.cpp, in line with the fx6 standings. A valid constrained llama.cpp number needs its native `/completion`
`json_schema` field; `scripts/llama.sh` does that (queued as fleetq #4208, see the status line at the end of this
doc).

## 3. Levers

| lever | state | effect |
|---|---|---|
| mask on a helper thread, overlapped with the forward pass | already in J1 (`ConstraintRun::prefetch`) | wait 0.5 us per step |
| u32 bitset mask, apply on host | already in J1 | apply 40 us per step |
| logits to host only for constrained rows | already in J1 (`HeadMode::LastLogits` per row) | the roughly 0.29 ms host-logits / no-feedback residual |
| llguidance token trie / mask caches | llguidance 1.9.1 internal | mask 28 us per step |
| persistent mask worker (this lane) | shipped, neutral | spawn 17.5 us to wake 4.3 us (CPU test), 0.05 % of a step |
| `fill` for all-zero mask words (this lane) | shipped, neutral | A/B shows no change; bit-identical to the old loop (asserted in the CPU test). The CPU test's old-versus-new timing is not like for like (its old loop omits the finite-logit tracking) and is not quoted |
| device-side mask (Metal kernel) | **not done** | would remove the readback and 0.19 ms of CPU work, at most 2 % of a step; needs a new kernel on all backends and the T1 token-feedback exclusion for constraints lifted, both outside this lane |

A/B base against head (levers + timers off): constrained 36.39 against 36.40 tok/s, unconstrained greedy 37.04 against
37.13 (+0.2 %, inside the 0.3 % visit spread), sampled 26.25 against 26.32. **No regression, no gain.** The two
micro-levers stay in only because they are tested and identical in output; they can be dropped without loss.

## 4. Correctness

- **Pinned mask digests (new test `structured::tests::mask_digests_are_stable`).** An FNV-1a digest over every mask
  word and every chosen token of 8 seeded walks, for the existing schema plus three added ones (nested object, enum,
  array of objects), each walk also JSON-validated against its schema. The digests were recorded on **main** (fleetq
  #4189, base tree) and asserted on head (#4202): `base=c2de089bb7602251`, `nested=c2f2146cc7161b78`,
  `enum=29c3e9c34bcf5e37`, `array=0489a17db0ea47d8`. Head passes: the allowed-token set at every step is identical to
  main's.
- **Greedy HTTP identity, Metal.** `scripts/ssbench.py identity` against a base and a head server in the same slot:
  34 cases (every `scripts/j1-structured/e2e.py` schema with both prompts via chat, plus 3 new schemas
  `ss_nested`, `ss_enum`, `ss_array` with 2 prompts each, `/v1/completions` for 5 schemas, 3 forced tool calls):
  **0 differ, 0 errors** (`raw/run/identity-base-r1.json` against `identity-head-r1.json`).
- **Tests.** `cargo test --release -p intarsia-server -p intarsia-core -- structured constrain`: 13 passed (server) and
  2 passed (core). `cargo test --release --workspace --no-fail-fast`: exit 0 (non-exclusive CPU slot #4202,
  `raw/test2/`), and again on the tree merged with `origin/main` `b85dd52f` (#4212, `raw/test3/`): 13 + 2 passed, workspace exit 0.
- Unconstrained requests take the same path as main: the timers sit inside the constrained branch of
  `Seq::token_from` and the `ConstraintRun` methods, which an unconstrained request never reaches.

## 5. Notes and caveats

- **Incidental finding, not fixed (out of scope):** unconstrained requests with `temperature > 0` decode at 26.3
  tok/s against 37.1 for greedy on this model (-29 %), 11 ms per token more than a constrained greedy step. A
  constrained request with temperature > 0 pays it too. The host sampler (softmax / top-p over 151 936 logits) is the
  likely cause; it was not profiled here.
- n = 4 per cell, two visits per engine: the visit spread is 0.1-0.7 % and the effects discussed are all smaller than
  or equal to that, except constrained against unconstrained (1.8 %) and sampled (-29 %).
- The per-sequence persistent worker was exercised under batching only by the CPU fixture
  `constrained_and_unconstrained_batch_together`; J1's concurrent constrained Metal HTTP run (`e2e.py --concurrency 4`)
  was not repeated on head. The coordinator's Metal check should include it.
- Single Qwen3-8B, single schema (fruits-40) for speed. Mask cost depends on the grammar state; free-string states
  allow about 143 000 tokens, closed enum/number states very few. The fruits schema spends most steps in free strings.
- The shipped engine change is opt-in diagnostics plus two neutral micro-levers in `crates/intarsia-core/src/constrain.rs`
  and `engine.rs` (shared files): the coordinator's Metal no-op check applies before merge.

## 6. Files, jobs, downtime

- Code: `crates/intarsia-core/src/constrain.rs` (timers `INTARSIA_CONSTRAIN_PROFILE`, persistent worker, `fill`),
  `crates/intarsia-core/src/engine.rs` (timers around apply / sample), `crates/intarsia-server/src/structured.rs`
  (tests only), `docs/tuning.md` (new env var row).
- Scripts: `docs/results/2026-10-22-so-structured-output/scripts/` (`ssbench.py`, `run.sh`, `build.sh`, `test.sh`,
  `llama.sh`, `aggregate.py`). Raw: `docs/results/2026-10-22-so-structured-output/raw/`.
- fleetq jobs (owner `claude-f31ccf91-so`): #4157 cancelled unrun; #4159 build base + head, non-exclusive, done;
  #4160 exclusive, released after 2 minutes (the script word-split `uv run --script` wrongly, no data); #4189 non-exclusive
  tests (main digests recorded; head digest test failed on its placeholder, microbench input bug); #4194 released
  unrun; **#4197 exclusive, the measurement above**; #4202 non-exclusive tests, passed; #4212 the same on the merged tree, passed; #4208 llama.cpp native check.
- Downtime: none. No role was taken down.
