# Metal next levers: speculative decoding and multi-request batching (2026-10-15)

Research and planning only. No code was changed and nothing was benchmarked for this document. Every number is one of:
**measured** (cited to a file in this repo), **sourced** (cited to a URL, with a confidence label), or **estimate** (derived here, with the
arithmetic shown). Where a number does not exist it says so.

Branch `metal-next-levers`, written against `spike/metal` at `3c92dda`.

## 0. Summary and recommendation

Metal batch-1 decode sits at 36.7 / 36.6 / 35.6 tok/s (128 / 1k / 4k context), 90.5 / 90.1 / 87.6 % of the 40.6 tok/s byte floor
(`docs/results/2026-10-15-metal-overnight-report.md`). Kernel tuning has stopped paying (D1 fusion, B5/B6 and the P4/P5 epilogues were all
at or under 0.7 %). The remaining lever is to read the 15.1 GB of weights once per step and get more than one token out of that read.
Two ways do that: several positions of one sequence (speculation) or one position of several sequences (batching). Both need the same
kernel capability, which today does not exist in a bandwidth-efficient and bit-exact form.

**Ranked recommendation**

1. **Shared prerequisite plus n-gram (prompt-lookup) speculation, greedy.** Build bit-exact multi-row kernels (rows = 2..16 activations per weight
   pass, per-row reduction order identical to the batch-1 GEMV, per-row split-KV attention). Put prompt-lookup drafting on top. No new weights, no
   second model, no regression when nothing matches (the step falls back to a plain M=1 decode). About 9-12 working days (shared kernels 6-8, engine loop and drafter 3-4). Expected 1.0x on novel
   prose, 1.3-2.2x on input-grounded work (summarise, quote, edit code, RAG). Details in section 4.
2. **Continuous batching for the serve path, decode-first.** Reuses the same kernels with rows from different sequences. Aggregate decode
   throughput is the largest available gain (upper bound about 7x at 8 concurrent streams and 1k context, realistic 5-6x; estimate), but only when requests actually overlap, and
   it is the largest build (about 15-25 days). Section 5.
3. **Draft-model speculation (Qwen3-0.6B, then 1.7B).** Reuses the verify path from item 1. Estimated 1.2-1.6x on any text, but acceptance of a stock
   0.6B/1.7B draft for the 8B target has not been measured by anyone I could find, the draft costs about 4 ms per proposed token on Metal (estimate),
   and the weights are not on this machine. Section 4.3.
4. **EAGLE-3 heads.** Last: no hidden-state hooks in the model, available heads were trained with thinking off, the published gains rely on tree
   verification that a contiguous single-offset KV cache does not support. Section 4.4.

**First milestone (measurable with localeval):** `intarsia serve --spec ngram` on Metal with the bit-exact verify path, gated in this order:
(G0) micro-benchmark go/no-go on the multi-row GEMV, (G1) byte-identical ids spec-on vs spec-off, (G2) localeval speed at 128/1k/4k with spec
off unchanged and spec on reported with acceptance, (G3) a realistic input-grounded dataset. Full criteria in section 7.

## 1. Baseline facts this plan rests on

| Fact | Value | Source |
|---|---|---|
| Decode (SAFE `a2102bf`) 128 / 1k / 4k | 36.7 / 36.6 / 35.6 tok/s = 27.2 / 27.3 / 28.1 ms per token | measured, overnight report |
| Byte floor | 40.6 tok/s at 614 GB/s, i.e. about 15.1 GB read per token (embedding table is not streamed) | measured/derived, overnight report |
| Per-token GPU time split (128 ctx, kernel-mode trace) | matmul_pair 13.47 ms (72 calls), matmul 10.50 ms (108), logits 2.09 ms (1), attention 0.76 ms (36), small ops 0.74 ms; sums to about 27.6 ms. 364 dispatches per step | measured, `2026-10-15-metal-decode-d1-fuse.md` section 1 |
| GEMV-only step | 26.07 ms, 580.8 GB/s (94.6 % of 614) | measured, `2026-10-14-metal-spike.md` |
| Prefill GEMMs | 58-65 TFLOPS (NAX), 4k prefill about 3500-4000 tok/s | measured, overnight report |
| KV per token | 36 layers x 2 x 8 kv heads x 128 x 2 B = 147,456 B (144 KiB); 4k context = 590 MB per stream per step | derived from `recipes/qwen3.toml` dims |
| Roofline ridge | 60 TFLOPS / 614 GB/s is about 100 FLOP/B. A bf16 weight is 2 B and costs 2 FLOP per row, so a weight pass stays bandwidth-bound up to roughly 100 rows. Compute is not the limit for k <= 16 | derived |

"k tokens cost about one weight pass" is the working hypothesis, not a result. Evidence against it exists and is quoted in section 4.6 (mlx-lm
EAGLE-3 prototype, llama.cpp MTP on M1 Max). The hypothesis only holds if the kernels reach GEMV-class bandwidth at M = 2..16, and in this
codebase today they do not (next section).

## 2. What exists in the codebase today

### 2.1 Serve and engine (backend independent, `crates/intarsia-core`)

| Piece | Where | State |
|---|---|---|
| Engine API | `intarsia-core/src/engine.rs`: trait `Engine::submit(GenRequest) -> Receiver<GenEvent>` | one request in, token events out. No notion of a step shared across requests |
| Scheduler | `engine.rs` `LocalEngine`: one worker thread, FIFO `mpsc` queue, `Inner { model, state }` with **one** `SequenceState` | "Phase 0 implementation: one worker thread, FIFO queue, a single sequence". `intarsia serve --max-queue` (default 64) is the wait line; the rest get 429 (`crates/intarsia-cli/src/cmd/serve.rs`) |
| Decode loop | `engine.rs` `generate_one`: prefill in `prefill_chunk` pieces, then `pick(model, st, &[tok], ...)` per token | greedy uses `HeadMode::LastArgmax` (on-device argmax, 4 bytes back); sampling uses `HeadMode::LastLogits` and the host sampler |
| Sampler | `intarsia-core/src/sampler.rs`: `Sampler::sample` draws **one** `rng.next_f32()` per sampled token, `sample_row(logits, params, u)`; `argmax` breaks ties to the lowest index | per request seeded RNG, deterministic |
| KV cache | `intarsia-core/src/state.rs`: `SequenceState` with `AttnKv { k, v: [n_kv_heads, cap, head_dim] }` contiguous per layer, `len`, `cap`, `tokens`, `truncate(len)` | `truncate` on an attention-only model is just `len = n` (free rollback). Recurrent (GDN) slots can only roll back to a snapshot (`RECURRENT_SNAPSHOTS = 4`). Prefix reuse is "longest common prefix with the one held sequence" |
| Model forward | `intarsia-core/src/model.rs`: `Model::forward(st, ids, HeadMode, row_cb)`; `HeadMode::{None, LastLogits, LastArgmax, AllLogits}`; `Workspace` sized by `prefill_chunk` rows (`ws.max_tokens`); head works on **one row**: `ws.logits` is `[1, vocab]`, `Op::Argmax` takes one row | `AllLogits` streams every row of a chunk to a callback through the f32 `Logits` op (tiled GEMM for M > 1, see 2.2). A multi-token forward with all-row output already exists for accuracy tooling |
| Graphs | `model/graph_step.rs`, `Backend::supports_graphs` | CUDA/HIP only. The Metal backend does not override `supports_graphs`, so Metal decode is encoded eagerly each token (364 dispatches, host encode time is part of the 27 ms) |
| Speculation / batching hooks | `grep -ri "specul\|draft\|ngram\|eagle"` over `crates/` finds nothing in code. `Defaults::max_batch` in `recipe.rs` (default 1, printed by `intarsia recipe`) is not read by the engine. `ARCHITECTURE.md` section 2/4 describes a continuous-batching scheduler and "Speculation: MTP, DFlash-style, n-gram" as plan only | no implementation |
| Telemetry | `telemetry.rs`: `step()` counts forward passes, `running/waiting`, watchdog fires on "no engine step while queued" | counts steps, not tokens: needs acceptance counters |
| Exactness vocabulary | `registry.rs`: `Exactness::{Fast, Deterministic, Exact}` with `Exact` documented as "Bit-identical to the serial reference order (lossless speculation claims)" | the hook for this work already exists; Metal's GEMV and dense-op kernels are mostly `Deterministic`, not `Exact`, and nothing yet asserts row-count invariance |
| Prior art on the same problem | `docs/decisions/0007-...md` section 3 and `docs/backends/hip.md`: HIP MoE is batch-variant above 16 tokens; "multi-row verify of 16 rows is not bit-equal to serial decode"; "the registry has no per-kernel batch-invariance flag (open)". `ARCHITECTURE.md` section 4: verify needs a q_len>1 split-KV kernel (R9700: 8930 -> 475 us at 18k) | open issue to close here |

### 2.2 Metal backend (`crates/intarsia-backend-metal`, shaders in `kernels/metal/`)

What a forward pass of T rows runs today (kernel table in `src/kernels.rs`):

| Op | T = 1 (decode) | T = 2..15 | T >= 16 |
|---|---|---|---|
| `Matmul` (q, o, down) | `matmul_bf16_gemv` (many GEMV variants, 16-byte loads, 94.6 % of peak) | `matmul_bf16` plain 32x32-tile `simdgroup_matrix` GEMM, tiles staged through f32 threadgroup memory (`gemm_bf16.metal`) | NAX GEMM (`gemm_nax.rs`, `NAX_MIN_M = 16`) |
| `MatmulPair` (k+v, gate+up) | `matmul_pair_bf16_gemv` (`when m == 1`) | falls back to two tiled GEMMs | `matmul_pair_bf16_nax` |
| `Logits` | f32-out GEMV (`gemv_f32out.metal`, `when m == 1`) | tiled f32-out GEMM | same tiled GEMM |
| `AttnDecode` | `attn_decode_bf16_splitkv` (`splitkv_ok` requires `m == 1`; split count is a function of the visible key count only) | not applicable | not applicable |
| `AttnPrefill` | flash (`flash_ok`, any m >= 1; tiles several query rows per threadgroup, the older `attn_prefill_bf16` uses 8) | flash | flash NAX (`nax_flash_ok`) |
| `QkNormRopeKv`, `KvAppend`, norms, `silu_glu` | row-independent kernels, element-to-thread assignment kept across forms | same | same |

Three findings follow, and they drive the plan:

1. **The M = 2..15 matmul path is untuned and unmeasured.** No file in `docs/results/` reports T(M)/T(1) for M below 16. The existing kernel there is
   the Phase-A tiled GEMM, written before the GEMV work. Whether it is 1.5x or 5x a GEMV pass at M = 4 is **unknown**. Do not plan on it.
2. **The measured M >= 16 path is not a free lunch either.** `docs/results/raw/2026-10-14-metal-prefill-p1/full/bench-pick.txt` (TFLOPS per shape):
   q/o (4096x4096) M=16: 4.80 TFLOPS, about 300 GB/s effective; M=32: 9.53 TFLOPS, M=64: 18.9, M=128: 37.9, i.e. the same time for 2x, 4x, 8x the rows (a latency floor, not a bandwidth limit).
   Effective bandwidth at M=16 (weight bytes / time; TFLOPS from `bench-pick.txt`): down (K=12288) about 306 GB/s, gate/up about 350 GB/s, k/v about 120 GB/s; the batch-1 GEMV moves weights at about 580 GB/s.
   So NAX at M = 16..128 costs a flat **1.65-1.9x** of the GEMV time for the big matmuls (about 4.8x for the small k/v pair), and is padded to 16 rows. Padding a
   k = 5 verify to M = 16 would make the matmul part of a step about 1.9x (bytes-weighted over a layer): roughly 24 ms -> 45 ms, and `Logits` at M > 1 takes the unmeasured tiled f32 GEMM, so a verify is **at least about 48 ms** (lower bound, estimate). That is a usable
   *stop-gap* for first measurements, not the end state.
3. **Rows currently take different kernels from decode, so outputs are not bit-identical.** A verify pass would use tiled GEMM or NAX instead of
   GEMV, flash prefill attention instead of split-KV decode attention, and the tiled f32 GEMM instead of the f32-out GEMV for logits. Near-ties can flip
   an argmax. The repo already measures how often rounding-level differences show up (long-prompt top-1 1430/1434, KL 3.7e-4 between kernel
   generations). With 150 draft/verify opportunities per 256 tokens a flip is expected eventually. Greedy identity therefore has to be built, not assumed.

## 3. Shared prerequisite: bit-exact multi-row kernels

Both features reduce to "run R rows through the model, where each row has its own position and (for batching) its own sequence". One kernel set serves both.

| ID | Kernel | What | Why this form | Effort |
|---|---|---|---|---|
| K1 | `gemv_bf16` with R rows (R = 2..16, template parameter), also `MatmulPair` and f32-out `Logits` forms | each lane loads a 16-byte weight chunk once and accumulates it against R activation rows. Each row keeps exactly the M = 1 variant's reduction order (same variant picked from `(k, n)`, never from R) | weight bytes unchanged, FMAs x R. At 580 GB/s a GEMV consumes about 290 G bf16 weights/s, i.e. 0.6 TFLOP/s of FMA per row: fine for R <= 4-8, but at R = 16 it is about 9 TFLOP/s of scalar fp32 FMA plus bf16 converts, which may be ALU-bound (ALU peak not measured here). G0 therefore tests R up to 8; R = 9..16 may need a `simdgroup_matrix` formulation (which then needs its own bit-exactness argument). **Row r is bit-identical to a batch-1 decode of that row**, so verify rows and batched rows equal serial decode by construction | 3-4 days (template + 3 launchers + `Exact` admission tests vs M = 1 + `gemv_b3`-style bench) |
| K2 | `attn_decode_bf16_splitkv` for q_len R | one launch, row r uses `visible = pos0 + r + 1` keys and the split count the M = 1 kernel would pick for that count; K/V of the new rows are already appended | per-row split choice is what keeps bits equal to decode (the R9700 lesson: do not fall through to prefill attention for a handful of query rows at 4k keys, where few threadgroups each read all keys; `splitkv_ok` requires `m == 1` today, so a verify pass lands on the flash prefill kernel) | reference = the serving decode step's exact launch configuration (see below). v0: R sequential M = 1 launches per layer, each with the same position source and simdgroup count the serving step uses (0.5 day, costs about R x 21 us x 36 = 0.8 ms per extra row at 128 ctx; per-row attention at 4k is larger); v1: fused multi-row kernel (2 days) |
| K3 | row-wise argmax over `[R, vocab]` and a `[R, vocab]` logits workspace | `Op::Argmax` currently takes one row and `ws.logits` is `[1, V]` | on-device greedy for all verify rows; 4R bytes back instead of R x 608 KB | 1 day |
| K4 | per-row positions | `Rope` / `KvAppend` / attention take `pos0 + row` for speculation, and a `(seq_slot, pos)` table for batching | design K1-K3 with a row descriptor now so batching later is only scheduling | included above |

**What "bit-identical to decode" means.** The reference is the serving decode step's exact launch configuration, not any M = 1 launch: the same GEMV variant chosen from `(k, n)` (including the 8-byte-aligned `a8` variants), the same position source and simdgroup count for attention (`attn.rs` documents that a host `pos0` launch and a device-side `OpCall::pos` launch may use different simdgroups per threadgroup and then agree only up to rounding order), the same row-norm form. K1/K2 pin to those choices; a faster variant at R = 8 that reorders accumulation does not qualify.

Governance: register K1-K3 as `Exactness::Exact` and extend the admission harness so each case compares R-row output to the serving decode step's launches
**bitwise**, not to the CPU reference with a tolerance. Add the per-kernel batch-invariance flag that decision 0007 lists as open. CPU reference
backend gets composition fallbacks (R x the M = 1 op), the same pattern D1 used, so CUDA/HIP keep building and the fallback is correct (slow).

Gate G0 (go/no-go for the kernel build; the criteria are in section 7): K1 at R = 4 and R = 8 stays within the T(R)/T(1) limits there on the Qwen3-8B shapes in the `gemv_b3` harness
(interleaved rounds, inside a fleetq slot), with the variant pinned to the M = 1 choice. If it does not, speculation on Metal is not worth building, because the "one weight pass" premise fails here.
The mlx-lm numbers in section 4.6 say that risk is real.

## 4. Speculative decoding

### 4.1 Engine changes common to every drafter

Entirely in `intarsia-core` (shared, no backend code beyond K1-K4):

* `engine.rs`: replace the inner `pick` loop of `generate_one` with `step(draft)`: feed `[last, d1..dk]`, get R = k+1 argmax tokens (K3), accept the longest
  prefix where `argmax[i] == d[i+1]`, emit argmax at the first mismatch (or the bonus token after a full match), then `st.truncate(len_before + accepted + 1)`.
  Rollback is free for attention-only models (`truncate` sets `len`); the rejected rows' K/V stay in the buffer and are overwritten by the next append.
* Per-emitted-token semantics must mirror `pick` and the loop that follows it exactly: `min_tokens` EOS/stop-id masking (the mask must apply to the row's
  position, i.e. `generated + i`), `stop_ids`, stop strings that end in the middle of an accepted run (stop at that token, discard the rest of the run and truncate
  KV to it), `max_tokens` and context-window clamping, `cancel` between steps, the `IncrementalDecoder`/`StopMatcher` fed in order.
* `model.rs`: new `HeadMode::VerifyArgmax` (or `forward_rows`) returning `Vec<u32>`; `Model::forward` already chunks by `ws.max_tokens`, k+1 <= 16 fits.
* Hybrid models (Qwen3.6 GDN, recurrent slots): exclude in v1 (`has_recurrent()` -> no speculation). Supporting them needs a recurrent snapshot per verify.
* Telemetry: `spec_proposed`, `spec_accepted`, `spec_steps` in `telemetry.rs`, exported on `/metrics`, and optionally on `GenEvent::Done` so localeval-side reports can
  carry acceptance. The watchdog counts steps, which keeps working, but tokens/step is no longer 1.
* CLI: `--spec {off,ngram,draft}`, `--spec-k`, `--spec-ngram-n`, default off. `ModelArgs` (`crates/intarsia-cli/src/common.rs`) and `intarsia generate` get it so the identity gate can use `generate --show-ids`.

**Greedy equivalence.** With K1-K3 the verify rows are bit-identical to what a batch-1 decode of the same prefix would compute, so spec-on and spec-off
produce identical ids by construction (not "within tolerance"). Without them, the only claim available is "lossless with respect to the verify path's numerics",
which is the claim vLLM and others make and which vLLM's own docs call a design goal rather than a guarantee
(<https://docs.vllm.ai/en/v0.8.5/features/spec_decode.html>, high confidence). The Thinking Machines write-up traces such differences to reduction order changing with
batch size and reports a 1.6-2.1x slowdown for a fully batch-invariant stack
(<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/>, high confidence); K1/K2 pay for invariance only on the decode-sized shapes, where weights
dominate, so the cost here is the R-fold FMA work, not a slower reduction.

**Sampling (temperature > 0).** Checked against `Sampler`: one `next_f32()` per sampled token, taken in token order. With a deterministic drafter and bit-identical rows one can
sample row by row with the request's RNG, compare to the draft, and stop at the first mismatch: the RNG draws consumed equal the number of emitted tokens, so a seeded
sampled run produces the same ids with speculation on or off. This is exact-match acceptance (what mlx-lm does), not Leviathan-style rejection sampling; it gives the target
distribution exactly and is lossy only in speed (low acceptance at high temperature). Needs `HeadMode::VerifyLogits` (R x 608 KB readback) rather than device argmax. Do it
after the greedy path.

### 4.2 Option A: n-gram / prompt-lookup drafting

* Drafter: host-side, per request, over `prompt + generated` ids. Find the most recent earlier occurrence of the last n tokens (n = 3 typical, falling back to 2), propose the next
  k = 4-8 tokens after it. Microseconds on CPU; no memory; no model; `TextCodec` untouched. New file `intarsia-core/src/spec.rs` (trait `Drafter { fn propose(&mut self, ctx: &[u32], k: usize) -> Vec<u32> }`).
* No match -> the step is a plain M = 1 decode (graph/GEMV path unchanged), so the worst case is zero overhead beyond the lookup. Matches cost one R-row step.
* Changes: section 4.1 + `spec.rs` + K1-K4. Shared vs backend: only K1-K3 are Metal-specific; CUDA/HIP take the composition fallback until someone ports.
* Effort: 1 day for the drafter on top of the 4.1 engine work (2 days) and the shared kernels (6-8 days, section 3). Total 9-11 days. A functional prototype using existing paths (`AllLogits`, tiled GEMM, no
  exactness) is 2-3 days and runs **in parallel with G0**: G0 decides whether the kernel build happens, the prototype measures match rate p, mean accepted m and the token-flip rate on real text (it is not a speed result, the tiled GEMM is slow).
* Expected gain (estimate). Per step, with probability p a match exists and k = 4 drafts are verified, yielding 1 + m tokens on average; otherwise a plain step.
  Speedup = (p(1+m) + 1 - p) / (p(1 + 4 delta) + 1 - p), where delta is the cost of an extra verify row relative to a step (K1/K2 target 0.04, pessimistic 0.10):

  | match rate p | m accepted given match | delta 0.04 | delta 0.10 |
  |---|---|---|---|
  | 0.05 (novel prose) | 2.0 | 1.09x | 1.08x |
  | 0.2 | 2.0 / 3.5 | 1.36x / 1.65x | 1.30x / 1.57x |
  | 0.4 (RAG, summary of given text) | 2.0 / 3.5 | 1.69x / 2.26x | 1.55x / 2.07x |
  | 0.7 (code edit, quoting) | 2.0 / 3.5 | 2.16x / 3.10x | 1.87x / 2.70x |

  p and m are **assumptions**, not measurements. External anchor: the prompt-lookup author reports about 2.4x on CNN/DM summarisation and on HAGRID QA (Mistral-7B, A100, greedy) and a much smaller gain
  on multi-turn chat (<https://github.com/apoorvumang/prompt-lookup-decoding>, medium confidence: author benchmarks, no code-edit data, sampling untested).
* Risks: acceptance collapses on thinking-mode reasoning text and on novel generation (expected ~1.0x); the localeval speed prompt ("Note N: the X Y will Z the ..." filler) is highly repetitive
  and will flatter n-gram, so it cannot be the only workload (section 7). Chat-template boilerplate matches help every workload slightly.
* Comparison: llama.cpp has it (section 4.6); mlx-lm 0.31.3 / 0.32.0 does not (grep of the installed `generate.py` finds no lookup or n-gram code).

### 4.3 Option B: small draft model (Qwen3-0.6B / 1.7B)

* Tokenizer: both share Qwen3's vocabulary (`vocab_size` 151936 in all three config.json files; 0.6B/1.7B tie embeddings, 8B does not;
  <https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json>, <https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json>, high confidence). Whether the padded rows beyond the real tokenizer entries
  behave identically was not verified.
* Weights are **not on this machine** (`~/models` has `Qwen3-8B` only). 0.6B is about 1.2 GB bf16, 1.7B about 3.4 GB. A download needs the user's go-ahead.
* What must change beyond 4.1: load a second `Model` with its own backend instance, workspace and `SequenceState` (`LoadOptions`, `memplan.rs`: the KV budget is computed from measured free memory,
  so two models need a split budget, `--mem-budget`); keep the draft KV in step with the target (draft prefill of the whole prompt counts toward TTFT; roll back with `truncate`; after a full
  accept the draft must also consume the last proposed token, as mlx-lm does); a `Drafter` that runs k greedy draft steps. Draft steps use the existing M = 1 GEMV path, so the draft needs **no** new kernels.
  Draft token feeding currently round-trips through the host (argmax readback per token); chaining argmax -> embedding on device would save the sync, a later optimisation.
* Cost of a draft token on Metal (estimate, not measured): 0.6B has 28 layers, about 0.88 GB of non-embedding weights plus the 0.31 GB tied head = 1.2 GB, 2.0 ms at 580 GB/s. Small matrices
  (1024-wide) reach less than the 8B's 94.6 % efficiency, and Metal here has no graph capture, so assume 60 % efficiency plus about 0.7 ms of small ops and sync: **about 4 ms**. 1.7B: 28 layers,
  2.8 GB + 0.62 GB head = 3.4 GB, **about 7.5 ms**. The 8B step is 27.3 ms.
* Expected gain (estimate): tokens/s = 1000 * E / (gamma * c_draft + T_verify), E = (1 - alpha^(gamma+1)) / (1 - alpha), T_verify = 27.3 * (1 + delta * gamma) ms, baseline 36.6 tok/s.

  | draft | gamma | alpha 0.5 | 0.6 | 0.7 | (delta = 0.04) |
  |---|---|---|---|---|---|
  | 0.6B (4 ms) | 2 | 46.7 (1.27x) | 52.3 (1.43x) | 58.4 (1.60x) | |
  | 0.6B | 3 | 44.0 (1.20x) | 51.1 (1.40x) | 59.5 (1.62x) | |
  | 0.6B | 4 | 40.6 (1.11x) | 48.4 (1.32x) | 58.2 (1.59x) | |
  | 1.7B (7.5 ms), alpha 0.6 / 0.7 / 0.8 | 3 | 41.0 (1.12x) | 47.7 (1.30x) | 55.6 (1.52x) | |

  With delta = 0.10 the 0.6B gamma = 3 row becomes 39.5 / 45.8 / 53.3 tok/s (1.08x / 1.25x / 1.46x). With the NAX stop-gap verify (at least 48 ms, so these are upper bounds) 0.6B gamma = 3 is at most 36.3 / 42.2 tok/s at alpha 0.6 / 0.7 (0.99x / 1.15x): the stop-gap is for first measurements only.
  A draft costs about 15 % of a step per proposed token (0.6B), so alpha near 0.5 or below is roughly break-even and only larger alpha pays.
* Acceptance anchor: vLLM PR #24322 measured Qwen3-1.7B drafting for Qwen3-32B on MT-Bench: acceptance length 2.79 at k = 3 and 3.12 at k = 4 (temperature 0), batch-1 TPOT 46.1 -> 22.6 ms, **on CUDA** (<https://github.com/vllm-project/vllm/pull/24322>,
  medium confidence). 2.79 at k = 3 corresponds to alpha of about 0.76 (solving E; derived here). For the 8B target with a 0.6B draft I found **no measurement**; the alpha range above is a guess.
* Effort: 4.1 (2 days) + K1-K4 (5-6) + second-model plumbing, draft KV sync and TTFT accounting (4-5 days) = about 12-14 days, i.e. 4-5 days on top of Option A.
* Risks: unmeasured alpha; draft cost on Metal may be nearer 5-6 ms; thinking-mode text (Qwen3 emits long `<think>` sections) lowers alpha; memory and `--mem-budget` splits; a second `Model`'s encode time is
  host-side serial work added to every step.

### 4.4 Option C: EAGLE-style heads

* Weights exist: `RedHatAI/Qwen3-8B-speculator.eagle3` (Apache-2.0, 2025-07-27, about 1B parameters per the card, **trained with thinking disabled** on ShareGPT/UltraChat; acceptance length 2.39 / 2.69 on HumanEval at k = 3 / 7, 2.48 / 2.81
  on GSM8K, 2.13 / 2.30 on CNN/DM, on one A100 under vLLM 0.11.0) (<https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3>, high confidence on the card; size in MB not shown). `Tengyunw/qwen3_8b_eagle3` (MIT) exists too, with vendor-reported
  throughput on H200/RTX 5090 and no acceptance or thinking-mode information (<https://huggingface.co/Tengyunw/qwen3_8b_eagle3>, low confidence). No Qwen3-8B MTP or Medusa head was found (Qwen3-8B has no native MTP; absence of evidence, not verified against the model card).
* What must change: hidden-state taps at three intermediate layers of the target (no hook exists in `Model::run_chunk`), a new head graph (fused feature projection + one decoder layer + draft lm_head) loaded as its own
  small model, per-step draft state fed with target features, a recipe/loader for the speculator format, and tree verification for the published gains. Verification of a tree needs a tree-masked attention, which the
  contiguous single-offset KV state does not support; mlx-lm hit exactly this limit
  (<https://github.com/ml-explore/mlx-lm/discussions/890>, **low confidence**: LLM-written post, unverified).
* Evidence on Apple Silicon is thin and weak: the same discussion reports 1.05x for Llama-3.1-8B 4-bit on an M3 Ultra with 34 % acceptance and about 0.5-1 ms per extra verified token (low confidence).
* Effort: 15-20 days including a chain-only (no tree) version. Expected: below Option B for chain verification, because the acceptance lengths above (2.1-2.8) are at tree depth and the head still costs a weight pass of about
  1 GB per proposed token. **Ranked last.** Revisit only after a thinking-mode-trained head exists or tree attention is built for batching anyway.

### 4.5 Why bit-exactness is the central requirement, not a footnote

Without K1/K2, a verify pass runs tiled-GEMM / NAX / flash-prefill kernels for rows that a plain decode would run through GEMV / split-KV. The two disagree at rounding level (this repo's own long-prompt top-1 agreement is 1430/1434). In a
256-token generation with about 100 verify steps a flip of one near-tie token is plausible, and after a flip the sequences diverge for the rest of the generation. The same kernels are what batching needs: if batched rows are not bit-equal to
M = 1, a request's output depends on which other requests happen to be running, which breaks reproducibility, `qualify`'s identity gates and prefix-cache equivalence. Building K1/K2 once makes speculation and batching both "free" in the
accuracy sense and gives the repo its first `Exact`-class Metal kernels.

### 4.6 What mlx-lm and llama.cpp offer on Apple Silicon

| | mlx-lm | llama.cpp |
|---|---|---|
| Draft model | Yes. `--draft-model`, `--num-draft-tokens` (CLI default 3). Draft and target caches both trimmed on rejection. Acceptance rule: draft token equals the target's own sampled token (exact-match, chain only, no tree). Read locally in installed 0.31.3 `mlx_lm/generate.py` (speculative loop, `_rewind_cache`) | Yes. Local build b11429 has `--spec-draft-model`, `--spec-draft-n-max/-n-min`, `--spec-draft-p-min`, `--spec-draft-device/-ngl` (read from the binary's strings; current docs: <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>, high confidence) |
| n-gram / lookup | Not in 0.31.3 or 0.32.0 (no "lookup"/"ngram" in `generate.py`; checked locally) | Yes: `--spec-type` includes `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` (defaults n = 12, m = 48 for simple/map-k), plus `--lookup-cache-static/-dynamic` in b11429 |
| EAGLE-3 / MTP | Prototype discussion only (#890) | `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark` types in current docs (not found in b11429's strings, only the ngram implementations were) |
| Metal evidence | EAGLE-3 prototype 1.05x (M3 Ultra, Llama-3.1-8B 4-bit, low confidence) | MTP on M1 Max, Qwen3.5-9B Q4_K_M, temperature 0: baseline 25.3 tok/s; n-max 0: 22.4; n-max 2: 21.9 (76 % accept); n-max 6: 19.3 (44 % accept), i.e. slower than baseline everywhere, root cause not established (<https://github.com/ggml-org/llama.cpp/issues/23752>, high confidence on the numbers, closed "not a bug"). Few-row Metal matmul kernels aimed at verification are described in secondary blog posts only (low confidence) |
| Batching in the server | `mlx_lm/server.py` `BatchGenerator` (`--decode-concurrency`, `--prompt-concurrency`). Local code: `is_batchable = draft_model is None and all(c has merge)`, so **a loaded draft model turns batching off** | `llama-server --parallel N` / `--cont-batching` present in b11429. I found no published Metal aggregate-throughput numbers (`llama-batched-bench`); third-party mlx numbers (vllm-mlx: Qwen3-0.6B-8bit 328 -> 1112 tok/s at 5 requests, M4 Max, self-reported, <https://vllm-mlx.is-a.dev/guides/continuous-batching/>) are low confidence |
| Greedy identity | Not claimed; acceptance by equality with the verify pass's own token | Not claimed |

What this says for us: the two reference engines both ship draft-model speculation; mlx-lm cannot combine it with batching, which supports ordering them as separate features. Neither publishes a clean Apple-Silicon speedup for a
dense bf16 8B target, and the one Apple-GPU datapoint with real numbers (llama.cpp MTP, quantised, M1 Max) is negative. Our own bf16 advantage (a 27 ms step, where a fixed per-row cost is a smaller fraction than in a 4-bit step of about 9 ms,
mlx-lm's 8.8 ms per token above) is an **inference**, not a measurement. Both engines use the plain verify pass, so neither demonstrates the bit-exact variant proposed here.

## 5. Multi-request (continuous) batching in the serve path

### 5.1 Today

One worker, one `SequenceState`, FIFO. A second client waits for the first to finish its whole generation (`max_queue` 64, then 429). Prefix reuse works because there is exactly one held sequence to compare against. `localeval speed --concurrency 2,4` already measures
`aggregate_decode_tok_s` per N parallel streams (`/opt/code/deepeval/src/localeval/speed.py`), so today's baseline is measurable: expect aggregate about equal to one stream (about 36 tok/s) and per-stream rates of 1/N plus queueing in TTFT.

### 5.2 What must change

| Area | Change | Where | Shared / backend |
|---|---|---|---|
| Scheduler | per-step loop over active sequences: pick up to B decode rows plus at most one prefill chunk, run one forward, route tokens back to each request's `Sender<GenEvent>`; admission by free KV; cancellation per request | `engine.rs` replaces `LocalEngine` worker body; telemetry `running/waiting` becomes real | shared |
| Sequence state | N `SequenceState`s (one per slot) instead of one; prefix reuse becomes "pick the slot with the longest common prefix, else the LRU free slot" | `engine.rs` `Inner`, `state.rs` | shared |
| KV layout | v1: one arena buffer per layer `[slots, nkv, cap, d]` so the batched attention kernel gets `(arena, slot)` per row; memory is `slots x cap x 144 KiB` (4096-token slots: 590 MB each, 8 slots = 4.7 GB). v2: paged blocks + block table (needed only for long, uneven contexts) | `state.rs` `SlotSpec::AttnKv`, `memplan.rs` (`max_batch` finally read) | shared + K2 kernel |
| Forward API | `Model::forward_batch(rows: &[Row { slot, pos, token }], head)`; `Workspace` already holds `[max_tokens, hidden]` rows so B decode rows fit | `model.rs` | shared |
| Row-wise ops | Rope / KvAppend take per-row `(slot, pos)`; embedding gather per row; K1 multi-row matmuls take B rows (B up to 16 with K1; above 16 the NAX GEMM is the measured path, 1.65-1.9x of GEMV time up to M = 128 for the big matmuls, section 2.2) | K1-K4 | Metal; core composition fallback for CUDA/HIP |
| Attention | one launch for all B rows, each row bound to its slot's K/V and visible length (K2 with a `(slot, pos)` table) | `attention_splitkv.metal`, `attn.rs` | Metal |
| Head | `[B, vocab]` logits workspace, row-wise argmax (K3); sampling rows on the host from the same readback | `model.rs` | Metal + shared |
| Chunked prefill interleaved with decode | a 4k prefill takes about 1.2 s (measured 4k at about 3500 tok/s) and stalls every decode stream if not interleaved; with `prefill_chunk` 256 rows, one chunk is about 70 ms. Mixed steps (decode rows + one chunk) are the standard fix; the flash NAX prefill attention already runs the chunk, K2 runs the decode rows, in the same step | `engine.rs` scheduler | shared |
| Server | `--max-queue`, 429, SSE streaming, health/watchdog unchanged in shape; `/metrics` gains running/waiting/batch-size histograms | `intarsia-server` | shared |

### 5.3 Expected gain (estimate)

Step time model: weights 15.1 GB read once (27.3 ms measured) plus B x ctx x 144 KiB of KV read (bandwidth-scaled from the measured step), assuming K1/K2 reach batch-1 efficiency and ignoring per-row dispatch cost:

| context | B = 1 | B = 4 | B = 8 | B = 16 |
|---|---|---|---|---|
| 128 | 27.3 ms, 37 tok/s | 27.4 ms, 146 | 27.6 ms, 290 | 27.8 ms, 575 |
| 1k | 27.6 ms, 36 | 28.4 ms, 141 | 29.5 ms, 271 | 31.7 ms, 505 |
| 4k | 28.4 ms, 35 | 31.7 ms, 126 | 36.0 ms, 222 | 44.8 ms, 357 |

(aggregate tok/s). This is an upper bound: it assumes bandwidth-class kernels at every R, one attention launch per layer for all rows, and no scheduling gaps. Realistic target: 70-85 % of it. The 4k column is KV-bound (B = 8 reads 4.7 GB of KV per
step, +29 % bytes over weights alone), which is the case for paged/fp8 KV later. If K1 stops at R = 16 and larger batches go through NAX, the matmul part rises to about 1.9x (24 ms -> 45 ms, step at least 48 ms) and aggregate throughput still keeps growing: B = 32 at 128 context is
about 32 / 0.048 s = 650 tok/s (estimate, upper bound).

* Effort: scheduler and multi-state engine (5-6 days), `forward_batch` and row-wise ops (3-4), batched attention and arena KV (3-4, partly in K2), chunked-prefill mixing (3-4), server and telemetry (2), exactness and concurrency tests (3). **About 15-25 days** on top of K1-K4.
* Risks: only helps when requests overlap (single-user chat sees nothing); fairness and TTFT under load; memory = slots x cap; prefix-cache semantics change; scheduler bugs are harder to reproduce than kernel bugs; the Metal command-buffer model has one stream (`synchronize` takes the stream lock), so concurrent encode and GPU work needs a deliberate design;
  per-step host encode (364 dispatches, eager) is now amortised over B tokens, which is a gain, but a batch change must not recreate a per-row encode cost.
* Interaction with speculation: both consume R rows per step. At B above about 4 the weight pass is already shared, so speculation adds little aggregate throughput and should switch off (rows are better spent on other sequences). Speculation is a low-load latency feature, batching is a load feature.

## 6. Ranking and effort

| Rank | Item | Days (estimate) | Gain (estimate) | Needs | Main risk |
|---|---|---|---|---|---|
| 0 | K1-K4 shared exact multi-row kernels | 6-8 | enabling; also removes the dispatch/encode cost per token for rows 2..R | none | G0 fails (T(R)/T(1) > 1.15) |
| 1 | n-gram speculation | 3-4 on top of K1-K4 (K2 v1 included in row 0); 2-3 more for the parallel prototype | 1.0x prose, 1.3-2.2x input-grounded | K1-K4 | real acceptance unknown |
| 2 | Continuous batching, decode-first | 15-25 on top of K1-K4 | 4-8x aggregate at 4-8 streams, 1k ctx; 0x for a single user | K1-K4 | needs concurrent clients; scope |
| 3 | Draft model 0.6B (then 1.7B) | 4-5 on top of 1 | 1.2-1.6x on all text if alpha >= 0.6 | download weights (user approval); 4.1 | alpha unmeasured, draft cost on Metal |
| 4 | EAGLE-3 chain | 15-20 | below 3 | hidden taps, head loader, probably tree attention | thinking-off training, no tree |

Why batching is above the draft model despite the larger build: its gain scales with a known quantity (bytes and measured kernels), it applies to every request that overlaps, and K1/K2 plus the arena KV are reusable by everything else. The draft model's gain depends on an acceptance number nobody has measured for this pair and
caps near 1.6x. If the product is strictly single-user, swap ranks 2 and 3.

## 7. First milestone: n-gram speculation, greedy, bit-exact, measured with localeval

Scope: `intarsia serve|generate --spec ngram [--spec-k 4]` on Metal, attention-only models (Qwen3-8B BF16), default off. Four gates, each a separate commit so a failure stops cleanly. The day counts on the gates are **verification and measurement time only**; the build (K1-K4 6-8 days, engine loop and drafter 3-4 days) comes on top, for the 9-12 days total in section 0. Gate order: G0 runs first and in parallel with the functional prototype (section 4.2); G1-G3 follow the kernel build. Run benchmarking inside a fleetq slot; localeval queues itself.

* **G0 (go/no-go, 3 days): multi-row GEMV micro-benchmark.** K1 (Matmul, MatmulPair, f32-out Logits) at R = 2, 4, 8 against the M = 1 variants on the Qwen3-8B shapes (q/o 4096x4096, k/v 1024 pair, gate/up 12288 pair, down K = 12288, lm_head 151936x4096) in the `gemv_b3` harness style: interleaved rounds, rotating 3 GiB weight pool, GPU-idle check.
  Pass: T(4)/T(1) <= 1.10 and T(8)/T(1) <= 1.20 summed over a layer's shapes, measured with the variant pinned to the M = 1 choice (a faster variant that reorders accumulation does not count), and R-row output **bitwise equal** to the serving decode step's launches (`Exactness::Exact` admission case). Also record, once, the unmeasured status quo: tiled GEMM at M = 2..15 and NAX at M = 16 for the same shapes, so section 2.2's gap becomes a number.
  Fail: stop; speculation on Metal is not justified at this precision, write the T(R) curve into this document.
* **G1 (identity, 2 days).** `intarsia generate --show-ids` with `--spec off` and `--spec ngram`, temperature 0, short prompt and `tools/prompts/golden_4k.jsonl` prompts, 48 and 256 new tokens (the D1 identity-gate procedure, `docs/results/raw/2026-10-15-metal-decode-d1/gates/identity/`), stdout byte-identical, plus chat-template prompts that
  hit `min_tokens`, `stop`, a stop string landing mid-run, and `max_tokens` landing mid-run. Spec off must be byte-identical to the current `spike/metal` binary (M = 1 bits unchanged). `intarsia accuracy` decode KL unchanged (3.5e-4 class), `cargo test --workspace`, `kernels test --backend metal` all admitted.
* **G2 (speed, 1 day).** `localeval speed --url ... --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off`, same slot, A/B spec off vs on, 3 rounds, GPU clock logging as in the overnight runs. Criteria: spec off within noise of 36.7 / 36.6 / 35.6; spec on reported with proposed/accepted counts from `/metrics`; **no row below spec off**.
  Expect the harness's decode-ceiling warning (`decode_ceiling`) to fire when spec on beats the byte floor; that is the intended signal, not a measurement bug. Caveat: the harness prompt is synthetic "Note N: ..." filler, so n-gram acceptance here is an upper-bound curiosity.
* **G3 (realistic workload, 1 day).** Build a small JSONL for `localeval quality --dataset` (rows need `input`, optional `messages`/`expected_output`; format in `quality.py` `load_dataset`) from `tools/prompts/corpus_docs.txt` and `corpus_code.txt`: summarise-this-document, answer-from-this-passage, rename-a-function-in-this-file, plus 20 open-ended prompts with no grounding.
  Compare wall time and decode tok/s spec off vs on, thinking off, and report acceptance per category. Success: at least 1.3x on the grounded rows, 0.97x or better on the open-ended rows, ids identical for every row.
* **Cheap parallel probe (0.5 day, optional).** Before G0, time the 8B in mlx-lm for 1 vs 4 vs 8 verified tokens (a forward over `[1,T]`, bf16) on this machine, and run `mlx_lm.generate --draft-model Qwen3-0.6B` once, to get an independent T(R)/T(1) and a first alpha for the 8B/0.6B pair (needs the 0.6B download, user approval). mlx's own kernels are not ours, but the ratio tells whether the premise holds on M5 Max at all.

Stop conditions follow the repo's rule from the overnight work: two consecutive attempts that gain under 0.2 ms/token of verify cost stop the K1 tuning and the result is documented.

## 8. Sources and confidence

* Repo (measured): `docs/results/2026-10-15-metal-overnight-report.md`, `2026-10-15-metal-decode-d1-fuse.md`, `2026-10-14-metal-spike.md`, `docs/results/raw/2026-10-14-metal-prefill-p1/full/bench-pick.txt` and `bench-pair.txt`, `docs/decisions/0007-phase-1b-2a-2b-implementation-decisions.md`, `docs/backends/hip.md`, `docs/ARCHITECTURE.md` sections 4 and 3.2.
* Local reads: installed mlx-lm 0.31.3 (`mlx_lm/generate.py`, `mlx_lm/server.py`, speculative loop and `is_batchable`); llama.cpp b11429 binary strings (flag names only; behavior not run).
* High confidence: mlx-lm source behavior; llama.cpp speculative docs and issue #23752 numbers; vLLM and Thinking Machines statements about batch invariance; Qwen3 config.json vocab sizes; RedHatAI model card figures.
* Medium confidence: vLLM PR #24322 acceptance lengths (CUDA, TP size not read); prompt-lookup README numbers (author benchmarks).
* Low confidence, labelled where used: mlx-lm discussion #890 (LLM-written, unverified); secondary llama.cpp blog posts on few-row Metal kernels; vllm-mlx self-reported throughput; arXiv items seen only as summaries (an Apple-Silicon Polish-language study, arXiv 2604.16368, and 2609.24150, were not read and are not used for any number here); Tengyunw card.
* Not found anywhere: acceptance of stock Qwen3-0.6B/1.7B for the 8B target; any Metal batched-throughput benchmark for llama.cpp; a measured T(M)/T(1) for M = 2..15 on this backend.
* Everything marked "estimate" is arithmetic in this document, not a result. The tables in 4.2, 4.3 and 5.3 come from a short script over the stated formulas; the inputs (alpha, p, m, delta, draft cost) are the assumptions to replace first.
