# Inkling-2-Mini skeleton on cpu (lane ink-skel, 2026-11-21)

Status: in progress.

## Review findings (Fable spec review, `.worktrees/ink-fable/spec-review.md`)

| # | Severity | Finding | Status in this lane |
|---|---|---|---|
| 1 | BLOCKER | tiny fixture never fires the q / k clip (theta 22.6 at head_dim 16) | spec lane regenerates the fixture (clip ~3); gate against the new files |
| 2 | BLOCKER | G1(b) vacuous after v00; per-visit gate must be teacher-forced (`ref_dump_bf16_tf`, blocks / bank, routing margins) | spec lane regenerates; tests follow the new G1 |
| 3 | MAJOR | tau never exercised on real weights (n_floor 8192) | out of this lane (needs a >8k golden); tiny fixture covers it |
| 4 | MAJOR | G2 / G3 thresholds without headroom; gate on KL + confident-row top-1 | follow the rewritten gates from ink/spec |
| 5 | MAJOR | W4A4 story (MXFP8 act rule rounds up; humming is not W4A16) | no W4A4 mode here; `act_quant` attr reserved on both expert ops, refused when set |
| 6 | info | 4k gap is near-ties, not a long-range bug | - |
| 7 | proposal | vLLM module-boundary capture | out of scope |
| 8 | MAJOR | ring semantics for multi-row calls | applied in the op contracts: margin 16 (>= 6), rings read in place (history before the call, chunk rows from the call's inputs), written after reading, last `rows` positions only; no linearizing copy |
| 9 | MAJOR | rows-exact needs the slot-dtype round on every value that is also state | applied: conv inputs and K/V enter the ops already in the slot dtype (= activation dtype, checked), every reader sees that value |
| 10 | MAJOR (perf) | ~700 launches per token | applied: `InkInProj` (one launch, row-range input select), `ConvMix` (conv + residual + next mix + optional blend + norm), KV write inside `RelSinkAttn`, all-layer message mix once per pass; table below |
| 10b | MAJOR | per-row position, MTP tap export, device argmax on raw logits | positions are `pos0 + t` per call (row-table form left to the batched lane); taps: see "Taps"; argmax on device runs on raw logits, softcap host-only (monotone) |
| 11 | MINOR | spec text (route_scale, two bank entries, two eps, bf16 probabilities) | route_scale is an attr; loader refuses a bank other than 2 entries; source eps fixed 1e-6 separate from `rms_norm_eps` |
| 12 | MINOR | oracle duplicate ring writes | n/a (Rust writes only the last `rows` positions) |
| 13 | MINOR | head_dim == d_rel in the tiny fixture | spec lane |

## Briefs from the coordinator (kept here so they survive compaction)

- 2026-11-21: merge ink/spec (6a42b90bf: regenerated tiny fixture, teacher-forced dumps, gates rewritten) before the tiny tests; gate on the new definitions.
- G1(b) budget = **3x** the oracle's own teacher-forced bf16 error (`tf_check(factor=3)`), not 2x.
- G2: KL21 mean <= 4e-3 (short), <= 6e-3 (long). G3: <= 1.05e-2 (short), <= 1.65e-2 (long), 1.4e-2 (4k). Plus confident rows (p1 - p2 >= 0.5): top-1 misses within the section i allowance (at most 1 per 2000).
- Metal lanes ink/m-attn, ink/m-mix, ink/m-moe branched from 3feffee6e: op contracts stay stable; message main before any change.
- Op 75 is reserved for lane ink-m-attn's `InkAttnDecode` (InkKvPrologue + RelSinkAttn fused, decode rows 1..16). This lane never appends at 75 (a new op of mine takes 76, after telling main); the model graph keeps emitting the pair 70 + 71.
