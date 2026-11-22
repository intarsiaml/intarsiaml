# Inkling-2-Mini skeleton on cpu (lane ink-skel, 2026-11-21)

Status: cpu skeleton done; G1, G2 and G3 (short, long, 4k) pass. Branch `ink/skel` (not pushed, not merged).
Op contracts unchanged since the first message to main (3feffee6e).

## Review findings (Fable spec review, `.worktrees/ink-fable/spec-review.md`)

| # | Severity | Finding | Status in this lane |
|---|---|---|---|
| 1 | BLOCKER | tiny fixture never fires the q / k clip (theta 22.6 at head_dim 16) | spec lane regenerates the fixture (clip ~3); gate against the new files |
| 2 | BLOCKER | G1(b) vacuous after v00; per-visit gate must be teacher-forced (`ref_dump_bf16_tf`, blocks / bank, routing margins) | spec lane regenerates; tests follow the new G1 |
| 3 | MAJOR | tau never exercised on real weights (n_floor 8192) | resolved by the spec lane (section k: vLLM 12k module-boundary replay, tau diff 6e-8); the tiny fixture (n_floor 4) covers it here |
| 4 | MAJOR | G2 / G3 thresholds without headroom; gate on KL + confident-row top-1 | applied: gated on the rewritten section i (KL21 mean, confident-row misses) |
| 5 | MAJOR | W4A4 story (MXFP8 act rule rounds up; humming is not W4A16) | no W4A4 mode here; `act_quant` attr reserved on both expert ops, refused when set |
| 6 | info | 4k gap is near-ties, not a long-range bug | - |
| 7 | proposal | vLLM module-boundary capture | out of scope |
| 8 | MAJOR | ring semantics for multi-row calls | applied in the op contracts: margin 16 (>= 6), rings read in place (history before the call, chunk rows from the call's inputs), written after reading, last `rows` positions only; no linearizing copy |
| 9 | MAJOR | rows-exact needs the slot-dtype round on every value that is also state | applied: conv inputs and K/V enter the ops already in the slot dtype (= activation dtype, checked), every reader sees that value |
| 10 | MAJOR (perf) | ~700 launches per token | applied: `InkInProj` (one launch, row-range input select), `ConvMix` (conv + residual + next mix + optional blend + norm), KV write inside `RelSinkAttn`; the cross-pass messages are one `AttnResMix` per looped visit (4 streams in one launch). 563 launches per decode token on cpu (`plan show`) |
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

## What was built

| Piece | Where |
|---|---|
| Ops 67..=74 (`AttnResMix`, `LoopBlend`, `SconvRes`, `InkKvPrologue`, `RelSinkAttn`, `MoeExpertsMxfp8`, `InkInProj`, `ConvMix`) + SiTU on `MoeExperts` (`KernelImpl::situ`, `ShapeInfo::situ`) | contract `crates/intarsia-core/src/inkling2/ops.rs`, f64 cpu reference `crates/intarsia-backend-cpu/src/ink_ref.rs`, sweeps `inkling2/cases.rs` (decode rows 1, 3, 7, 16, a 40-row chunk past the rings, real widths once) |
| Config | `inkling2/config.rs` (`ModelDims::ink`, refuses varloop, shared experts, a core that is not two whole blocks, differing SWA head shapes) |
| Block | `crates/intarsia-core/src/model/inkling2.rs`: weights per physical layer, the 48-visit table with every partial / block location fixed at load (`plan_visits`), workspace planes, the pass (`ink_pass`), teacher forcing and oracle-named dumps (`Model::ink_dump`) |
| Slots | per visit: SWA KV `Ring` (window + 16 rows, `RingKind::InkKv`, `[Hk, rows, Dh]` per plane) or full KV `AttnKv`; K/V conv window `Ring` (K-1+16 rows, 2 planes); attn / mlp conv window `Ring` (2 planes). 144 slots. Rings are in the act dtype (slot-dtype rule) |
| Loader | prefix `model.llm.`, vision and `mtp.safetensors` never read; stacked in-projection `[q | r | gate | sink | k (| k taps) | v (| v taps)]`; lag-major conv taps; queries x D^-0.5 (attn_res, mlp, output, cross-pass, streams q k v mlp); w13 de-interleaved (gate = rows 0::2); NVFP4 global scale = `amax / outer_scale_denominator` per expert; MXFP8 by stored dtype (e4m3 + E8M0 u8); embed / unembed are views of the first 200058 rows; softcap 30 on host logits |
| Recipe, docs | `recipes/inkling2.toml`, `docs/recipes.md` `[layer.ink]`, `recipe new` coverage arm (`builtin:inkling2 ... 37/37 tensor templates resolve; covers this model`), `docs/tuning.md` row for the test env var |
| Tokenizer / template | the HF codec: `tokenizer.json` (no BOS), `chat_template.jinja` (no generation prompt; the model opens `<|message_model|>`), EOS 200006 from config |
| Accuracy | turn format `inkling2` (one open token per role, `<|end_message|>` closes); the goldens' `template = chatml` metadata does not fit this tokenizer, so `accuracy` falls back to the recipe's `chat_turns` (no-op where the golden's format resolves) |
| Engine guards | Inkling-2 refuses TP, MTP / DFlash, batched and graph decode, the decode twin, layer pauses; a failed pass resets the sequence; `reuse_len` never returns a cut past the ring margin (no-op without ring slots); the prefix store refuses ring slots |

## Gates

| Gate | Result |
|---|---|
| G0 | `intarsia kernels test --backend cpu --op <op>`: all 8 + `moe_experts` (situ cases) ADMITTED (cpu fast = reference, 0 error) |
| G1(a) | f32 activations vs `logits_f64`: top-1 40/40, max abs 8.8e-4 (limit 1e-2; the f32 oracle 9.2e-4) |
| G1(b) | bf16 teacher-forced per visit, `tf_check` x3: 0 failures, every tensor of all 16 visits at about 1.0x the oracle's own bf16 error (so it also passes x2) |
| G1 state | chunks 7 and 13 and prefill 9 + token-by-token decode == one-shot prefill, bitwise; rollback by length (speculative tail), prompt reuse at 30 of 40, refusal past the margin, reset to 0; two interleaved sequences == solo; failed pass resets; NoKernel names each op the graph emits (`SconvRes` is not emitted: `ConvMix` covers it) |
| G2 short | KL21 2.994e-3 (<= 4e-3), confident misses 0 / 363 (<= 1), top-1 99.30 % |
| G2 long | KL21 3.879e-3 (<= 6e-3), confident misses 1 / 1206 (<= 2), top-1 99.04 % |
| G3 short | KL21 9.218e-3 (`accuracy --golden`, <= 1.05e-2; side script 9.06e-3), confident misses 0 / 367, top-1 97.89 % |
| G3 long | KL21 1.416e-2 (`accuracy --golden`, <= 1.65e-2; side script 1.410e-2), confident misses 0 / 1211, top-1 97.51 % |
| G3 4k | KL21 9.822e-3 (`accuracy --golden`, <= 1.4e-2; the oracle 9.84e-3), confident misses 2 / 8229 (<= 9), top-1 95.73 %; 51 min on cpu |

KL21 and confident misses: `intarsia accuracy --golden ... --mode prefill --write-self-golden`, the misses (and a KL21 re-check that
reproduces the tool's 2.994e-3 exactly) from the self-golden with a 30-line script against each golden (scratch, not committed).
The accuracy tool's own decode mode on G2 short reads 9.3e-3 KL over its 128 decode rows: those rows are bitwise the prefill rows
(128 / 128 identical), the subset just holds the chat_reasoning outlier (one row at 0.61 nats); G2 is defined on prefill rows.
The goldens' chat prompts are ChatML text tokenized as plain text (no Inkling turn tokens), so no row is masked.

Greedy on cpu, W4A16 (rounding none): raw "The capital of France is" -> " Paris.\n\nA. Noun clause\n\nB. Adjective clause" (the
oracle's continuation); chat "What is the capital of France?" -> `<|message_model|><|content_thinking|>`"The user asks a simple factual
question: capital of France. Need answer accurately. France's capital is Paris. ..." (the oracle's).

`plan show --model ~/models/Qwen3-8B --backend cpu`: byte-identical before / after (stdout and stderr).
`cargo test --release` over the workspace minus `intarsia-backend-metal` and the eight GPU test files of `intarsia-models`
(`batch_metal`, `batch_metal_real`, `batch_recurrent`, `gdn_verify_metal`, `hybrid_reuse_real`, `prefix_reuse_real`,
`mtp_store_metal`, `prefix_store_metal`: no GPU in this lane): all pass. `scripts/prepush.sh`: pass.

## Speed (cpu, endor M5 Max, f64 reference kernels)

Decode 1.75 tok/s (chat, 22-token prompt, 48 tokens; 0.27 before the SiTU experts ran their rows in parallel); prefill of the
long set (1569 positions, 256-row chunks, model load included) 621 s before grouping the prefill experts by expert, 278 s after
(5.6 tok/s; bitwise the same self-golden on the short set). Weights 13.3 GB, peak RSS about 28 GB.
Top ops per decode token: `moe_experts` 22.6 s / 2112 calls, `rel_sink_attn` 2.7 s, `conv_mix` 2.2 s, `ink_in_proj` 2.1 s.

## Not done / limits

- Metal and CUDA kernels (other lanes). Metal registers none of ops 67..=74 (asserted from its kernel table); `qualify`'s plan stage
  on Metal was not run (no GPU in this lane). CUDA lists them in `NOT_ON_CUDA`, HIP in `UNIMPLEMENTED`.
- Batched decode, decode graphs, TP, MTP / DSpark speculation, prefix store: refused with an error, not implemented.
- W4A4 activation mode: `ink.act_quant` reserved, refused when set.
- `SconvRes` (69) has a reference and sweeps but no caller in the graph.

## Friction

- The goldens say `template = chatml`, which this tokenizer cannot resolve; chat prompts in them are ChatML text, not the Inkling
  template, so the "chat" rows test plain text.
- `accuracy` has no confident-row metric; the G2 / G3 confident-row counts needed a side script over `--write-self-golden`.
- The f64 cpu experts were the long pole (decode 0.27 tok/s); parallel rows and per-expert grouping keep the reference bits.
