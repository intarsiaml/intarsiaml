# Inkling-2: exact speculative decoding for sampled requests with the DSpark draft (lane ink-sampled)

Branch `ink/sampled-dspark` from `origin/main` `3226af2b9`. Status: in progress.

## Briefs received

### Brief 1 (coordinator, 2026-11-26 ~14:05 EST)

Goal: exact speculative decoding for sampled requests (temperature > 0) with Inkling-2-Mini's DSpark drafter (`--spec dflash`), on Metal
(endor) and HIP (one R9700, tatooine). Today sampled requests take plain steps under DSpark (`2026-11-26-inkling2-hip-spec.md`
185-187), and `SampledSpec` (`INTARSIA_SPEC_SAMPLED`) serves `--spec mtp` only (default on for cuda / metal, off on hip). Serving defaults
to temperature 1.0 for this model, so most real chat traffic gets none of DSpark's 19-55 % greedy gain. Three consultants rank this the
largest lever for real traffic (`docs/plans/2026-11-26-inkling2-drafter-quant.md`, modelled +15-45 % on sampled traffic, unmeasured).

Requirements:
- The output distribution must be exactly that of plain sampling from the target (the contract `SampledSpec::Match` has for MTP). Read
  how Match works for MTP first and reuse it; a new rule only if Match cannot apply to a block drafter, explained here before building.
- Verify logits go through the same final softcap and sampling transforms (temperature, top-p / top-k, repetition settings the serving
  path applies) as a plain sampled step; check the sampled path for the capped-vs-uncapped mistake #527 fixed in `verify_check`.
- Say which quantities the rule needs and where they come from (DSpark's proposal probabilities if the rule uses them).
- Greedy (temperature 0) stays bitwise identical to main: md5 at short and 4k, 64 / 256 tokens, spec off and spec default, both backends.
- Policy: opt-in on both backends, documented in `docs/tuning.md`; no default changes (the user decides default-on after the numbers).

Gates:
1. Law test: chi-square (or equivalent) of the emitted-token distribution against plain sampling at temperature 0.6 and 1.0 (the
   recipe's top-p), on the tiny fixture (cpu) and on the real model on each GPU backend; p-values and sample counts.
2. Seeded run-to-run: same seed, same output within a backend; spec on vs off differ only in RNG consumption, not in distribution.
3. Greedy md5 unchanged. Tiny tests `inkling2`, `inkling2_dspark`, `inkling2_dspark_cut`, `inkling2_verify_softcap` pass on cpu, Metal and
   hip as on main.
4. Speed: sampled decode tok/s at temperature 1.0 and 0.7, spec on vs off, paired processes (4 + 4) at short / 1k / 4k, plus acceptance
   per pass. endor: 45 s GPU-idle gap before every measured process, `macmon pipe -i 250`, compare within a power group only.

Overlaps: this lane owns `crates/intarsia-core/src/spec.rs` and `src/model/dspark.rs`. ink-hip-p12 owns `kernels/hip/ink_proj.hip`,
matmul GEMVs, `quant/p12.rs`; ink-hip-batch (#529) `kernels/hip/ink_attn*` / `ink_mix*`; ink-hip-verify `kernels/hip/ink_moe.hip`,
`ink_route*`, `ink_spec.hip` (report before touching). `--max-batch N > 1` keeps plain batched steps.

Fleet: endor Metal GPU in `--exclusive` slots (at most 4, each <= 30 min); HIP per COMMON-1126 (builds `--load cpu`, GPU inside
`--takes-down radiance` windows, at most 3 windows, each <= 5:00, <= 12:00 total; no `--mem` on tatooine); pushes in an endor
`--load cpu` slot. No tool call over 4 min.

Deliverables: this doc; a PR (title `ink-sampled: ...`), opt-in, `docs/decisions/LOG.md` row marked proposed, not merged.

## 1. The rule, and why it is exact

**Match applies to the block drafter unchanged; no new rule.** `SampledSpec::Match` (MTP, `crates/intarsia-core/src/spec.rs`) is
speculative sampling with a point-mass draft: drafts are the drafter's argmax; at verify row `i` draw `x_i` from the target's processed
distribution `p_i`; keep the draft iff `x_i == d_i`, else emit `x_i` and end the pass. The token emitted at every position is `x_i`, a
draw from `p_i` given the tokens before it, so the output is plain sampling's in law whatever the drafts are, provided (a) the drafts
are fixed before the draws they are compared with (they may depend on the context only, never on the current pass's draws) and (b)
each row's `p_i` is exactly the plain step's distribution at that position.

DSpark satisfies (a): its block (anchor + 7 mask rows over the target's taps) and its Markov head chain are greedy (argmax, `prev` =
the previous draft), and how many drafts it keeps (Metal / HIP: the cumulative cut 0.5; elsewhere the gate 0.8) depends on the draft's
own top-1 probabilities only. Nothing in the draft reads a draw. The block structure does not matter for Match: the rule only needs the
draft ids in order.

(b) is the verify pass's row-exactness (row `t`'s logits bitwise the decode step's at `st.len + t`; `inkling2_verify_softcap` on cpu /
Metal / CUDA / HIP) plus the same transforms after the logits: the final softcap `30 tanh(x / 30)` (`Model::softcap_logits`, the
function a decode readback uses; the verify head leaves pre-softcap logits in `ws.logits`, the #527 trap), then the request's
temperature / top-k / top-p through the request's own sampler. The serving path applies no repetition, presence or frequency penalty,
min_p or logit_bias (`Seq::head_mode`'s comment: the server drops them on both paths); stop ids are masked only inside `min_tokens`,
where `plan_step` never drafts; constrained and logit-traced requests never speculate.

**Quantities the rule needs.** The draft ids (from `Model::dflash_draft`, the same call greedy drafting makes) and, per verify row, the
target's draw at that position. **No proposal probabilities `q` of the draft**: a point-mass draft needs none. `SampledSpec::Reject`
would need sampled drafts and `q` from DSpark's Markov head per step; it is not built for DFlash (refusal is by construction: the
DFlash path only ever takes the keyed Match form).

**The form used: Match with the plain step's own draws (the keyed form, `engine/keyed.rs`).** The MTP single-sequence path draws Match's
`x_i` with `sampler::dist` + `Sampler::uniform`, a different routine from a plain step's, so a seed's text changes and the RNG position
depends on how many rows each pass looked at. The batching engine's MTP path (lane mtp-sampled) instead draws each row with the plain
step's own code and RNG (`Seq::token_from`, one draw per emitted token, rows after the first mismatch never drawn). This lane uses that
form in the single-sequence engine for `--spec dflash`:

- **Host sampler** (HIP and cpu; Metal requests with top-k / top-p): row `i` is read back as its logits before the softcap
  (`Model::forward_verify_logits_uncapped`), capped with `Model::softcap_row` (= `softcap_logits`) right before its draw, drawn by
  `Sampler::sample` (or `sample_topk`) exactly as a plain step's row; the rows behind the first mismatch are never capped or drawn.
- **Device sampler** (Metal, the recipe's `gumbel_sampling`: plain-temperature requests, i.e. what `serve` gives a request without
  sampling fields for this checkpoint: temperature 1.0, top-p 1.0, top-k 0, `SamplingDefaults::effective`): one `GumbelArgmax` over
  the verify head's `[q, V]` pre-softcap logits with the op's softcap (`Model::softcap_cap`, as the decode step's `GumbelArgmax`), row
  `r` keyed with the position of the token it draws (`st.len + r + 1`; a plain step at `st.len = L` uses `L + 1`), `q` tokens read back
  (`Model::forward_verify_gumbel`). The request's plain steps keep `HeadMode::LastGumbel` while it speculates (`Seq::keyed_gumbel`).

Both make the output **seed-exact**: the token at every position is the one the plain step at that position draws (same logits bits,
same draw), so a seeded request's ids are the same with drafting on or off. Exactness in law follows from plain sampling's. Seeded
equality is therefore gate 2's check (stronger than "differs only in RNG consumption"), and on HIP it doubles as the bitwise check of the
logits read back after the captured verify replay, the stated reason `SampledSpec` stays off there.

**Opt-in.** `INTARSIA_SPEC_SAMPLED_DFLASH=1` (`Model::set_spec_sampled_dflash` in tests), default off on every backend
(`spec::sampled_dflash_default_for`). A separate variable because `INTARSIA_SPEC_SAMPLED`'s Metal / CUDA default (`match`, for `--spec
mtp`) would otherwise switch this on with nothing set. Single-sequence engine, one rank, `--spec dflash` only (`--spec auto` and the
batching engine unchanged: `Seq::enable_keyed` stays MTP-only, so `--max-batch` > 1 takes plain batched steps as before).

Shared files touched (add-only): `crates/intarsia-core/src/engine.rs` (`Seq::new`, `Seq::head_mode`, a field), `engine/keyed.rs`
(draft call, row read, lazy cap), `model.rs` (two `VerifyRead` forms, `forward_verify_logits_uncapped`, `forward_verify_gumbel`,
`softcap_row`, `verify_gumbel_ok`, a field), `model/mtp.rs` and `model/batch.rs` (the setter and the field), `crates/intarsia-models/src/load.rs`
(startup line). No kernel changed.
