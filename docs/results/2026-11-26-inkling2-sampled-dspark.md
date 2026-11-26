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

## 2. Correctness gates

### 2.1 Tiny fixtures (gate 3, and gate 2 on the fixture)

`tests/inkling2_dspark_sampled.rs` (lane): `sampled_dspark_seeded_ids_equal_spec_off`: for configurations T 1.0 / top-p 1 (the recipe),
T 0.6 / top-p 1, T 0.7 / top-k 20 / top-p 0.9, three prompts, `max_tokens` 1 / 5 / 40 and seeds 1 / 7, the ids with sampled DSpark
speculation on equal `--spec off`'s, with the random tiny draft and with scripted proposals (the spec-off continuation with one token
spoiled at a moving offset, so every accepted length 0..=k happens; k 3 and 7; and the same proposals under another seed, where they
are mostly wrong). The cpu (#12122) and hip (W1a) runs had the first version of this test (k 2 / 3 / 7, `max_tokens` 1 / 2 / 5 / 40,
one engine per prompt); Metal and the final branch have the trimmed one (k 3 / 7, `max_tokens` 1 / 5 / 40), cut for the 5-minute HIP
windows. `sampled_dspark_is_opt_in_and_leaves_greedy_alone`: with the knob unset or `0` a sampled request never drafts and a
greedy one does; with it on, greedy ids equal `--spec off`'s. `tests/inkling2_dspark_sampled_gumbel.rs`: the seeded test with the
device sampler on the cpu backend (`INTARSIA_GUMBEL_SAMPLING=1`, reference `GumbelArgmax`; asserts `verify_gumbel_ok`). On Metal the
recipe's device sampler is on, so `inkling2_dspark_sampled` there covers `forward_verify_gumbel` for the plain-temperature configurations
and the host path for the top-k one.

| suite | cpu (endor `--load cpu`, #12122) | Metal (endor `--exclusive`, #12134) | hip (tatooine, card 0) |
|---|---|---|---|
| `inkling2` | 17 / 17 | 17 / 17 | see below |
| `inkling2_dspark` | 4 / 4 | 4 / 4 | 4 / 4 (W1a) |
| `inkling2_dspark_cut` | 1 / 1 | 1 / 1 | 1 / 1 (W1a) |
| `inkling2_verify_softcap` | 1 / 1 | 1 / 1 | 1 / 1 (W1a) |
| `inkling2_dspark_sampled` (lane) | 2 / 2 | 2 / 2 | 2 / 2 (W1a) |
| `inkling2_dspark_sampled_gumbel` (lane) | 1 / 1 | 1 / 1 | not run (no device sampler on hip) |

HIP `inkling2`: with the default parallel test threads both trees fail a varying set of loads, the same way: branch 6 / 17 and main
12 / 17 in W1a, branch 12 / 17 and main 6 / 17 in W2 (card 0 alone). Every failing load is `NoKernel` for `head_screen` /
`logits_gather` / `screen_argmax`: the screened-head tests set `INTARSIA_INK_HEAD_SCREEN=1` process-wide (`load_screen`,
tests/inkling2.rs:90) and other threads' loads race into it (HIP has no screen kernels; ink-hip-spec recorded the screened pair as
failing by design and got 15 / 17). A harness race, the same on main; W3 runs both trees with `--test-threads=1`.

### 2.1b Law test on the fixture (gate 1, cpu)

`tests/inkling2_dspark_sampled.rs` `law_t10_recipe_top_p`, `law_t06_recipe_top_p`, `law_t07_top_k20_top_p09` (ignored by default; cpu
backend, run on tatooine's CPU in a `--load cpu` slot, fleetq #12159, 84 s): prompt 0 of the fixture, 4 tokens per request, n = 1500
requests per arm with disjoint seed sets (spec off: seeds 0..1500; sampled DSpark on: from 1 000 000). The draft is scripted to the
most frequent token of a 300-request spec-off pilot at each position, so a draft is kept with that token's probability and both branches
of the rule run (accepted 1008 of 3870 proposed at T 0.6, 414 of 4211 at T 1.0, 973 of 3903 at 0.7 / top-k 20 / top-p 0.9). Chi-square
of homogeneity per generated position 2..4 (cells as in 2.2); the test fails on any p <= 1e-3 or a Fisher p <= 1e-3.

| T, top-k, top-p | p at positions 2 / 3 / 4 | df | Fisher p |
|---|---|---|---|
| 1.0, 0, 1.0 (recipe) | 0.198 / 0.291 / 0.515 | 60 / 89 / 113 | 0.318 |
| 0.6, 0, 1.0 (recipe top-p) | 0.705 / 0.095 / 0.155 | 26 / 49 / 61 | 0.166 |
| 0.7, 20, 0.9 | 0.134 / 0.876 / 0.410 | 20 / 39 / 59 | 0.416 |

### 2.2 Law test on the real model (gate 1, GPU backends)

Harness `tests/inkling2_sampled_real.rs` (one arm per process): Inkling-2-Mini, prompt "Once upon a time, in a small village by the sea,
there lived" (raw, no template), `--spec dflash --spec-k 3` with the recipe's draft rule (cut 0.5 on Metal and hip), 6 tokens per request,
top-k 0, top-p 1.0 (what `serve` gives a request without sampling fields for this checkpoint), one request per seed and temperature.
Arm off: `INTARSIA_SPEC_SAMPLED_DFLASH=0` (plain sampled steps), seeds 0... Arm on: `=1`, seeds from 1 000 000 for the law (disjoint
from the off arm's) plus the shared seeds 0.. for the seed-equality check. Token 1 comes from prefill in both arms; positions 2..6 are the
speculative ones. `raw/law.py`: Pearson chi-square of homogeneity per temperature and position (tokens with pooled expected count >= 5
in the smaller arm are cells, the rest go to 16 bins by `id % 16`, bins still below 5 form one remainder cell), and Fisher's combination
over the five positions of one temperature (a summary: the positions are dependent within a request).

| backend | T | n off + n on | p at positions 2 / 3 / 4 / 5 / 6 | Fisher p | shared seeds with identical ids |
|---|---|---|---|---|---|
| Metal (device sampler: `forward_verify_gumbel`) | 1.0 | 506 + 917 | 0.382 / 0.098 / 0.785 / 0.753 / 0.746 | 0.608 | (none in this run; see the speed A/B: 8 / 8 processes per temperature) |
| Metal | 0.6 | 506 + 917 | 0.607 / 0.778 / 0.906 / 0.916 / 0.896 | 0.996 | |
| hip (host sampler, lazy cap) | 1.0 | 489 + 719 | 0.535 / 0.061 / 0.668 / 0.486 / 0.675 | 0.452 | 150 / 150 |
| hip | 0.6 | 489 + 528 | 0.407 / 0.598 / 0.879 / 0.883 / 0.968 | 0.971 | 100 / 100 |

No p-value below 0.06 in 20 tests; nothing distinguishes the arms. Distinct tokens per position: 21-66 at T 0.6, 106-348 at T 1.0.
Power is modest (about 500-900 requests per arm and temperature, sized to the 5-minute HIP windows and the 30-minute Metal slot): the
seeded equality below is the sharper check, and it holds exactly. Metal: slot #12134, arm off 15:03:06-15:10:34 (stopped on time, 1012
lines), arm on 15:10:34-15:23:53 (stopped at the slot's end, 1834 lines). hip: arm on in W1a (card 1, 759 lines before the abort) and W3
(card 1, 838 lines, budget 125 s), arm off in W3 (card 0, 978 lines); the shared seeds compare card 1 (on) with card 0 (off).

### 2.3 Seeded run-to-run and spec on vs off (gate 2)

The rule is seed-exact, so the check is ids equality, not only equality in law:

- Tiny fixtures (2.1): every seeded request on cpu, Metal and hip, both samplers, k 2 / 3 / 7.
- Real model, hip: the 250 shared (seed, temperature) pairs of 2.2, identical between arms and between cards; the speed A/B (3.1): for
  each temperature all 8 processes (4 off, 4 on, two cards) emit identical ids on all 6 prompts x 128 tokens.
- Real model, Metal: the speed A/B (3.2), same check.
- Run-to-run within a backend: the 4 processes of an arm emit the same ids (speed A/B), on both backends.

### 2.4 Greedy is unchanged (gate 3)

Greedy `generate --show-ids`, 256 tokens, `md5.json` = 6 short prompts of the golden set + the 4108-id prompt, main (`3226af2b9`) vs this
branch (`br`, before the merge of #529), `--spec off` and the recipe default (DSpark k3, cut 0.5). md5 of the first 64 and of all ids:

- **Metal** (slot #12134): all 7 prompts identical across main-off, main-def, br-off, br-def at 64 and at 256 tokens (`raw/metal-A/md5-table.txt`;
  e.g. len4k `0f621a97` / `9e85ba11`, count `9d58b98d` / `9f731a14`). Spec counters of main-def and br-def identical (582 passes, 814 of
  1379 accepted).
- **hip** (W3, card 0 main, card 1 branch): all 7 identical across the four runs at 64 and 256 (`raw/hip-w3/md5-table.txt`; len4k
  `0f621a97` / `d575dd13`). Spec counters main-def = br-def (508 passes, 853 of 1274 accepted).

(Two chat prompts stop at EOS before 256 tokens, the same length in every run.)

## 3. Speed (gate 4)

Sampled decode, `generate --prompt-ids speed.json --max-tokens 128 --ignore-eos --top-k 0 --top-p 1.0 --seed 7 --temperature T`, the
recipe's DSpark default (no `--spec` flag: `--spec dflash`, k3, cut 0.5), arm off = `INTARSIA_SPEC_SAMPLED_DFLASH=0` (today's behaviour:
sampled requests take plain steps), arm on = `=1`; same binary. `speed.json`: a warm-up prompt (left out), three short prompts (36 /
40 / 101 ids, golden set), the 1021-id and the 4108-id prompts of the earlier Inkling-2 spec lanes. One process per arm and rep, 4 reps,
the arm order alternating per rep (odd: off1 on1 off07 on07; even: reversed), pairs (on, off) within a rep. tok/s is the engine's own
`decode[...]` clock; short = the three short prompts pooled (sum of tokens / sum of time). CI = t-interval over the 4 paired log ratios.
Acceptance and tokens per pass from the per-prompt `spec[...]` lines this branch adds to `generate`. Since every rep emits the same tokens
(same seed, seed-exact rule), the CI is timing noise over one token stream per prompt: it says nothing about other seeds or prompts.

### 3.1 HIP (one R9700 per process; W2, fleetq #12146, binary `br2` = base + this lane)

Both cards ran at once, the same arm at the same time (card 0 reps 1-2, card 1 reps 3-4), each pair within one card.

| T | class | off tok/s | on tok/s | on / off | 95 % CI | acceptance | tokens / pass |
|---|---|---:|---:|---:|---|---:|---:|
| 1.0 | short | 70.07 | 82.12 | **1.172** | 1.163-1.181 | 49.0 % | 2.12 |
| 1.0 | 1k | 68.62 | 73.45 | **1.070** | 1.049-1.092 | 49.5 % | 2.04 |
| 1.0 | 4k | 67.40 | 95.82 | **1.422** | 1.386-1.458 | 73.1 % | 2.88 |
| 0.7 | short | 70.25 | 83.18 | **1.184** | 1.179-1.189 | 50.7 % | 2.16 |
| 0.7 | 1k | 68.47 | 73.53 | **1.074** | 1.060-1.088 | 51.7 % | 2.07 |
| 0.7 | 4k | 67.33 | 93.97 | **1.396** | 1.378-1.414 | 76.8 % | 2.90 |

Per short prompt at T 1.0: 1.237 / **0.935** / 1.467 (acceptance 48 / 29 / 68 %); at 0.7: 1.132 / 1.015 / 1.505. The story prompt at
T 1.0 loses 6.5 %: at 29 % acceptance (1.53 tokens per pass) a verify pass costs more than the 1.5 plain steps it replaces. This lane
adds no brake for sampled requests beyond the draft's own cut (the rule the greedy default uses); a per-request back-off on low
acceptance is a possible follow-up. For reference, greedy on the md5 set (W3): 76.7 tok/s spec off, 109.4 tok/s DSpark default.

### 3.2 Metal (endor, M5 Max)

**Not measured yet.** The exclusive slot (`.scratch/metal/slotB.sh` = `raw/metal/slotB.sh`: the same A/B, 45 s GPU-idle gap before
every process, `macmon pipe -i 250` for the slot, `raw/speed.py --macmon` groups processes by GPU power) is queued as fleetq #12167
behind ink-q8's cpu accuracy jobs #12149 / #12150 (estimated start 22:51; a reservation at 16:30 was refused because it collides with
them). The Metal path's correctness is in (tiny suites, the law test with the device sampler, greedy md5); its speed is the open item.

## Fleet log (fleetq ids, radiance downtime, endor slots)

| fleetq | host / kind | what | result |
|---|---|---|---|
| #12112 | endor `--load cpu` | `cargo check` core + models tests | ok |
| #12120 | endor `--load cpu` | first build + run of the lane's tiny tests | seeded test failed on its own length assertion (a held-back text tail is one more `Token` event repeating the last id, on and off alike); fixed |
| #12122 | endor `--load cpu` | cpu tiny gates (6 suites) + fixture law test | see section 2 |
| #12124, #12132, #12142 | tatooine `--load cpu` | HIP builds (branch `br1`, main, branch tests `br2`) | ok |
| #12129, #12141 | endor `--load cpu` | Metal CLI branch + main, Metal test binaries | ok |
| #12136 | tatooine, radiance down | W1a (tiny suites hip + real-model law, arm on) | **aborted by me**: I released the job (meant to requeue it) seconds after it had started; fleetq stopped the local ssh, the remote window kept running without a slot; stopped in #12139 (`--load none`). Radiance down 14:43:37 to 14:45:58 (then held down by #12138, ink-hip-p12's own window): **2:21** on this lane's account |
| #12139 | tatooine `--load none` | stop the orphaned W1a window | done; no ink-sampled process or container left |
| #12134 | endor `--exclusive` slot 1 (14:55:11-15:23:53, 28:42) | Metal tiny suites, greedy md5 main / branch, real-model law (arm off stopped 15:10:34, arm on at the slot's end) | all pass; 2.1, 2.2, 2.4. My orphaned cpu law process from #12122 (released at 14:54) ran on until 14:56, inside this slot's first 1.5 min (killed; it only touched the CPU, during the tiny suites, not during md5 or law) |
| #12138 | (ink-hip-p12) | held radiance down after W1a | radiance came back at its end (fleetq: radiance-rank0 `up` at 14:54) |
| #12146 | tatooine, radiance down (W2) | inkling2 tiny alone (parallel threads), speed A/B both cards | radiance down 15:00:37-15:03:52: **3:15**; back ("radiance answers ... after 15 s") |
| #12156 | tatooine, radiance down (W3) | inkling2 tiny `--test-threads=1` branch / main, greedy md5 main / branch, real-model law | radiance down 15:16:04-15:19:29: **3:25**; back |
| #12159 | tatooine `--load cpu` | fixture law tests (cpu backend) | 3 / 3 pass; 2.1b |
| #12165 | endor `--load cpu` | host checks on the merged branch: `docs_env_vars` 2 / 2, `recipes_doc` 3 / 3, core lib 268 / 268 (4 ignored), models `--tests` check | ok (`cargo fmt --check` reports only files this lane does not touch; the repo is not rustfmt-clean) |
| #12157 / #12167 | endor `--exclusive` slot 2 | Metal speed A/B | queued (#12157 cancelled for a reservation that fleetq refused; requeued as #12167) |

Radiance: 3 windows (W1a, W2, W3), **9:01 in total** (2:21 + 3:15 + 3:25), each under 5:00; radiance answered after every window.
endor exclusive slots: 1 used (28:42), 1 queued.
