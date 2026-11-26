# Consult answer (Fable): custom drafter and custom IntarsiaML quant for Inkling-2-Mini

Read-only, main `3c6ff64f0`. "Measured" = from a results doc; "modelled" = my arithmetic. Doc paths are under `docs/results/`.

## Verdicts

**Custom drafter: not now.** DSpark already gives 2.36-2.54 tokens / pass on the R9700 and the draft block is only 2.11 ms of a
~22.8 ms cycle (`2026-11-26-inkling2-hip-spec.md:96-103`). A retrained drafter can only move acceptance, and each +1 pp of
acceptance is worth about +1.0 % tok/s (modelled below). A fine-tune is plausibly +2-5 pp, so +2-5 %, for 25-40 agent-hours plus
an overnight endor slot. Three cheaper levers on the same path are each worth as much or more: verify cost per row on HIP (never
attributed per op), sampled-request speculation for DSpark (today sampled requests get none of the 19-55 %), and a prompt-lookup
extension of the draft. No results doc shows a drafter ever trained or fine-tuned in this repo (grep for distill / fine-tune
over `docs/results`, `docs/plans`, `docs/decisions` finds nothing relevant), so there is no prior to reuse either.

**Custom quant: yes, in two tiers.** Tier 1 (lossless, default-eligible): port P12 to HIP. It is our own format, already
default-on on Metal, and HIP lacks it (`2026-11-25-inkling2-hip.md:22`, `2026-11-25-inkling2-hip-mix.md:48`). Tier 2 (lossy,
opt-in): a new 8-bit dense format with a per-row scale ("I8R", linear int8 or a short-exponent float) to replace the current
opt-in fp8, whose e4m3 with one scale per pack spends 4 bits on exponent and only 3 on mantissa. Modelled noise power is 5-12x
lower at the same bytes. Experts, KV and mxfp4 dense: leave alone. A full "IntarsiaML checkpoint format" is not worth it for
speed; a small sidecar is worth it only once a calibrated lossy tier exists.

## 1. Custom drafter

**Arithmetic (R9700).** Measured: verify 2 / 3 / 4 rows = 16.48 / 19.00 / 20.80 ms, decode 13.30 ms, draft 2.11 ms
(`hip-spec.md:99-103`); cut 0.5 k3 on the greedy set: 57.5 % acceptance, 2.36 tokens / pass, 103.45 tok/s (`hip-spec.md:96,149`).
Modelled from those: accepted / pass 1.36, proposed / pass 1.36 / 0.575 = 2.37, rows 3.37, verify ~19.7 ms, + draft 2.1 = 21.8;
measured cycle 2.36 / 103.45 = 22.8 ms, so ~1.0 ms host. Holding proposals fixed, tokens / pass = 1 + 2.37 x acceptance:

| acceptance (modelled) | tokens / pass | tok/s short | vs today |
|---|---:|---:|---:|
| 57.5 % (measured) | 2.36 | 103.5 | - |
| 62 % | 2.47 | 108 | +4.5 % |
| 65 % | 2.54 | 111 | +7.7 % |
| 72 % | 2.71 | 119 | +14.8 % |
| perfect, k3 (ceiling) | 4.00 / 23.9 ms | 167 | +62 % |

**Why I expect only +2-5 pp.** (a) A numerics mismatch between vendor training and our W4A16 engine is not costing much:
changing the ingest rounding moved acceptance by <= 0.2 pp at k2 / k3 (`2026-11-21-inkling2-mtp.md:764-775`), and the Python
oracle's P(>=1) 0.722 (`mtp.md:405`) matches the engine's 71-74 % under the gate. (b) So the gain must come from domain
adaptation. That is real for some traffic (short chat prompts accept least: `hip-spec.md:160-161`; 7 of 15 short prompts lose
under the cut on Metal, `2026-11-26-inkling2-spec-cut.md:28`), but it is unmeasured.

**If you do it anyway.** Fine-tune DSpark, do not design a new architecture: the engine path (taps 5 / 23 / 41 / 48, 8-row
block, rank-256 Markov head, `crates/intarsia-core/src/model/dspark.rs:6-17`) and the torch oracle
(`tools/inkling2_dspark_ref.py`, imports torch) exist. Freeze the fold / unembed; train layers, fuse weights, Markov head.
Data: labels are the target's argmax, so mix ~1M self-generated tokens with ~4M teacher-forced tokens from real chat / code
traces. Human prose alone is the wrong distribution (oracle prose P(>=1) 0.408 vs 0.722 on own continuations,
`mtp.md:405-406,718-723`). Needs a new tap-dump path (4 x 2304 bf16 = 18.4 KB / token, ~90 GB for 5M tokens, modelled). Where:
endor PyTorch-MPS, one overnight exclusive slot (my guess, unmeasured). Not the R9700 (shared, <= 5 min windows). A
from-scratch drafter would need the Sparks: a separate ask. Train only after the target's quant default is settled (section 3).

**Cheaper levers on the spec path, checked against the docs.**

| Lever | Status in docs | Expected (modelled unless marked) |
|---|---|---|
| Verify cost per row on HIP | Only totals measured; the per-op lines were lost (`hip-spec.md:64`). 4-row byte floor is 8.76 ms + ~4.0 ms of extra expert bytes (union 8 -> 18.9, `mtp.md:417`; 10.9 x 4.48 MB x 44 visits = 2.15 GB, + ~0.37 GB MXFP8) = ~12.8 ms vs 20.8 measured. Excess 8.0 ms vs 4.5 ms at one row, so ~3.5 ms is row-dependent excess. On Metal the experts were 70 % of the growth (`mtp.md:594-596`) | recover 1.5-2 ms of ~19.7: +8-10 % spec tok/s, bitwise |
| Sampled speculation with DSpark | `SampledSpec` serves `--spec mtp` only and is off on HIP (`crates/intarsia-core/src/spec.rs:164-166,208-216`); DSpark takes plain steps on sampled requests (`hip-spec.md:185-187`); "most chat traffic is sampled" (`mtp.md:1288`). The softcap precondition test exists (`hip-spec.md:238-239`) | the largest user-visible gain: sampled traffic goes from 1.0x toward the greedy 1.2-1.5x (Match rule accepts with p_target(draft); smaller than greedy, unmeasured) |
| Prompt-lookup extension | `--spec ngram` exists, Metal only (`docs/tuning.md:46`); ran on Inkling in the screen gate (`2026-11-21-inkling2-fp8.md:278`). Not combined with DSpark anywhere | on copy-heavy spans (code edits, quoting, tool JSON) rows 5-8 at near-1 acceptance; 8 rows ~29 ms extrapolated (unmeasured past 4 rows) = >200 tok/s on those spans; zero elsewhere |
| Draft length policy | Done: cut 0.5 k3 is default on Metal and HIP; k5 was <= +1.5 % (`spec-cut.md:71-76`); EMA added nothing on greedy (`mtp.md:727`) | a HIP cut / k sweep: <= +2 %, one window |
| Tree / multi-candidate verify | Not tried. Inkling has short convs on K / V and sublayer outputs plus SWA rings (`recipes/inkling2.toml:4-5`); branches need per-branch ring state and break the linear rows-exact chain | a sibling row costs +1.8-2.5 ms (~10 % of a pass) for maybe +0.1 tokens: flat to negative. Don't |
| Union-fused expert down kernel (Metal) | Sized, parked: 1.0-1.5 ms per 4-row pass (`mtp.md:814-830`) | ~5 % on Metal spec; the HIP analogue is row 1 |

## 2. Custom IntarsiaML quant

**Tier 1, lossless: P12 on HIP (default-eligible).** Format in `crates/intarsia-core/src/quant/p12.rs`; -24.9 % bytes, bit-exact
on all 177 tensors (`2026-11-21-inkling2-pack12.md:135-137`). Bytes: projections 2.58 -> 1.94 GB, unembed 0.92 -> 0.69 GB, so
5.49 -> 4.62 GB / token; floor 8.76 -> 7.37 ms (136 tok/s short, modelled). HIP dense GEMVs are at 95-100 % of their floor
(in_proj 3.06 + wo 1.14 + unembed 1.44 = 5.64 ms, `2026-11-26-inkling2-hip-dec3.md:45-49`). Metal realised 1.23 ms of a 1.6 ms
paper saving (`pack12.md:250-252`); the same 77 % on HIP gives ~1.1 ms: **plain 76.9 -> ~84 tok/s (+9 %), spec short 103.5 ->
~106-108 (+3-5 %)**, all modelled. Risks: gfx1201 ALU cost of the unpack is unknown; the `_rows` kernels gain less at 4 rows
(1.07-1.15x on Metal, `pack12.md:222-229`) and lose at 16; keep the unembed split under the 1 ms stall (`hip-dec3.md:59-63`);
+1.66 GB resident beside the bf16 copies prefill reads (`pack12.md:285-286`), on a 32 GB card.

The exact fp8 head screen (default on Metal, `fp8.md:329`) is worth less than it looks under the spec default: it serves
one-row heads only and verify runs the full head (`fp8.md:267-268`). Port it to HIP only after P12.

**Tier 2, lossy: I8R dense (opt-in).** Evidence on today's lossy modes (`fp8.md:188-195`): fp8 attention costs KL 4.6e-3
against our own bf16 and fails G2 short (4.39e-3 vs 4e-3) for +21 %; the fp8 head alone costs 1.6e-4; mxfp4 fails G3. The fp8
mode is e4m3 with one scale per pack (`fp8.md:169`): RMS relative error ~3.6 % per weight (3 mantissa bits). P12's statistics
show a 16-binade window per row already holds all but 8.2e-5 of elements (`pack12.md:136`), so the exponent bits are mostly
wasted. Per-row int8 with an MSE-chosen clip has RMS error ~1 % of the row's RMS: 5-12x less noise power at the same 1 byte
(modelled, Gaussian-ish rows; outlier rows can reuse P12's escape list). If KL scales with noise power, attention KL falls from
4.6e-3 to ~4e-4-9e-4 and G2 short would pass (modelled; must be measured).

- Tensors: wq / gate / wk / wv / wo and the unembed. Keep `wr_du`, the sink and sconv projections lossless (P12): they are 45 MB
  and the fp8 lane names them as the sensitive ones (`fp8.md:224-226`).
- Calibration ladder, stop at the first that passes: (1) none, round-to-nearest with a per-row clip search; (2) GPTQ per layer
  (24 Hessians of 2304 and 24 of 3072, CPU minutes); (3) our own KL-driven bit allocation (8 / 12 bits per tensor) scored on
  G2 / G3. No AWQ: its scale folding collides with the q / k L2 clip and norms.
- Bytes: 5.49 -> 3.75 GB / token (as fp8, `fp8.md:193`). R9700 plain: dense 5.64 ms -> ~3.1 ms, **76.9 -> ~96 tok/s (+25 %),
  spec short ~115**, modelled. Metal against today's P12 + screen default: projections 1950 MB -> 1300 MB = -1.16 ms of 12.65:
  **79 -> ~87 tok/s (+10 %)**, modelled. fp8 measured +21 % on the pre-P12 baseline (`fp8.md:193`).
- Policy: changes greedy text, so opt-in. HIP has no e4m3 or P12 in-projection kernel (`hip-mix.md:48`), so a lossy tier there
  needs a new kernel regardless; build the better format once.

**Leave alone.** Experts: NVFP4 is the checkpoint's own format, and HIP expert kernels are latency-bound, not byte-bound (gu 50 /
dn 30 us against 38 / 19 us of bytes, `hip-dec3.md:199`). KV cache: 0.30 GB at 4k; fp8 KV saves ~0.24 ms (1.8 %) at 4k and ~4 %
at 16k (modelled), lossy on every token. mxfp4 dense: killed (`fp8.md:223-226,262`).

**Checkpoint format.** Not for speed: P12 packs in ~2.5 s at load and the lane rejected a sidecar for that reason
(`pack12.md:129-131`). It becomes worth it with tier 2, because a calibrated quant cannot be recomputed at load: ship a sidecar
safetensors (source-tensor hashes, format, calibration set, its G2 / G3 numbers) next to the original checkpoint, not a new
whole-checkpoint format. Side benefits: reproducible opt-in numerics across backends, and dropping the duplicate resident copies.

## 3. Interaction

- **Acceptance under a quantised target.** The drafter chases the quantised target's argmax. fp8 / fp8 keeps top-1 agreement with
  bf16 at 98.2-98.5 % (`fp8.md:207-208`), so the loss is bounded near 1.5-2 % of tokens / pass; with I8R it should be
  unmeasurable (the 0.2 pp ingest-rounding result is the analogue).
- **Verify cost.** Dense weights are read once per pass whatever the row count, so dense quant cuts the fixed part of a pass
  (19.7 -> ~17.3 ms with I8R, modelled) and leaves the per-row expert cost. Rows become relatively dearer: re-tune the cut after.
  Every new format needs `_rows` twins bitwise equal to the one-row kernel, as P12 and e4m3 have (`fp8.md:173`).
- **Quantising the drafter.** Free in policy terms (never changes output). The head is already MXFP4 (`dspark.rs:143-146`);
  `INTARSIA_DSPARK_LAYERS=fp8` exists opt-in (`docs/tuning.md:556`), with no measurement I could find. Ceiling ~0.4 ms of 22.8
  (< 2 %). Low priority.
- **Ordering.** Bitwise first (HIP P12, verify rows), lossy study in parallel on the cpu reference (no GPU windows, as the fp8
  lane did, `fp8.md:182`), drafter fine-tune last and against the final target numerics.

## 4. Plan

| # | Item | Effort | Gate | Expected gain |
|---|---|---|---|---|
| 1 | HIP verify-pass profile at 2-4 rows, then the top row-dependent excess (likely the `_rows` experts) | 2 h + 1 window; fix 8-12 h + 3 windows | bitwise: md5 = `--spec off`, rows-exact tests, G2 unchanged | +8-10 % spec decode (modelled) |
| 2 | P12 on HIP (in_proj, wo, unembed, `_rows` twins) | 12-16 h + 4-5 windows | P12 = bf16 twin bit for bit at rows 1..16, md5 equal, G2 equal; default-on | plain +9 %, spec +3-5 % (modelled) |
| 3 | Sampled DSpark speculation (Match rule), Metal first, then HIP | 10-14 h + 2 endor slots + 2 windows | sampled verify distribution = plain sampled decode; greedy md5 unchanged; no loss at t = 1.0 | sampled traffic from 1.0x toward greedy's gain (unmeasured) |
| 4 | I8R study on cpu, then one Metal kernel if it passes | 6-8 h CPU; kernel 10-12 h + 2 endor slots | G3 pass, and G2 short <= 4e-3 if it is to be offered as the fast mode; opt-in | Metal +10 %, R9700 +25 % plain (modelled) |
| 5 | Prompt-lookup extension of DSpark drafts | 6-8 h + 1 window + 1 endor slot | md5 = `--spec off`; no loss on the non-copy greedy set | large on copy-heavy spans, 0 elsewhere |

Drafter fine-tune is sixth: 25-40 h + an overnight endor slot, gate = acceptance up >= 3 pp on held-out traffic with ids
identical, +2-5 % (modelled, low confidence).

**Don't do**
- Train a drafter from scratch: needs the Sparks, weeks, and the ceiling is the acceptance arithmetic above.
- Tree / multi-candidate verify: conv and SWA ring state per branch, +10 % pass cost per sibling row.
- mxfp4 dense: failed G3 with confident-row misses (`fp8.md:194-195,262`).
- Re-quantise the experts below NVFP4: lossy on lossy, and HIP experts are not byte-bound.
- fp8 KV cache: < 2 % at 4k, lossy every token.
- A whole new on-disk checkpoint format for lossless packing: saves 2.5 s of load.
- More k / gate sweeps as a main lane: Metal showed <= 1.5 % left.
- Polish the existing per-pack e4m3 fp8 on HIP: build I8R instead if a lossy tier is wanted.

## 5. Not verified

The HIP P12 and I8R speeds, the I8R KL, the fine-tune's acceptance gain, the sampled-spec gain and verify cost past 4 rows are
all modelled; none has a measurement in the docs. I did not read `docs/accuracy.md` end to end; gate bounds are quoted from
`fp8.md:188` and `2026-11-25-inkling2-hip.md:165-166`.
