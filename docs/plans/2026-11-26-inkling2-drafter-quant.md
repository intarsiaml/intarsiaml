# Inkling-2-Mini: custom drafter and custom IntarsiaML quant (consensus, 2026-11-26)

The user asked: "what about a custom drafter and custom intarsiaml quant. consult with the best gemini and claude models and give a
consensus", then "ask gemini carbon also". Three independent read-only consultants answered the same brief
([brief](2026-11-26-inkling2-drafter-quant/brief.md)) against main `3c6ff64f0`:

- Claude Fable 5.2 ([answer](2026-11-26-inkling2-drafter-quant/fable.md))
- Gemini 4 Neon High ([answer](2026-11-26-inkling2-drafter-quant/gemini-neon.md))
- Gemini 4 Carbon High ([answer](2026-11-26-inkling2-drafter-quant/gemini-carbon.md))

The coordinator checked the claims below against main. All gains are modelled unless marked measured.

## Verdict

**Custom drafter: no, not now (3 of 3).** The checkpoint already ships a drafter trained for this model: DSpark, a 3-layer block
drafter with 4 target taps and a Markov head. Its acceptance is 57-74 % (2.3-2.7 tokens a pass) on the R9700, and it is ~6-8 % of a
cycle. A fine-tune could add maybe +2-8 % greedy decode. It needs tens of GPU-hours of rollouts, which tatooine's <= 5 min windows
and endor's 140 W budget cannot give: Neon and Carbon say the Sparks (a separate ask); Fable says an overnight endor slot. Training
from scratch: no (3 of 3). Revisit only after the target's numerics are final.

**Custom quant: yes, in two tiers (3 of 3).**

1. **Lossless, default-eligible:** port P12 (our 12-bit lossless packing of in_proj, wo and the unembed, default-on on Metal) to
   HIP. -0.88 GB of the 5.49 GB/token floor: about +9-12 % plain decode and +3-7 % spec decode on the R9700.
2. **Lossy, opt-in:** an 8-bit dense + unembed format with per-row scales. Today's opt-in fp8 uses one e4m3 scale per pack (three
   packs, `crates/intarsia-core/src/model/inkling2.rs:603-610`). -1.74 GB/token: about +25 % plain on the R9700 and +10-16 % on
   Metal. A calibrated mixed NVFP4 + fp8 tier (Neon, Carbon) could reach -2.3 to -2.5 GB/token.

**No new container format (3 of 3).** Standard safetensors plus metadata, and only for calibrated quants (a calibration can't be
recomputed at load) and load-time caches (Neon and Carbon: DSpark's 2-6 s fold, the `w13` de-interleave).

**Cheaper levers beat a custom drafter (3 of 3).** Sampled DSpark speculation, prompt lookup in front of DSpark, a HIP draft-length
sweep and (Fable) cheaper verify rows.

## Coordinator checks (against main)

| Claim | Check |
|---|---|
| HIP has no P12 | true: `docs/results/2026-11-25-inkling2-hip.md:22`, `-hip-mix.md:48`; `quant/p12.rs:360` gates it to metal/cuda |
| HIP dense GEMVs already at the byte floor | true: `docs/results/2026-11-26-inkling2-hip-dec3.md:45-49` (in_proj 3.06, wo 1.14, unembed 1.57 ms floors) |
| Sampled requests get no DSpark speculation | true: `SampledSpec` is `--spec mtp` only, and `default_for` is off on HIP (`crates/intarsia-core/src/spec.rs:164-216`) |
| Prompt lookup is Metal-only and not combined with DSpark | true: `docs/tuning.md:46` |
| fp8 opt-in: one scale per pack | true: three packs `[q,r,gate,sink]`, k, v, each with one scale (`inkling2.rs:603-610`) |
| fp8 / fp8 passes G3 but not G2 short | true: G2 short 4.39e-3 against the port bar 4e-3 (`docs/results/2026-11-21-inkling2-fp8.md:193`, bar `docs/plans/2026-11-21-inkling2-port.md:351`) |
| `INTARSIA_DSPARK_LAYERS=fp8` was never measured (Fable) | **wrong**: measured slower on Metal, 63.98 vs 68.50 tok/s (`docs/results/2026-11-21-inkling2-mtp.md:508`) |
| HIP k5-7 gives +12-18 % (Neon) | **discounted**: it comes from the oracle table (`2026-11-21-inkling2-mtp/analysis.md:56-61`), which over-predicted Metal, where k5 measured <= +1.5 % (`2026-11-26-inkling2-spec-cut.md:71-76`). HIP's cost per row past 4 rows is unmeasured, so one window decides it |
| P12 default-on on CUDA (Neon, Carbon) | partly: the CUDA in_proj P12 is off by default (decode there is ALU-bound, `docs/tuning.md:216`). That same unpack cost is the main risk on HIP |

**Correction found while checking (not a consultant claim).** After the HIP prefill flip (#526), G2 short is 4.491e-3, over the port's
bar of 4e-3 (before the flip: 2.566e-3). The flip lane graded against the accuracy tool's generic 4-bit limit 3e-2. Lane ink-hip-g2
(`docs/results/2026-11-26-inkling2-hip-g2.md` on branch `ink/hip-g2`) attributes all of it to the flash prefill attention default:
`INTARSIA_HIP_INK_ATTN_FLASH=0` alone gives 2.566e-3. MoE WMMA and chunk 2048 change nothing on short. 96 % of the delta is in two
chat prompts. Confident misses stay 0 and top-1 rises 98.95 -> 99.47 %. G2 long (4.083e-3 / 6e-3) and G3 short / long / 4k (8.565e-3,
1.197e-2, 9.964e-3; 0 / 0 / 3 misses) pass at the defaults. Keeping flash on by default is the user's call.

## Plan (ranked)

| # | Lane | Kind | Gate | Expected (modelled) |
|---|---|---|---|---|
| 1 | P12 on HIP (in_proj, wo, unembed, rows 1..16) | bitwise; default-on on gates | a microbench first (P12 vs bf16 GEMV at 1 / 4 rows on gfx1201; stop if the unpack eats the gain); bitwise against the bf16 twins at rows 1..16; md5 = main with spec off and on; G2 unchanged; 4 + 4 ABBA | +9-12 % plain, +3-7 % spec |
| 2 | HIP verify profile at 1..8 rows, k / cut sweep, then the largest row-dependent excess | bitwise; default-on on gates | md5 = `--spec off`; rows-exact tests; ABBA | 0-10 % spec (the profile decides) |
| 3 | Sampled DSpark (`SampledSpec::Match` for `--spec dflash`), Metal then HIP | distribution-exact; opt-in, default = user | law test (chi-square) at t 0.6 / 1.0; greedy md5 unchanged; paired sampled A/B | sampled traffic 1.0x -> +15-45 % |
| 4 | Per-row 8-bit dense study on the cpu reference: e4m3 per-row vs int8 per-row (MSE clip) vs today's per-pack fp8 | lossy; opt-in only | G2 (4e-3 / 6e-3) and G3 on cpu, no GPU windows | picks the format before any kernel |
| 5 | Prompt lookup (`propose_copy`) in front of DSpark | bitwise | md5 = `--spec off`; no prose loss | large on copy-heavy spans, 0 elsewhere |

Later or conditional: a calibrated mixed NVFP4 + fp8 checkpoint (after 4), a DSpark fold cache, and a DSpark fine-tune (after 1-4,
Sparks ask).

**Don't do (3 of 3 unless noted):**
- A drafter from scratch.
- Tree or multi-candidate verify: the expert union grows and the conv / SWA rings are linear.
- RTN mxfp4 dense: failed G3.
- Re-quantizing the NVFP4 experts.
- fp8 KV at <= 4k: < 2 %.
- Quantizing the draft layers: measured slower.
- Screening multi-row verify heads: measured flat.
- A bespoke on-disk container.
