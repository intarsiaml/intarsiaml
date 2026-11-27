# Inkling-2-Mini: which 8-bit dense format for the opt-in fast mode (lane ink-q8, 2026-11-26)

Status: accuracy study on the cpu reference only (no GPU, no kernel, no default change). Branch `ink/q8-study`.

Status: done. **Arm C (int8, one f32 scale per output row, MSE-optimal clip) passes G2 and G3 on every set** (G2 short 1.68e-3 /
long 3.81e-3, G3 short 6.57e-3 / long 1.243e-2 / 4k 1.035e-2, 0 / 1 / 1 confident misses) and halves the bytes like fp8 (3.506 ->
1.756 GB per token for in_proj + wo + unembed). A (today's fp8) and B (e4m3 per row) fail G2 short (4.39e-3, 4.12e-3) and pass the
rest. A per-row e4m3 scale does not help; keeping the small projections bf16 (D) does not help. The numerics floor of these gates is
large (see the noise controls); the ranking rests on the error size, not on one-row differences. Default unchanged, no GPU kernel.

## Briefs received

1. 2026-11-26 (coordinator), lane ink-q8 (accuracy study, cpu only, no GPU): pick the opt-in 8-bit dense format for Inkling-2-Mini by
   accuracy on the cpu reference before anyone writes a GPU kernel. Three consultants agree an 8-bit dense + unembed tier with
   per-row scales is the right opt-in fast mode (`docs/plans/2026-11-26-inkling2-drafter-quant.md` on `docs/drafter-quant-plan`;
   modelled +25 % plain decode on the R9700, +10-16 % on Metal); they disagree on the element format. Arms, each applied to the
   in-projection packs, `wo` and the unembed:
   A. today's fp8 (per-pack e4m3 scale), reproduce `2026-11-21-inkling2-fp8.md`'s row first;
   B. e4m3 with one f32 scale per output row;
   C. int8 with one f32 scale per output row, scale from an MSE-optimal clip search per row (symmetric, round to nearest);
   D. (optional, if B or C passes G2) the same with `wr_du`, the sink projection and the sconv weight projection kept bf16.
   B and C are new opt-in values of the existing env hooks (host requant at load, cpu reference dequantises); no GPU kernels, no
   default change; the w-dtype dispatch stays one match (lane ink-hip-p12 may add a P12 arm on HIP in the same match: whoever
   lands second merges onto the other). Report per arm: GB / token, G2 short / long (KL21, confident-row misses, top-1), G3 short /
   long / 4k, KL vs our own bf16, decode-mode KL if the fp8 lane measured it. Verdict: which arm passes G2 (4e-3 / 6e-3, misses
   <= 1 / 2) and G3, and the bytes it saves. Fleet rules: all runs cpu on endor (`fleetq run --host endor --load cpu --expect MIN`),
   no GPU, no Sparks, no single call or wait over 4 minutes, pushes through fleetq. Deliverables: this doc, branch pushed, a PR
   only if an arm passes G2 and G3 (not merged), final message with the arms table, the verdict and what a GPU kernel for the
   winner needs on HIP and Metal.

## What was built

| Piece | Where | Notes |
|---|---|---|
| `fp8_row` / `int8_row` values | `crates/intarsia-core/src/quant/head.rs` (`HeadQuant::{Fp8Row, Int8Row}`, `quantize_host`) | The two env hooks (`INTARSIA_INK_ATTN_QUANT`, `INTARSIA_LMHEAD_QUANT`, and the recipe keys) take them. `fp8_row`: `scale = amax_row / 448` (f32), `e4m3(w / scale)`. `int8_row`: per row, 28 clip candidates (clip / amax from 1.0 down to 0.325) then 17 fine ones around the best, each scored by the row's squared error with round-to-nearest-even and levels -127..127; `scale = clip / 127`. |
| Emulation | same | The host requant produces the values the cpu reference would compute (`q * scale_row` is exact in f32 for both formats), stored as dense f32 weights: the cpu `Matmul` / `Logits` / `InkInProj` float path reads them. The activations stay bf16 (W8A16) and the sum is f32, so the only difference from `scale_row * sum(x * q)` is the accumulation order (about 1e-6 relative, far below the quantisation effect). Any backend but the cpu refuses the values (`emulation_guard`): no GPU kernel exists. |
| Where it applies | `crates/intarsia-core/src/model/inkling2.rs` (`load`), `model.rs` (`load_head`) | The stacked in-projection `[q \| r \| gate \| sink \| k (\| k taps) \| v (\| v taps)]` row by row (so a per-row format needs no pack boundaries), `wo`, and the 200058 real unembed rows. Not the embedding gather, the routers, the controllers or the experts. |
| `INTARSIA_INK_ATTN_QUANT_KEEP` | `inkling2.rs` | Arm D: `wr`, `sink`, `sconv` (and `gate`, `wq`) kept at the checkpoint's bf16 values. |
| Controls | `head.rs` | `int8a_row` (int8, plain `amax / 127`), `id_row` (no quantisation, exact f32 copy of the checkpoint), `noise_row` (every weight times `1 + eps * u`, `u` uniform in [-1, 1) hashed from seed and position; `INTARSIA_Q8_NOISE_EPS`, `INTARSIA_Q8_NOISE_SEED`). |
| Nominal bytes | load log | `inkling2: row-quant nominal bytes per token (48 visits): ...`, `unembed row-quant nominal bytes per token: ...` count 8-bit weights plus one f32 scale per row (the process holds f32). |
| Tests | `head.rs` | parse / cpu-only, per-row error against the per-tensor fp8, int8 levels, zero rows, row shards, clip search; the existing 5 head tests still pass (8 / 8). |

## Method

Same harness as the fp8 lane (`docs/results/2026-11-21-inkling2-fp8.md`): `intarsia accuracy --backend cpu --mode prefill
--write-self-golden`, the self-golden scored by `confident.py` (the skel lane's script, copied unchanged) against the vLLM W4A4
golden (G3: `golden/inkling2-mini-vllm-{short,long,4k}-top20`), the W4A16 oracle golden (G2: `golden/oracle/inkling2-mini-oracle-w4a16-{short,long}-top20`)
and our own bf16 self-golden (the quantisation effect alone). KL21 is the 21-way KL mean over all scored rows. Misses are
confident-row top-1 misses (golden `p1 - p2 >= 0.5`; allowance 1 + rows / 1000). Binaries: `.scratch/bin/intarsia` (sha256 `081bad67...`, built from the working tree before the controls existed: the arms A, B, C
and the bf16 / long / 4k runs), `intarsia2` (`02649cc4...`, adds `int8a_row`, `id_row`, the stats; short controls) and `intarsia3`
(`b880d0e5...`, adds `noise_row`; noise runs). The arm code did not change between them: `Cs` (intarsia2) is tensor-identical to
`C` (intarsia). Scripts: `.scratch/acc.sh` (arms), `.scratch/py/summary.py` (rows below).

Harness proof: our bf16 self-golden is tensor-identical to the fp8 lane's `bf16-short` (`sgcmp.py`: 24 / 24 tensors equal), arm A
reproduces its row exactly (short: G3 8.956e-3 / 0 of 367 / 97.72 %, G2 4.387e-3, vs own bf16 4.859e-3; long: 1.282e-2, 5.300e-3),
and the identity control `id_row` (the same emulation path with no quantisation) is tensor-identical to bf16. The int8 arm `C`
re-run with statistics on (`Cs`) is tensor-identical to `C`: the study is deterministic.

## Results

Per-token bytes are `in_proj + wo + unembed + scales` (the three tensors the mode changes; `fp8.md`'s whole-model column is this
plus the rest of the 5.49 GB bf16 model, which is unchanged: whole model = 5.49 - 3.506 + column). Nominal 8-bit bytes (the process
holds dequantised f32): the load log of each run.

| Arm | in_proj + wo + unembed GB / token | whole model GB / token | saved vs bf16 (attention + unembed) |
|---|---:|---:|---:|
| bf16 (default) | 2.584 + 0.922 = 3.506 | 5.49 | - |
| A fp8, per-pack scale (3 per layer) | 1.292 + 0.461 = 1.753 | 3.74 | 1.753 GB (50.0 %) |
| B fp8, per-row scale | 1.294 + 0.462 = 1.756 | 3.74 | 1.750 GB (49.9 %) |
| C int8, per-row scale | 1.294 + 0.462 = 1.756 | 3.74 | 1.750 GB (49.9 %) |
| D (B or C, `wr` + `sink` + `sconv` bf16) | 1.340 + 0.462 = 1.802 | 3.79 | 1.704 GB (48.6 %) |

B and C cost 2.9 MB of f32 scales over the whole token (10 968 rows per visit x 48 + 200 058 unembed rows, 4 bytes each) over A's
12 bytes per layer: 0.17 % of the bytes. Per-row scales are free in bandwidth. Keeping the three small projections bf16 (D) costs
0.046 GB per token (2.6 % of the saving).

### Accuracy, cpu reference, prefill mode

KL21 mean (G3 against the vLLM golden: bounds 1.05e-2 short / 1.65e-2 long / 1.4e-2 4k; G2 against the W4A16 oracle: bounds 4e-3 short /
6e-3 long); `misses` are confident-row top-1 misses over the confident rows (G3 allowance 1 / 2 / 9, G2 1 / 2); `own` is KL21
against our own bf16 self-golden (the quantisation effect alone). Top-1 is reported, not gated.

| Arm | G3 short | G3 long | G3 4k | G2 short | G2 long | own bf16 short / long / 4k |
|---|---|---|---|---|---|---|
| bf16 (default) | 9.062e-3, 0 / 367, 97.89 % | 1.410e-2, 0 / 1211, 97.51 % | 9.810e-3, 2 / 8229, 95.73 % | 2.994e-3, 0 / 363, 99.30 % | 3.879e-3, 1 / 1206, 99.04 % | - |
| A fp8 per pack | 8.956e-3, 0 / 367, 97.72 % | 1.282e-2, 0 / 1211, 97.51 % | 1.183e-2, 2 / 8229, 95.54 % | **4.387e-3**, 0 / 363, 99.12 % | 5.300e-3, 1 / 1206, 98.92 % | 4.859e-3 / 5.622e-3 / 7.584e-3 |
| B e4m3 per row | 7.511e-3, 0 / 367, 96.84 % | 1.216e-2, 1 / 1211, 97.32 % | 1.145e-2, 1 / 8229, 95.34 % | **4.117e-3**, 0 / 363, 97.72 % | 5.073e-3, 1 / 1206, 98.34 % | 5.196e-3 / 5.678e-3 / 7.410e-3 |
| C int8 per row, MSE clip | 6.574e-3, 0 / 367, 98.07 % | 1.243e-2, 0 / 1211, 97.64 % | 1.035e-2, 1 / 8229, 95.69 % | 1.680e-3, 0 / 363, 99.12 % | 3.808e-3, 1 / 1206, 98.92 % | 4.068e-3 / 4.644e-3 / 5.313e-3 |
| D1 = B + `wr`,`sink`,`sconv` bf16 | 8.876e-3, 0 / 367, 97.54 % | - | - | **4.401e-3**, 0 / 363, 98.24 % | - | 5.556e-3 / - / - |
| D2 = C + `wr`,`sink`,`sconv` bf16 | 6.564e-3, 0 / 367, 97.89 % | - | - | 2.593e-3, 0 / 363, 98.95 % | - | 4.227e-3 / - / - |
| control: int8, plain `amax / 127` scale | 6.373e-3, 0 / 367, 98.07 % | - | - | 1.754e-3, 0 / 363, 99.47 % | - | 3.813e-3 / - / - |

(Bold: over the G2 short bound.) Decode-mode KL: the fp8 lane measured it on Metal only (`accuracy --backend metal --mode both`,
fleetq #10870; fp8 / fp8 decode-mode KL21 1.396e-2 short / 1.569e-2 long against 1.057e-2 / 1.219e-2 for bf16). B and C have no
Metal kernel and the cpu `--mode decode` is not the same measurement, so it is **n/a** here; it is the first thing to
measure when a kernel exists.

Per tensor, the weights' own relative rms error (`INTARSIA_INK_ATTN_QUANT_STATS=1`): e4m3 2.64 % for every segment (a floating
format has a constant relative error, per-row or per-pack scale alike); int8 with the MSE clip about 0.9 % for q, k, v, gate and
`wo`, 1.0-2.4 % for `wr_du` and the sink projection, 4.5 % for the K / V sconv weight projections. 0 weights moved by more than
5 % of their row's amax in either format.

### The numerics floor, and what it says about the table

The first surprise: C is *closer* to the oracle and to vLLM than our bf16 is (G2 short 1.68e-3 against 2.99e-3, G3 short 6.57e-3
against 9.06e-3), and the quantisation effect alone is 4e-3 KL, about the same as e4m3's although its weights are 3x more
accurate. The bf16 oracle gap of the short set is dominated by one row (KL 0.61 of a total of 1.70 summed over 569 rows). So the
gates themselves were measured with a noise control: `noise_row` multiplies every weight (in_proj, `wo`, unembed) by
`1 + eps * u`, `u` uniform in [-1, 1), several seeds, at three sizes: rms 0.056 % (eps 2^-10), rms 0.9 % (eps 0.0156, the size of
the int8 error) and rms 2.5 % (eps 0.0433, the size of the e4m3 error). `id_row` (no perturbation at all) is tensor-identical to
bf16, so the emulation path adds nothing.

Short set (min / mean / max over seeds):

| Perturbation | seeds | G3 short | G2 short | own bf16 |
|---|---:|---|---|---|
| random noise rms 0.056 % | 5 | 5.73e-3 / 6.75e-3 / 8.31e-3 | 1.55e-3 / 2.03e-3 / 3.04e-3 | 1.83e-3 / 3.60e-3 / 4.51e-3 |
| random noise rms 0.9 % | 3 | 6.63e-3 / 7.88e-3 / 9.12e-3 | 1.40e-3 / 2.35e-3 / 3.54e-3 | 3.65e-3 / 3.90e-3 / 4.38e-3 |
| random noise rms 2.5 % | 3 | 8.08e-3 / 8.54e-3 / 9.20e-3 | 4.06e-3 / 4.60e-3 / 5.64e-3 | 5.10e-3 / 5.70e-3 / 6.47e-3 |
| int8 per row (C, C2, D2) | - | 6.37e-3 / 6.50e-3 / 6.57e-3 | 1.68e-3 / 2.01e-3 / 2.59e-3 | 3.81e-3 / 4.04e-3 / 4.23e-3 |
| e4m3 per row / per pack (A, B, D1) | - | 7.51e-3 / 8.45e-3 / 8.96e-3 | 4.12e-3 / 4.30e-3 / 4.40e-3 | 4.86e-3 / 5.20e-3 / 5.56e-3 |

Long set (one seed per size): noise rms 0.056 % G2 3.60e-3 and 2.73e-3, G3 1.27e-2 and 1.24e-2, own 2.96e-3 and 4.28e-3; rms 0.9 %
G2 3.97e-3, G3 1.31e-2, own 3.67e-3; rms 2.5 % G2 5.78e-3, G3 1.41e-2, own 5.84e-3 (C: 3.81e-3 / 1.24e-2 / 4.64e-3; A, B: 5.30e-3,
5.07e-3 / 1.28e-2, 1.22e-2 / 5.62e-3, 5.68e-3).

Reading:
- **Any perturbation of the weights, however small, moves the own-bf16 KL to 2-4e-3** (rms 0.056 %: up to 4.5e-3, with one
  confident miss against our bf16 in two of five seeds): bf16 activations amplify any change through 48 visits. Our bf16 baseline is itself an
  unlucky draw: its G3 short 9.06e-3 is above the maximum of the five tiny-noise seeds (8.31e-3). G2 / G3 differences below
  about 2e-3 (short) are noise; "better than bf16" for a quantised arm means nothing.
- **Above that floor the error follows the weight error**: random noise of int8's size lands where int8 lands (G2 short 2.4e-3
  mean against 2.0e-3, own 3.9e-3 against 4.0e-3) and noise of e4m3's size lands where every e4m3 arm lands (G2 short 4.6e-3
  against 4.3e-3, own 5.7e-3 against 5.2e-3, long 5.8e-3 against 5.1-5.3e-3). The per-row scale does not matter for e4m3 (A, B, D1 are
  the same arm) and keeping `wr` / `sink` / `sconv` bf16 does not matter for either format (D1 / D2 inside the seed spread of B / C):
  the error is the 3-bit mantissa, spread evenly over all tensors, not the few sensitive ones.
- **G2 short is a coin toss at the bound for e4m3, with margin for int8**: every e4m3 variant and the 2.5 % noise land at 4.1-5.6e-3 on
  the 4e-3 bound (none passes; the mean is 4.3e-3); int8 and noise below 1 % land at 1.4-3.5e-3 and pass in all twelve runs.

## Verdict

**Pick int8 with one f32 scale per output row (arm C) as the 8-bit dense + unembed format.**

| Gate | bound | A fp8 per pack | B e4m3 per row | C int8 per row |
|---|---|---|---|---|
| G2 short (KL21, misses) | 4e-3, 1 | **4.387e-3** fail, 0 | **4.117e-3** fail, 0 | 1.680e-3 pass, 0 |
| G2 long | 6e-3, 2 | 5.300e-3 pass, 1 | 5.073e-3 pass, 1 | 3.808e-3 pass, 1 |
| G3 short | 1.05e-2, 1 | 8.956e-3 pass, 0 | 7.511e-3 pass, 0 | 6.574e-3 pass, 0 |
| G3 long | 1.65e-2, 2 | 1.282e-2 pass, 0 | 1.216e-2 pass, 1 | 1.243e-2 pass, 0 |
| G3 4k | 1.4e-2, 9 | 1.183e-2 pass, 2 | 1.145e-2 pass, 1 | 1.035e-2 pass, 1 |
| KL vs own bf16 short / long / 4k | - | 4.9e-3 / 5.6e-3 / 7.6e-3 | 5.2e-3 / 5.7e-3 / 7.4e-3 | 4.1e-3 / 4.6e-3 / 5.3e-3 |
| bytes per token, in_proj + wo + unembed | bf16 3.506 GB | 1.753 GB | 1.756 GB | 1.756 GB |

- **C is the only arm that passes G2 short**, and the only one that does so with margin (1.68e-3 against 4e-3; D2 = C with the
  three small projections bf16: 2.59e-3). Against the noise controls this is a real, repeatable difference, not luck: all twelve
  runs whose weight error is at most int8's (int8 variants, random noise up to rms 0.9 %) pass G2 short with 1.4-3.8e-3; all six
  runs at e4m3's error (A, B, D1, random noise rms 2.5 %) land at 4.06-5.64e-3, at or just over the bound (two of them within 2 % of it), and none passes: the separation of the two classes is what repeats, not any single value. The 4k set tells the same story:
  KL against our own bf16 is 5.3e-3 for C against 7.4-7.6e-3 for the e4m3 arms.
- **Both e4m3 arms fail G2 short and pass everything else**, as the fp8 lane found for A. B's one f32 scale per row buys nothing
  (4.12e-3 against A's 4.39e-3, own-bf16 KL 5.2e-3 against 4.9e-3): e4m3 has a constant *relative* error of 2.64 % whatever the scale
  (a scale only moves the range), so only a format with more mantissa bits can help. int8's step is uniform inside the row, which suits
  weights (rms 0.9 % for the big tensors).
- **D (keep `wr_du`, the sink projection and the sconv weight projections bf16) is not needed**: D1 4.40e-3 against B 4.12e-3 (fail
  either way), D2 2.59e-3 against C 1.68e-3, own-bf16 KL 5.6e-3 / 4.2e-3 against 5.2e-3 / 4.1e-3: inside the seed spread. It costs
  0.046 GB per token (2.6 % of the saving). The three tensors carry larger int8 weight error (1-2.4 %, sconv projections 4.5 %) and
  it makes no measurable difference.
- **The MSE clip search is not what makes int8 pass**: int8 with the plain `amax / 127` scale (`int8a_row`, short set only) gives G2 1.75e-3, G3 6.37e-3, own
  3.81e-3, the same as C. The clip search only costs load time (the C runs took as long as the e4m3 runs). If a kernel lane wants the
  simplest host quantiser, `amax / 127` is enough on this evidence; long and 4k were not run for it.
- **Bytes**: 3.506 -> 1.756 GB per token for in_proj + wo + unembed (-1.750 GB, -49.9 %), 5.49 -> 3.74 GB for the whole model; scales cost 2.9 MB.
  The same as fp8; per-row scales are free in bandwidth. The modelled +25 % (R9700) / +10-16 % (Metal) plain decode is the
  consultants' number for 8-bit dense + unembed and applies to both formats; this study does not measure speed.
- Not measured: decode-mode KL (the fp8 lane's was Metal only), speed (no kernel), any model but Inkling-2-Mini, sampled
  (non-greedy) quality. The cpu confident-row misses that quantisation adds are one per set at most.

**What a GPU kernel for C needs.** Both targets are the existing W8A16 GEMV structure with the per-pack scalar replaced by a per-row
scale in the epilogue, and the int8 element should be no more expensive to decode than e4m3 (unmeasured; gfx12 has native fp8 converts). HIP (R9700, gfx1201): the e4m3 GEMV's load structure (16-byte
global loads per lane, 1..16 rows, rows-exact tree) with the e4m3 decode replaced by a sign-extend and an int-to-float convert (the
activations stay bf16: packed i8 dot products would need quantised activations, a W8A8 mode this study did not measure), the row scale multiplied once after the f32 sum and before the bf16 rounding, which is exactly the cpu emulation's
`rA(scale_row * sum(x * q))` order. Metal: `ink_in_proj_f8_gemv`'s 8-byte chunks read as `char8` (Apple GPUs convert i8 to half /
float natively), same lane mapping and reduction tree; the prefill chunks (more than 16 rows) need a
dequantising GEMM (NAX path or an int8 variant of the simdgroup matmul) or the three-pack fallback must also hold int8. Contract
changes shared by both: `InkInProj`'s third input becomes `scales [Nq+Nk+Nv]` f32 (a length other than 3 selects the per-row
form, keeping the `[3]` e4m3 form and the validator one match over the w dtype: {bf16, e4m3 + scales, p12 + aux, i8 + row scales}),
a per-row-scale `MatmulI8` for `wo` and the unembed (`MatmulFp8` takes `scales [2]`), new admission cases at the real shapes
(`[6552,1056,1056] x 2304`, `wo` `[2304, 3072]`, unembed `[200058, 2304]`) at rows 1 / 4 / 16 / 2048, bitwise rows-exact, an f64 cpu
reference that is the existing emulation plus the scale after the dot, and a rework of the exact unembed screen (`HeadScreen`
bound: its row bound assumes e4m3's relative error; for int8 it is `scale_row / 2` per element, tighter, but it is a new derivation
and a new gather kernel set). The loader keeps one host quantiser (`head::quantize_host`) and stores i8 + a f32 scale buffer
instead of the dequantised f32 used here; the P12 arm of ink-hip-p12 sits in the same w-dtype match, so whoever lands second
merges onto the other. Decode-mode KL should be measured on the first kernel that runs.

## fleetq jobs

All on endor, `--load cpu`, `--mem 30` (4k runs 30 as well after #12149's update from 40).

| fleetq | what |
|---|---|
| #12107 | build `intarsia-cli` + head unit tests |
| #12109 | short: bf16, A, B, C |
| #12115 | long: bf16, A, B, C |
| #12116, #12126 | rebuilds (controls, noise control); head unit tests 8 / 8 |
| #12119 | short: id_row, int8a_row, C with stats, D1, D2 |
| #12127 | short noise: seed 1, seed 2 (rms 0.056 %), eps 2^-7 |
| #12143 | short noise ensembles: 3 levels x 3 seeds |
| #12150 | long noise controls (4 runs) |
| #12149 | 4k: bf16, C, A, B (3658 / 2310 / 3183 / 2391 s) |
| #12151 | `docs_env_vars` 2 / 2, `recipes_doc` 3 / 3, `intarsia-core` inkling2 lib tests 4 / 4 |
| #12202 | push `ink/q8-study` |
| #12263 | after merging origin/main (80719483a): build, head tests 8 / 8, `docs_env_vars` 2 / 2, `recipes_doc` 3 / 3, C short re-run tensor-identical to the pre-merge C |

Raw self-goldens, logs and the scripts (`.scratch/acc.sh`, `.scratch/py/{confident,sgcmp,perprompt,summary}.py`, `.scratch/table.sh`) are
in `.worktrees/ink-q8/.scratch` (git-ignored).

