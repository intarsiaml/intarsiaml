# Consult: a custom drafter and a custom IntarsiaML quant for Inkling-2-Mini (R9700 and Mac)

You are one of two independent consultants (the other is a different model family). Answer on your own evidence; the coordinator
compares the two answers and checks every claim against main before anything is built. Read-only: do not propose to run anything
yourself.

## The user's question (verbatim)

"what about a custom drafter and custom intarsiaml quant."

## Scope and rules you must respect

- Hardware in scope: one AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB, ~627 GB/s; tatooine, shared with a production server
  that we take down in <= 5 min windows) and endor (MacBook Pro, Apple M5 Max, 128 GB unified, ~560 GB/s measured floor, 140 W
  adapter: A/Bs need 45 s idle gaps). CUDA / DGX Spark work is out of scope for this push; if your plan needs the Sparks (2x GB10,
  121 GB each) for training, say so explicitly as a separate ask.
- Model: Inkling-2-Mini (MoE; experts NVFP4 W4A16 from the checkpoint, plus a few MXFP8 experts; dense attention projections,
  controllers and the 200058 x 2304 unembed in bf16). The checkpoint ships its own drafter, DSpark (`mtp.safetensors`), a
  DFlash-family block drafter: 8-row block (anchor + 7 masks), taps from target visits 5, 23, 41, 48, a rank-256 Markov head.
- Policy: bitwise-identical changes may go default-on on the coordinator's judgement after gates. Anything that changes the
  greedy output (lossy quant, different numerics) is opt-in unless the user flips it. Accuracy gates are in docs/accuracy.md
  (G2: KL21 and confident-row misses against the oracle; G3: against vLLM goldens). Speculative decode must keep greedy output
  identical to `--spec off` (a drafter only changes acceptance, never the output).

## Current measured state (main 3c6ff64f0, 2026-11-26)

R9700, one card, greedy, tok/s:

| | plain decode (`--spec off`) | DSpark default (k3, cut 0.5) | prefill 4k / 16k |
|---|---|---|---|
| short | 77.4 | 103.5 | |
| 1k | 75.4 | 89.9 | |
| 4k | 74.1 | 114.7 | 2988 / 2842 (PR #528 pending: 3040 / 3178, bitwise) |
| 16k | 69.7 | 91.6 | |

- Decode byte floor on the R9700: 5.49 GB of weights per token (attention projections 2.58, experts 1.84, controllers + router
  0.15, unembed 0.92) + KV reads (+0.30 GB at 4k) = 8.76 ms = 114 tok/s short (docs/results/2026-11-25-inkling2-hip.md section 3).
  Plain decode is now ~68 % of that floor.
- HIP verify cost (docs/results/2026-11-26-inkling2-hip-spec.md, "Verify cost"): decode 13.30 ms; verify 2 rows 16.48 (1.24x),
  3 rows 19.00 (1.43x), 4 rows 20.80 (1.56x). Verify rows 2..16 run the row-exact (decode-numerics) kernels so spec stays
  bitwise. Draft block ~2.1 ms, ~6 % of a cycle.
- DSpark acceptance on HIP: greedy set cut 57.5 % (2.36 tokens / pass), gate 70.9 % (2.54); prose cut 59.5 % (2.27).
- Metal (endor): plain decode ~79 tok/s at 143 tokens; P12 (lossless 12-bit packing of in_proj / wo / unembed) is default-on on
  Metal (+8.5-9.7 % decode) and NOT built on HIP; fp8 attention + fp8 unembed is an opt-in Metal mode (+21 %, KL +5e-3);
  mxfp4 dense was rejected (fails G3). See docs/results/2026-11-21-inkling2-fp8.md and -pack12.md.

## What to answer

1. **Custom drafter.** Is training our own drafter (or fine-tuning DSpark) worth it on this hardware, versus cheaper levers on
   the spec path (e.g. verify cost per row, tree / multi-candidate verify, draft length policy, prompt-lookup / n-gram drafting)?
   If yes: which architecture, trained on what data (self-distilled from our engine's greedy outputs? chat traces?), where
   (endor MLX/PyTorch-MPS? R9700 ROCm? needs the Sparks?), how long, and the expected tokens / pass and tok/s with your arithmetic
   from the verify-cost table above. Mark modelled vs measured numbers.
2. **Custom IntarsiaML quant.** Which format(s), for which tensors (attention projections, unembed, experts, KV cache, drafter),
   lossless or lossy, calibrated how (none / GPTQ / AWQ / our own KL-driven search against G2/G3)? Bytes saved per token against
   the 5.49 GB floor and expected decode tok/s on the R9700 and on Metal. Which part could be default-eligible (lossless) and
   which must stay opt-in. Is "an IntarsiaML checkpoint format" (pre-quantized, pre-packed weights on disk) worth it beyond speed?
3. **Interaction.** How a quantized target changes drafter acceptance and verify cost; whether the drafter itself should be
   quantized; ordering.
4. **Plan.** A ranked top 5 with effort (agent-hours + GPU windows), the gate each must pass, and the expected gain. Then a
   "don't do" list with one-line reasons.
5. **Evidence.** Every claim about current code or past results must cite a file (and line where you can) from the manifest.
   Before proposing anything, check the results docs for whether it was already tried or killed, and say so.

Keep it under ~2000 words. Lead with a one-paragraph verdict on each of the two questions.
