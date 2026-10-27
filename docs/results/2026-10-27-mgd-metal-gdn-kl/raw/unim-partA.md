Origin project: intarsiaml, lane MGD, 2026-10-27, Metal (M5 Max), Qwen3.6-35B-A3B NVFP4, main 2859c1a89.

Question: why is Metal (and HIP) decode KL vs the vLLM goldens ~1.6-2.0x CUDA on 35b-short?

Finding: no decode-path numerical defect. Evidence (deterministic runs, current main):
1. Full-window teacher-forced decode (every position) vs the vLLM short golden: decode KL21 5.42e-3 over 1431 positions, prefill 6.45e-3 over 1441. Decode is not worse than prefill.
2. Against the independent f64 CPU oracle, oracle set (10 short prompts + long_docs_4k, 5494 positions): decode 9.451e-3, prefill 9.456e-3 at the same positions; paired diff -4.7e-6, block-bootstrap 95% CI [-7.0e-4, +6.7e-4]. Any decode-only op error (GDN state, softmax, nvfp4 GEMV rounding, MoE sum order, rmsnorm) is bounded under ~7% of the oracle distance. Decode and prefill run the same gdn_delta_rule kernel (T=1 vs T=2048), same f32 state.
3. The 16-position gate window (last 16 tokens of 10 prompts = 160 positions) is a heavy-tail metric: top 10 positions carry 72% of the mean. Same weights, same decode kernels: with INTARSIA_METAL_FP8_NAX=0 INTARSIA_METAL_MOE_NAX=0 (pre-NXP prefill numerics) window decode reads 1.164e-2; with NAX on 8.2e-3 (scored 6.95e-3); full-window decode is identical in both. Window prefill vs oracle 6.0e-3 (NAX) / 1.02e-2 (non-NAX), window decode 8.8e-3: the window metric moves 1.7x with prefill accumulation order alone, the size of the "1.6-2x CUDA" gap.
4. baseline-gate vs the CUDA baseline today: only chat_multiturn fails per-prompt (1 of 10, was 3 of 10), one near-tie position (chat_multiturn pos 55, KL 0.35). Aggregate window decode 8.19e-3 vs CUDA 7.34e-3 (1.12x).
Limit: no CUDA decode-vs-oracle data exists (oracle self-goldens have prefill rows only).
Raw: docs/results/2026-10-27-mgd-metal-gdn-kl/ (branch lane-mgd/metal-gdn-kl).
