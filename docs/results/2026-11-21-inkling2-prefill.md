# Inkling-2-Mini prefill on Metal (lane ink-prefill, endor M5 Max, 2026-11-21)

Branch `ink/prefill` from 1dacac3a8 (ink/metal). Not pushed or merged. Every GPU run had a fleetq slot (timing `--quiet gpu,cpu`).

## Briefs received

1. 2026-11-21 (coordinator, via SendMessage): **both go-aheads approved; keep the order MoE experts first, the looped blend
   second, rel_mode third.**
   - **rel_mode conditions:** (1) `ink.rel_mode` defaults to 0 and mode 0 stays exactly today's contract; (2) a CPU reference
     test asserts the mode-1 pair equals the mode-0 pair bitwise; (3) Metal admission sweep cases for mode 1 at prefill shapes
     (T 17, 512, 2048; SWA and full visits; tau past 8192); (4) the graph picks mode 1 only when the backend admits both calls in
     mode 1, the CUDA lanes need nothing; (5) rerun G3 short/long/4k with mode 1 active on Metal and update the op-list entry
     in the spec doc on this branch.
   - **Blend:** this lane owns the prefill path (T > 16), in its own file plus the one-line hook in `launch_mix`, after
     ink/m-mix lands in ink/metal. m-mix keeps decode (T <= 16). No edits to m-mix's kernel files beyond the hook.
   - Report prefill tok/s at 2k / 8k / 12k next to oMLX's ~1300 when available.
2. 2026-11-21 (coordinator, via SendMessage): **ink/metal is integrated and runs end to end** (tip a88db4620: m-attn, m-moe,
   m-mix up to d99d2c0fc, ink/skel; G1-G3 pass on Metal). Its baseline: 5k-token prefill 1219 tok/s at chunk 256, 2221 tok/s
   at chunk 2048 (~920 ms per 2048 chunk vs this lane's ~700 ms kernel sum).
   - Merge ink/metal a88db4620 into ink/prefill now (unblocks the blend hook).
   - Measure end-to-end prefill at 2k / 8k / 12k, chunk 2048, on the merged tip before and after each kernel switched on (bench
     command in the ink-metal results doc).
   - Report the 920 vs 700 ms split: per-chunk host gap, router / routing, whatever else is missing. Prefill host gaps belong to
     this lane (ink-hostgap is decode only).
   - One line on 7.3 vs 11.9 ms per MoE visit (m-moe's setup: weights cycled from DRAM, union = 128 experts); no more time on it.
3. 2026-11-21 (coordinator, via SendMessage): endor's quiet-GPU line is 14+ jobs (~9 h). Builds, admission and tests go as
   default-load jobs with short honest `--expect` (they backfill); only sweeps and e2e tok/s take `--quiet gpu,cpu`. Merge
   ink/metal at a0751f302 (the rows_exact pin-list test fix).
4. 2026-11-21 (coordinator, via SendMessage): ink-d1row found inkmoe_bench can draw every call's experts from one 128-expert
   pool (~570 MB), so ~16% of expert reads hit the system cache, while in the stream each layer has its own pool (hit rate ~0);
   at 1 row that explains the bench-vs-stream gap fully. Check whether this lane's sweep and the MoE lane's numbers differ in
   pool / rotation setup; before choosing a default, rotate 24 per-layer pools in the bench or confirm with the in-stream
   per-chunk time.
   - Answer: this lane's 2048-row sweep cycles 5 NVFP4 / 3 MXFP8 pools of 573 / 1051 MB (`w[c % layers]`), so a pool comes
     back only after 2.3-3 GB of other weights: no cross-call cache reuse. At 2048 rows the experts are compute / staging-bound
     (a pool is ~0.9 ms of DRAM time of a 7.3 ms call). In-stream (8k prompt, kernel trace) the NVFP4 experts take 8.3 ms per
     visit (gu 5.25, dn 2.79, gather 0.25) vs the bench's 7.3: the stream's routing is less balanced than random. Any MoE
     default is decided on the in-stream per-chunk time (#10884 profiles m128 in-stream), not on the bench.

5. 2026-11-21 (coordinator, via SendMessage): next steps in order.
   1. Merge gate for ink/prefill b178a17f9: localeval speed-plain at 1k / 4k / 16k, base ink/metal vs this branch, own fleetq
      slot via the local-model-bench skill. Pass: prefill / TTFT better at 4k and 16k, decode tok/s within 1% at every length.
      On pass, send the hash to the coordinator (the integration lane merges).
   2. Router microbench crash: find the cause before the merge; harness -> fix or drop the bench; kernel at some shape -> fix
      and add the shape to admission.
   3. Host gap (57 ms / chunk, ~8%): batch the prefill command buffers (first among the speed items).
   4. Router on the accelerators: 14 ms -> ~1 ms per chunk.
   5. MoE (319 of 646 ms): first the shared-unpacked-tile idea (one unpack feeding two row tiles of the same expert); second, as a
      measured experiment, dequant each active expert's weights once per chunk into a bf16 scratch + accelerator matmul. Report
      µs per chunk vs the current kernel for each. Default-on only if bitwise or G2/G3 pass, and faster by the median of 3
      interleaved reps.
   6. rel bias variant: parked; ink/prefill-rel stays unmerged, graph default off.
   Work only in this worktree; non-Metal tests on spark2 (`--load cpu`); timing on endor `--quiet gpu,cpu`; no call or wait
   over 4 min.

6. 2026-11-21 (coordinator, via SendMessage; rule for all Inkling lanes): this session runs on endor, so no local
   cargo build / check / test or shader compiles outside a fleetq slot (they load endor's CPU and skew quiet timing jobs).
   macOS compile checks on bespin (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a git-archive copy in
   ~/intarsiaml-<lane>, toolchain in ~/.cargo/bin; no GPU timing or real model there); CPU tests on spark2; GPU work as endor
   fleetq jobs. Reading files, git and editing locally are fine. (Before this rule this lane ran a few local `cargo check` and
   `xcrun metal -c` checks on endor; from here on they go to bespin.)

7. 2026-11-22 (coordinator, via SendMessage; bug found by ink-dfuse): without metal4 (the no-metal4 tiny suite) ink_in_proj has
   no kernel at prefill shapes m 31-63 (NoKernel: that range is NAX-only); ink_route correctly falls back to the pair there.
   Fix: a non-NAX ink_in_proj path for medium m, bitwise with what ink/metal uses today on non-metal4 devices for the
   neighbouring m, or route m 31-63 to the existing GEMM fallback. Check: the tiny inkling2 suite with metal4 forced off passes
   on endor, plus rows-exact; default-load slot, no local builds. After the queued jobs; report the hash.

8. 2026-11-22 (coordinator, via SendMessage): approved with conditions.
   1. Recipe: `[tuning.metal] prefill_chunk = 2048` in recipes/inkling2.toml as its own commit on ink/prefill, only if #11109
      passes all of: G2/G3 short+long at 2048 within the limits (G2 4e-3 / 6e-3, G3 1.05e-2 / 1.65e-2); speed-plain serve 2048
      >= 256 at every length (prefill and decode, beyond spread, or within 1 % for decode); serve log shows "prefill chunk 2048".
      Commit message cites the #11109 numbers. Revert = that one commit.
   2. Merge: #10967 counts as the speed-plain gate for ink/prefill 378dc214e (1k / 4k clean wins, 16k positive in both pairs,
      decode within 1 %). Send ink-metal (af43898d63e5b5190) the hash to merge (378dc214e, or the recipe commit on top once
      #11109 passes).
   3. If #11109 fails a G2/G3 limit at 2048: stop and report the numbers; no other chunk sizes without asking.
   4. Results doc note: every localeval / accuracy number for Inkling on Metal before that commit ran at 256-row chunks.
   Next, in order: prefill cb batching (57 ms / chunk), router NAX, MoE unpack sharing; also the non-metal4 ink_in_proj
   m 31-63 fix (brief 7).

9. 2026-11-22 (coordinator, via SendMessage): command-buffer batching closed; cold-start page-in (~0.2 s) parked (serve
   keep-warm covers it, no residency work). While #10982 / #11105 / #11109 wait: (1) from existing traces (no new endor jobs)
   rank steady-state prefill ops at chunk 2048 (chunks 1+, warm) by ms/chunk next to each op's M5 Max NAX compute floor:
   experts, looped blend, attention, in_proj, router, everything over 5 ms/chunk; (2) pick the op with the largest
   ms-above-floor not covered by router NAX / v65, write a one-paragraph plan (expected ms/chunk saved, bitwise vs
   within-gates), build it on a new branch off the ink/prefill tip, flag default off; (3) compile checks on bespin, CPU tests on
   spark2, GPU only in endor fleetq slots, no local builds on endor; (4) report the ranking and the pick; the coordinator
   answers within the hour on disagreement, otherwise keep building.

## Status

- **Default-on (commit 355791b1a):** the loop blend of prefill chunks (rows > 16, not `exact_rows`) on the accelerators:
  h by the InkInProj NAX GEMM, gate + norm by `ink_blend_gf_nax` + `ink_blend_yn` (kernels/metal/inkling2_blend_nax.metal).
  `INTARSIA_INK_BLEND_NAX=0` restores m-mix's GEMV path. G2 / G3 with it on: see Gates.
- **Opt-in, not faster:** MoE prefill forms nax2 / nax3 (`INTARSIA_INK_MOE_NAX2=<form>`); all bitwise the first generation
  at 2048 rows, none faster in the bench or in the stream (tables below), including the 128-row tiles.
- **Side branch `ink/prefill-rel` (4212f2e45), not merged:** `ink.rel_mode` (approved contract variant): cpu reference +
  bitwise test pass; Metal mode 1 measured slower end to end, so the graph default is off. Tests and admission pass; G3 with mode 1 not run (off by default).

## Floors

- NAX bf16 GEMM peak on endor: 62 TFLOP/s (4096 x 4096, M 2048; docs/design/2026-11-15-mac-roadmap.md). Every NAX prefill term
  below is bf16 x bf16 on the accelerators (the NVFP4 / MXFP8 experts decode to bf16 in threadgroup memory first).
- FLOPs per 2048-row visit: experts 261 GFLOP (16384 pairs x 15.9 MFLOP), InkInProj 81.8, wo 29.0, router 1.2, blend h 3.6 per
  stream; at 62 TFLOP/s the experts' floor is 4.2 ms per visit, InkInProj 1.3, wo 0.47.

## Per-chunk kernel time model (2048 rows, 48 visits, ctx 0), before

Measured per call by the lanes' benches on endor (`--quiet gpu,cpu`), this lane's runs: `.scratch/base.log` (fleetq #10700).

| Term | per call (ms) | calls / chunk | ms / chunk | Notes |
|---|---:|---:|---:|---|
| MoE experts NVFP4 | 7.28 | 44 | 320 | 35.8 TFLOP/s; random routing, 5 weight copies cycled |
| MoE experts MXFP8 | 7.62 | 4 | 30 | 34.3 TFLOP/s |
| Blend (looped visits), m-mix d99d2c0fc | S=3 4.24, S=1 1.79, LoopBlend 3.93 | 24 x 2 | ~130 | extra over the plain ConvMix; h runs as 16-row GEMV passes |
| InkInProj NAX | 1.50 | 48 | 72 | 54 TFLOP/s |
| Prologue + attention NAX (SWA / full) | 0.98 + 0.87 / 1.00 + 1.02 | 40 / 8 | 90 | the prologue writes the 100 MB rel tensor |
| wo NAX | 0.54 | 48 | 26 | 53.6 TFLOP/s |
| ConvMix plain (Q=1 / Q=3) | 0.27 / 0.39 | 96 | ~29 | |
| AttnResMix (loop messages) | 0.22 | 24 | 5 | |
| Router + MoeRoute | pending | 48 | | |
| **Sum** | | | **~700** | ink/metal e2e: ~920 ms per chunk (2221 tok/s at 5k, chunk 2048) |

On 7.3 vs 11.9 ms per MoE visit: m-moe's 2048-row number was taken with the same bench (cycled weight copies, union = all 128
experts) on a shared GPU; this lane's runs at `--quiet gpu,cpu` gave 7.3 ms (NVFP4) / 7.6 ms (MXFP8) on the same code, so the
11.9 ms looks like GPU contention rather than a setup difference (not investigated further).

## Measurements (endor, `--quiet gpu,cpu`)

### End-to-end prefill, chunk 2048 (`intarsia generate --max-tokens 1`, two runs each; fleetq #10779)

| Config | 2k (1921 tok) | 8k (8034) | 12k (12190) |
|---|---:|---:|---:|
| ink/metal baseline (gen1 everything) | 1588 / 1558 | 1753 / 1606 | 1777 / 1776 |
| blend NAX (level 2) | | 1989 / 1902 | |
| MoE nax3_m64_n128_k32 | | 1752 / 1727 | |
| MoE m64_k32_v3 | | 1900 / 2011 (noise: the microbench says slower) | |
| rel_mode 0 (wt-rel binary) | | 1910 / 1703 | 1798 / 1831 |
| rel_mode 1 | | 1733 / 1787 | 1646 / 1733 |

Run-to-run spread at 8k is up to 10%; the interleaved 3-round A/B is #10884. oMLX: ~1300 tok/s marginal prefill.

### Where a 2048-row chunk goes (baseline, 8k prompt, kernel trace / cb trace)

cb trace: 4 chunks in 4014 ms, GPU busy 94%, idle 232 ms (~58 ms per chunk host gap). Kernel trace, GPU ms per chunk:

| Op | ms / chunk |
|---|---:|
| MoE experts NVFP4 (gu 231, dn 123, gather 11) | 366 |
| ConvMix (incl. the looped-visit blend) | 174 |
| RelSinkAttn NAX | 78 |
| InkInProj NAX | 72 |
| InkKvPrologue | 55 |
| MoE experts MXFP8 | 35 |
| wo (matmul_bf16_nax) | 26 |
| router (logits_bf16, 0.31 ms at 3.9 TFLOP/s) | 16 |
| LoopBlend | 9 |
| AttnResMix, MoeRoute, rest | 5 |
| **GPU sum** | **837** (+ ~58 host gap = the ~920 ms ink/metal saw at 5k) |

The 700 ms model missed the blend's real size (ConvMix 174 vs 29 + ~100 estimated), the e2e routing being less balanced
than random (gu 5.25 ms vs 4.84 per visit), attention at 8k context, and the router.

### MoE experts NVFP4, 2048 rows, µs per call (gu only = skip 2, dn only = skip 1)

| Form | full | gu only | dn only | Note |
|---|---:|---:|---:|---|
| gen1 (ink/m-moe) | 7287 | 4845 | 2698 | 35.8 TFLOP/s |
| nax2 m64_k64_v1 (A prefetch) | 7381 | 5408 | 2798 | |
| nax2 m64_k32_v3 (double-buffered) | 8502 | 6884 | 4496 | |
| nax2 m32_k64_v3 | 14257 | | | |
| nax2 v8 (no matmul, probe) | 6348 | | | MMA hides under staging |
| nax2 v28 (loads + epilogue only, probe) | 2510 | | | |
| nax3 m64_n64_k64 (A from device) | 12595 | | | device-tensor A is slow |
| nax3 m64_n128_k32 | 14190 | | | |
MXFP8: gen1 7690, m64_k32_v3 7424 (-3.5%), m64_k64_v1 7598, nax3 9450 / 13327.

Reading: dropping the matmul saves only 0.9 ms, so the kernel is bound by weight decode + threadgroup staging (~3.8 ms)
and loads / epilogue (~2.5 ms), not by the accelerators. With 64-row tiles every expert (mean 128 rows) decodes its weights
twice. The 128-row forms (m128_*, decode once per expert) are much slower (NVFP4 m128_k64_v1 17542, m128_k32_v3 12991;
MXFP8 18005 / 13995 µs; in-stream 1600 / 2000 tok/s at 8k vs 2643): 8 A chunks per thread per step and a 64 x 32 sub-tile per
simdgroup cost more than the halved decode saves. The gen1 kernel stays the default.

### Blend on 2048 rows (ink_mix_bench, µs)

| Call | GEMV (m-mix) | level 1 (NAX h) | level 2 (NAX h + gate) |
|---|---:|---:|---:|
| ConvMix blend S=3 | 4277 | 3165 | 1625 |
| ConvMix blend S=1 | 1783 | 1688 | 1066 |
| LoopBlend S=3 | 4370 | 3030 | 1338 |

### Admission and tests

- `intarsia kernels test --backend metal`: moe_experts / moe_experts_mxfp8 ADMITTED with every form forced (trace confirms
  the form ran; worst case unchanged, 3.472e-3 / 2.099e-4 and 3.378e-3 / 2.453e-4), loop_blend / conv_mix ADMITTED at blend
  levels 0 / 1 / 2 (3.226e-3 / 2.640e-4; trace confirms the NAX blend ran on the 40-row cases).
- `tests/ink_prefill_moe.rs` (2048 rows, real shape, output poisoned with NaN before each run; #10889): every nax2 / nax3 /
  m128 form, NVFP4 and MXFP8, bitwise gen1 (max_abs 0).
- `ink.rel_mode` (wt-rel, #10843): cpu `ink_rel_mode` (mode 1 == mode 0 bitwise over the InkAttnDecode sweep), Metal
  `ink_rel_mode` (prologue + NAX attention, mode 1 == mode 0 bitwise at T 17 / 512 / 2048, SWA and full past 8192),
  `ink_attn_decode`, `inkling2_attn` pass; admission with the appended mode-1 cases: ink_kv_prologue 123 cases ADMITTED,
  rel_sink_attn_bf16_nax 15 ADMITTED, rel_sink_attn_bf16 63 and ink_attn_decode 70 ADMITTED (unchanged).

### Gates with the NAX blend default-on (#10883, bin4 at 355791b1a; trace confirms `blend h nax` + `blend gate nax` ran)

| Gate | 21-way KL scored (bound) | ink/metal baseline | confident top-1 misses |
|---|---:|---:|---:|
| G2 short (oracle) | 2.087e-3 (4.0e-3) | 2.380e-3 | 0 / 363 |
| G2 long (oracle) | 3.239e-3 (6.0e-3) | 3.418e-3 | 1 / 1206 |
| G3 short (vLLM, prefill) | 8.420e-3 (1.05e-2) | 8.391e-3 | 0 / 367 |
| G3 long (vLLM, prefill) | 1.229e-2 (1.65e-2) | 1.248e-2 | 0 / 1211 |

All PASS. Tests: `inkmix` (3, incl. the rows-exact blend at 40 / 300 rows: exact_rows calls keep the GEMV path), `inkmoe` (2),
`ink_prefill_moe` pass.

### Interleaved end-to-end A/B, chunk 2048 (#10884, bin4 = ink/metal a0751f302 + this lane; 2 runs per round)

| Config | 2k | 8k (rounds 1 / 2 / 3) | 12k |
|---|---:|---:|---:|
| `INTARSIA_INK_BLEND_NAX=0` (ink/metal path) | 1796 / 1823 | 1810*, 2457 / 2468, 2208 / 2464, 2356 | 2373 / 2496 |
| **default (NAX blend)** | **1935 / 1952** | **2630, 2595 / 2718, 2609 / 2683, 2626** | **2652 / 2812** |
| MoE m128_k64_v1 | | 1616, 1570 / 1621, 1589 / 1617, 1595 | |
| MoE m128_k32_v3 | | 2022, 1974 / 2018, 2019 / 2025, 1975 | |

(* first run of the job, cold.) Median gain of the NAX blend: 2k +7%, 8k +9% (2440 -> 2640), 12k +12% (2435 -> 2730).
These absolute numbers are ~40% above the #10779 baseline (same commit but ink/metal a88db4620, and a busier box); compare
only within this table. oMLX ~1300, EXO 1750, target 2300-3100.

### Per-chunk split with the default (8k prompt, kernel trace, ms per 2048-row chunk)

cb trace: 4 chunks in 2948 ms (737 ms / chunk), GPU busy 92%, idle 57 ms / chunk.

| Op | ms / chunk |
|---|---:|
| MoE experts NVFP4 | 319 |
| ConvMix (incl. the NAX blend) | 75 |
| RelSinkAttn NAX | 67 |
| InkInProj NAX | 62 |
| InkKvPrologue | 48 |
| MoE experts MXFP8 | 30 |
| wo | 23 |
| router (logits_bf16) | 14 |
| AttnResMix + LoopBlend + MoeRoute + rest | 8 |
| **GPU sum** | **646** |

### localeval speed-plain, base ink/metal e3f92beff vs this branch 378dc214e (#10967, serve --max-batch 1, ABAB)

Brief 5 item 1. `localeval compare` per pair (its rule: under 10 % = noise; "rejected A" = the base run failed the spread check):

| prompt tok | prefill tok/s pair 1 (base -> branch) | pair 2 | decode tok/s pair 1 | pair 2 |
|---:|---|---|---|---|
| ~1k | 1004 -> 1119 (+11.5 %, rejected A) | 1034 -> 1158 (+12.0 %, faster) | 59.1 -> 60.5 | 61.1 -> 61.1 |
| ~4k | 1116 -> 1357 (+21.5 %, faster) | 1226 -> 1258 (+2.6 %, rejected A) | 57.7 -> 60.0 | 59.8 -> 59.5 |
| ~16k | 1193 -> 1304 (+9.4 %, noise) | 1261 -> 1348 (+7.0 %, noise) | 54.6 -> 56.0 | 56.8 -> 56.3 |

Verdict against brief 5 (prefill better at 4k and 16k, decode within 1 %): 4k better in the one valid pair (+21.5 %); 16k
+7 to +9 % in both pairs, under localeval's 10 % noise line; decode within 1 % or better everywhere a pair is valid (worst
-0.9 % at 16k). So: no regression, prefill up, but 16k is not a clean "faster" by localeval's rule.

**Why serve is ~1300 tok/s while `generate --prefill-chunk 2048` is ~2700:** `recipes/inkling2.toml` sets no
`prefill_chunk`, so serve and `intarsia accuracy` run **256-row chunks** (serve-2.log: "prefill chunk 256 (default)"). Every
NAX path in this lane (blend, router, MoE tiles) was tuned and timed at 2048 rows, and G2/G3 above ran at 256. **Every localeval / accuracy number for Inkling on Metal before the recipe commit ran at 256-row chunks (brief 8.4).** Proposal:
`[tuning.metal] prefill_chunk = 2048` in the recipe (as qwen3 / qwen3_5_moe). Check queued as #11109: G2/G3 short+long at
2048, then speed-plain ABAB serve 256 vs 2048 on the same binary.

### The "host gap" is a cold first chunk, not a per-chunk cost (cb trace .scratch/e2e/kw1.cb.jsonl, #10972, 8k prompt)

Idle per 2048-row chunk (span minus GPU busy, chunks split at `embedding_bf16`):

| chunk | command buffers | span ms | busy ms | idle ms |
|---:|---:|---:|---:|---:|
| 0 | 513 | 850.0 | 638.9 | 211.1 |
| 1 | 18 | 657.6 | 657.6 | 0.0 |
| 2 | 18 | 756.7 | 756.7 | 0.0 |
| 3 (1890 rows) | 18 | 733.8 | 733.7 | 0.0 |

The earlier "57 ms per chunk" was this lane's 229 ms prefill idle divided by 4 chunks; it all sits in chunk 0. Of chunk 0's 211 ms,
44 ms is the host committing late, and 167 ms is the GPU starting a committed buffer late. That late start is ~3.7 ms on the
**first** visit of each layer's routed experts (visits 1-18 and 43-48; the second and third core passes and every later chunk
start in 0.1 ms) plus one 17 ms on the first attention. This matches first-touch page-in of each layer's weights, once per
process. Chunks 1-3 run as 128-dispatch buffers with no measurable gap. A warm server (keep-warm on, localeval's per-round
warmup) does not pay it. Batching prefill command buffers therefore has nothing to remove in steady state. Not built;
brief 8's first next item is closed by this measurement. (A cold `generate` pays ~0.2 s once; prefetching the expert weights at load would
remove that and belongs to the residency / keep-warm owner.)

### Steady-state ranking vs floors (brief 9; kernel trace .scratch/e2e/profdefault.k.jsonl, chunks 1 + 2 of an 8k prompt, warm)

ms per 2048-row chunk, the mean of chunks 1 and 2 (GPU 651.7 / 660.6 ms; ctx 2k-6k). Floors: NAX bf16 62 TFLOP/s (see Floors);
memory-bound ops at ~550 GB/s DRAM (assumed, not measured on endor; ops whose operands stay in the SLC can beat it).

| Op | ms / chunk | floor ms / chunk | above floor | floor basis | covered by |
|---|---:|---:|---:|---|---|
| MoE experts NVFP4 (44 visits) | 323.2 | 185 | **138** | 261 GFLOP / visit | v65 / dq1 (#10982) |
| ConvMix incl. the looped blend | 76.5 | n/m | ~15 (fusion bound) | plain 18.6 ms (96 calls); blend 8 x 4 dispatches per looped call, 1.12 ms per call, already at cache speed per dispatch (blend_pair moves 57 MB in 0.043 ms) | - |
| RelSinkAttn NAX | 70.3 | 30 | **40** | SWA 25.8 GFLOP x 40, full ~103 GFLOP x 8 (mean 4k keys); 62 TFLOP/s is optimistic for softmax attention | - |
| InkInProj NAX | 61.6 | 63 | 0 | 81.8 GFLOP / visit | - |
| InkKvPrologue | 47.9 | 12 | **36** | ~140 MB / visit (rel 100.7 MB written, q 12.6, k / v 8.4, pq 14, pk / pv 8.4) | - |
| MoE experts MXFP8 (4 visits) | 30.8 | 17 | 14 | as NVFP4 | v65 / dq1 |
| wo (matmul_bf16_nax) | 22.7 | 22.5 | 0 | 29 GFLOP / visit | - |
| router (logits_bf16) | 15.2 | ~1 | 14 | 1.2 GFLOP / visit | router NAX (#10982) |
| attn_res_mix + loop_blend + moe_route + rest | 7.7 | - | - | under 5 ms each | - |

**Pick: InkKvPrologue (36 ms above floor; attention's 40 assumes an unreachable 62 TFLOP/s for softmax attention, and its
in-kernel-bias variant already lost).** The prologue runs at 1.0 ms per visit, about 140 GB/s for the ~140 MB it must move.
Per (row, kv head) it runs a 256-thread threadgroup in which only 128 threads do the conv / q / r part. The rel product
(rel[t, h, e] = sum_j r[t, h, j] relproj[j, e], 16 x 1024 per head, 75 % of the bytes) re-reads relproj's 32 KB for every row.
Plan: (1) `ink_kv_prologue_norel_bf16` = the prologue without the rel loop, at 128 threads; (2) `ink_rel_proj_bf16`, one
threadgroup per (kv head, 8-row block). It recomputes r with the prologue's own arithmetic (same lane map, simd_sum, clip, tau
and rA rounding), loads relproj once per block into registers, and writes rel with the same per-element fma order (j ascending).
New file kernels/metal/inkling2_prologue_split.metal, flag `INTARSIA_INK_PROLOGUE_SPLIT=1`, default off, branch
`ink/prologue-split` off the ink/prefill tip. **Bitwise** vs the current prologue: same inputs, the same fp32 chains, only the
threadgroup shape changes. Expected: 1.0 -> ~0.35 ms per visit, ~30 ms per chunk (~4.5 % of 656 ms). If it is not bitwise in
the Metal test, I stop and report rather than argue tolerance.

## Next ideas (ranked)

1. MoE experts (54% of the chunk): the gen1 kernel is staging / decode bound, not MMA bound (no-matmul probe 6.3 of 7.3 ms).
   Try decoding each weight block into threadgroup memory once per BN x BK step shared by two M tiles of the same expert
   (2 x 64 rows, 256 threads), and an fp8/fp4 -> bf16 decode with fewer ALU ops; the 128-thread m128 form is not the way.
2. Host gap: closed. It is a once-per-process first-touch cost in chunk 0, 0 ms in later chunks (section above).
3. Router as a NAX GEMM (logits_bf16 runs at 3.9 TFLOP/s: 14 ms -> ~1 ms per chunk).
4. rel_mode: stage the bias per (tile, distance band) in threadgroup memory instead of 16 FMAs per score; then re-measure
   (branch ink/prefill-rel).
5. Metal prefill chunk 2048 in the recipe (serve and accuracy use 256 today): pending #11109.
