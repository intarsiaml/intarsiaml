# Lane MP: Metal prefill vs mlx-lm and llama.cpp (2026-10-22)

Machine: endor (MacBook Pro, Apple M5 Max 40-core GPU, 128 GB). Model: Qwen3-8B BF16, the only model the Metal speed gate uses.
Branch `lane-mp/metal-prefill` off main `06fba3a9`. Raw data, scripts and the experiment patch are in [raw](2026-10-22-mp-metal-prefill/raw/).

## Headline

- **Before this lane, intarsia Metal was already first of the three engines in prefill at 1k, 4k and 16k.** It also led on decode. The "top 3" goal was met at baseline.
- **No code change ships.** I implemented and measured three levers. None was faster, so they were reverted. The diff is kept as
  [raw/attn-variants-experiment.patch](2026-10-22-mp-metal-prefill/raw/attn-variants-experiment.patch). The PR is docs only.
- **After equals before.** No engine or kernel code changed. The competitor numbers below are therefore still the current ones.
- **Qwen3.6-35B-A3B does not run on Metal.** There are no fp8, nvfp4 or MoE kernels, and the GatedDeltaNet ops are not registered
  (`crates/intarsia-backend-metal/src/kernels.rs:553`). The weights are on endor (`~/models/Qwen3.6-35B-A3B-NVFP4`), but bringing the model up
  is a multi-week port, not a prefill-tuning task.
- **No scoreboard with other engines exists.** I checked the `localeval` subcommands, `docs/benchmarking.md` and the deepeval README. The Metal
  speed gate (`scripts/metal-speed-gate/sameslot.sh`: intarsia, mlx-lm, llama.cpp) is the comparison.

## Current numbers vs mlx-lm and llama.cpp

Source: the Metal speed gate same-slot run of 2026-10-21 19:46 EDT (lane MG, fleetq #3302,
[2026-10-21-mg-metal-gate-clock/raw](2026-10-21-mg-metal-gate-clock/raw/)). It uses main `ab4ff9ba`, `localeval speed`, gen 256, 3 reps, clock-admitted arms
(1214–1342 MHz). Between `ab4ff9ba` and `06fba3a9` (`git log -- crates/intarsia-backend-metal kernels/metal crates/intarsia-core crates/intarsia-cli`), only
the golden-dir lookup and the reproducible-build commits landed. Neither touches a Metal kernel or the forward pass. I therefore did not take
another exclusive endor slot to re-run the gate.

Prefill tok/s: the raw rate is prompt / TTFT; the marginal rate is the slope between neighbouring sizes. Both intarsia arms are shown.

| engine | 1k raw | 1k marginal | 4k raw | 4k marginal | 16k raw | 16k marginal | decode 1k / 4k / 16k tok/s |
|---|---:|---:|---:|---:|---:|---:|---|
| **intarsia (Metal)** | **3963 / 3960** | **4462 / 4508** | **3753 / 3851** | **3689 / 3814** | **2717 / 2681** | **2493 / 2441** | **36.0 / 34.9 / 31.2** |
| mlx-lm | 2275 | 4185 | 3041 | 3401 | 2396 | 2242 | 32.7 / 32.1 / 27.9 |
| llama.cpp (Metal, BF16 GGUF, -fa on) | 2532 | 3156 | 2434 | 2400 | 1603 | 1443 | 33.8 / 33.1 / 29.6 |
| intarsia / mlx-lm (raw) | 1.74x | | 1.23–1.27x | | 1.12–1.13x | | 1.10 / 1.09 / 1.12 |
| intarsia / llama.cpp (raw) | 1.57x | | 1.54–1.58x | | 1.67–1.69x | | 1.07 / 1.05 / 1.05 |

Before and after are the same row, because nothing shipped.

## 1. Profile of a 4k prefill (fleetq #3378)

`intarsia generate`, 4096-token prompt, chunk 2048, base binary built from `06fba3a9` (`--features metal`, md5 `6243617c`).
`INTARSIA_METAL_PROFILE=kernel:` (per op) and `cb:` (gaps). GPU at 1617 MHz for every run (macmon). Raw: `raw/s1/`. Table:
`raw/scripts/perop4k.py`.

| kernel | shape (n x k) | calls | ms | share | us/call | TFLOPS |
|---|---|---:|---:|---:|---:|---:|
| matmul_pair_bf16_nax (gate, up) | 24576 x 4096 | 72 | 458.6 | 43.8 % | 6369 | 64.7 |
| matmul_bf16_nax (down) | 4096 x 12288 | 72 | 240.4 | 22.9 % | 3338 | 61.8 |
| matmul_bf16_nax (q, o) | 4096 x 4096 | 144 | 165.3 | 15.8 % | 1148 | 59.9 |
| attn_prefill_bf16_flash_nax | | 72 | 98.2 | 9.4 % | 1363 | about 50 |
| matmul_pair_bf16_nax (k, v) | 2048 x 4096 | 72 | 45.3 | 4.3 % | 630 | 54.5 |
| silu_glu_bf16 | | 72 | 18.8 | 1.8 % | 262 | |
| rmsnorm_add_bf16 | | 144 | 13.7 | 1.3 % | 95 | |
| qk_norm_rope_kv_bf16 | | 72 | 6.8 | 0.7 % | 95 | |
| total | | 724 | 1047.5 | | | |

Against the roofline:

- **GEMM.** The best NAX rate this repo has measured is 64.6 TFLOPS (P1 spike), about 1000 FLOP per clock per core at 1617 MHz. Gate/up runs at
  that peak. The q/o, kv and down GEMMs share K and M with gate/up and differ only in grid size: 1024 / 512 / 1024 threadgroups against 6144.
  Each launch therefore loses about 90–150 us to ramp-up and drain. Over a layer that is about 0.43 ms, 31 ms per 4k prefill (3 %). Recovering it
  needs fewer, larger launches (for example q+k+v as one launch). That means a new op in shared core, so it was not attempted. Tile picks were
  already swept (`docs/results/2026-10-15-metal-prefill-p45.md`, `2026-10-15-metal-gemm-midm.md`).
- **Attention.** It reaches 50–57 TFLOPS (cold clock) against 64.7, and its share is 9 % at 4k and 30 % at 16k
  (`docs/results/2026-10-15-metal-longctx-profile.md`). Section 2 below covers it.
- **Elementwise.** 39 ms (3.8 %), at the DRAM roofline (silu_glu moves 150 MB in 262 us). The only gain is fusion. A SwiGLU GEMM epilogue was
  tried in P4/P5 and was 8 % slower.
- **Command-buffer gaps.** Steady state shows none: chunk 2 runs at 99.9 % GPU busy. **Finding, `generate` only:** the first chunk after model
  load has two idle gaps of about 49 ms each before its first and second command buffers (98 ms, 8 % of a 4k `generate` prefill; one run in slot
  2 had 350 ms). The work had been committed, so this is probably first-touch residency of the freshly mapped weights. I did not measure the serve
  path. Its TTFT (1.06–1.09 s at 4k) is about the GPU busy time, and serve keeps weights resident (`INTARSIA_METAL_KEEPWARM`). So the gap is not a
  scoreboard lever. It does make `generate` wall time a poor A/B metric, so the A/Bs below compare command-buffer busy time.

## 2. Levers implemented and measured (all negative, reverted)

All three are in `raw/attn-variants-experiment.patch`.

### 2a. Prefill chunk 4096 / 8192 (fleetq #3392, `raw/s2/`)

Same binary, `--prefill-chunk`, two rounds. Prefill GPU busy time, ms:

| context | chunk 2048 | chunk 4096 | chunk 8192 |
|---|---|---|---|
| 4k (1611–1617 MHz) | 1058.5 / 1061.1 | 1060.7 / 1061.0 | 1054.6 / 1056.1 |
| 16k, scaled to 1617 MHz | 5377 / (run at 874 MHz, invalid) | 5308 / 5485 | 6728 / 6122 |

Greedy ids (48 tokens at 4k) are identical at all three chunk sizes, confirming chunk invariance. There is no speed gain. At 8192 the 16k prefill
is 10–25 % slower.

### 2b. Skip the O rescale when the running max is unchanged (exact)

Variant `attn_nax_d128_s4_skip`. When the max is unchanged the factor is exactly 1.0f, so the result is bitwise the same: `flash_chunk_invariant`
passes (fleetq #3399). `flash_bench`, chunk 2048 (fleetq #3378, `raw/s1/micro-*`): **no measurable gain**. The clock drifted from 1583 to 1210 MHz
during that pass. Per GHz, at pos0 0 / 2048 / 14336, s4 gave 35.9–39.8 / 36.0–36.3 / 33.9–34.0 and skip gave 36.9–40.5 / 34.5–36.3 / 30.7–31.6,
which is within the drift.

### 2c. Whole-tile MPP products in the NAX flash kernel (exact, slower)

Variant `attn_nax_d128_s4_big`. Per 32-key tile, S = Q.K^T is one 16x32x128 `matmul2d` and O += P.V one 16x128x32. The K and V tiles are read
by the primitive from the device tensor slice. This replaces 16 products of 16x32x16 on cooperative-loaded fragments.

Results:
- **Bitwise identical.** `flash_chunk_invariant` compares every variant with s4 bitwise and passes (#3399). Greedy ids at 4k match (#3415).
- **Much slower.** Clean clock, ABBA, 3 rounds (fleetq #3415, `raw/s3/micro-*`, 1532–1617 MHz):

| pos0 (chunk 2048) | s4 TFLOPS (per GHz) | big TFLOPS (per GHz) |
|---:|---|---|
| 0 | 55.9–56.9 (35.2–36.9) | 50.2–50.6 (31.1–31.3) |
| 4096 | 55.5–56.2 (34.8–36.3) | 43.1–44.6 (26.7–27.6) |
| 14336 | 51.8–55.0 (33.6–34.0) | 32.7–33.2 (20.2–20.5) |

The device-operand path loses the most at long context. A third variant was written but never compiled into a binary, tested or timed, because
fleetq #3419 was cancelled by the server outage: `big3` keeps S, P and O in the cooperative tensors and covers partial tiles with bounds-checked
dynamic slices. It uses the same device-tensor operand path, so I expect it to be slow too.

### 2d. Diagnostics (timing only, wrong output, `raw/s1/micro-*diag*`)

- Dropping the softmax entirely (P = bf16(S)) does not speed the kernel up. The ALU part of the online softmax is hidden.
- Reading every K/V tile from key 0 (cache resident) is about the same at pos0 0–4k and **about 13–15 % faster per GHz at 14k keys**. So at long
  context a share of attention time is K/V traffic through L2/SLC. That is the open lever for 16k and up. Sharing K/V wider without more
  registers is needed:
  - s8 forced for group 4 was flat (10-15 doc).
  - BK = 64 was 3x slower (10-15 doc).
  - Device-tensor operands are slower (2c).

## 3. Cooperative-tensor layout probe (for future NAX kernels)

`raw/probe/` (`probe.metal`, `run.swift`, output `layout.txt`; fleetq #3381) dumps `get_multidimensional_index` for each lane and element of
the matmul2d cooperative tensors on M5 Max. The left, right and destination operands of 16x32x16, 16x32x128, 16x32x32 and 16x128x32 are all
the MLX 16x16 fragment layout concatenated along the inner dimension: element `8 * j + e` is fragment `j`'s element `e`, with rows
`fm, fm + 8` and columns `4 * cq .. 4 * cq + 3`. A fragment kernel can therefore hand its fragments to a larger `matmul2d` without a shuffle.

## Gates

No engine or kernel code changed, so the accuracy gate, `kernels test --backend metal`, the decode-within-1 % A/B and the 8B greedy-ids gate
are **N/A**. Nothing was run for them, and they are not passes.

The bitwise claims for the reverted variants rest on:
- `cargo test --release -p intarsia-backend-metal --test flash_prefill_nax`, 5/5 ok (#3399);
- greedy ids for s4 vs big and for chunks 2048 / 4096 / 8192.

## Not done

- 35B-A3B on Metal: a port, not tuning. It needs MoE, GDN and nvfp4 (or MLX 4-bit) kernels. Decision for the user.
- The interleaved serve A/B x3 at 1k/4k/16k: there was no candidate to A/B. The script is ready (`raw/scripts/abab.sh`, ABBAAB, localeval).
- q+k+v as one launch to cut GEMM tail loss (about 3 % at 4k): needs a shared-core op.
- `big3` timing (cancelled, see 2c).

## fleetq jobs

| job | what |
|---|---|
| #3362, #3370, #3409 | builds |
| #3373 | slot 1; failed at once: `EPOCHREALTIME` without `zsh/datetime` |
| #3378 | slot 1: profile and micro |
| #3381 | layout probe |
| #3392 | slot 2: chunk sizes |
| #3399 | flash tests |
| #3415 | slot 3: big-op A/B; finished, but marking it done failed during the fleetq outage, so it showed stale and I released it by hand |
| #3419 | test and build for `big3`; cancelled by the outage |
