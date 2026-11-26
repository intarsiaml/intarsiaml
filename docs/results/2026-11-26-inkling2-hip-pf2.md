# Inkling-2-Mini prefill on HIP after the flip (lane ink-hip-pf2, 2026-11-26)

Branch `ink/hip-pf2` from main `87c9745c0`. One R9700 (tatooine card 0, gfx1201). Shared rules:
`.worktrees/hip-swarm/COMMON-1126.md`.

## 0. Brief (coordinator, 2026-11-26 ~11:45 EST; kept here against context loss)

- Goal: Inkling-2-Mini prefill faster on HIP, starting from the new prefill defaults that lane ink-hip-pfflip is flipping
  (MoE tensor-core prefill, flash prefill attention, chunk 2048). Until its PR merges, run main with
  `INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1 --prefill-chunk 2048` (2687 tok/s at 4k, 2793 at 16k).
- Do not edit `kWmDefault`, `INK_ATTN_FLASH_DEFAULT` or `recipes/inkling2.toml [tuning.hip]` (lane ink-hip-pfflip owns them);
  merge `origin/main` once its PR lands.
- Step 1 (first window): rocprofv3 `--kernel-trace` prefill profile at 4k and 16k with the new settings; per kernel class
  (experts WMMA, in-projection / wo / other GEMMs, attention, mix and row kernels, unembed, host gaps between chunks) ms per
  chunk against its roofline (bf16 / fp8 WMMA peak for GEMMs, bandwidth for the rest); pick the 1-3 levers with the most ms
  left. Candidates: unembed on every prefill row, in-projection on WMMA, MoE WMMA tile at 2048 rows, flash occupancy, gaps
  between chunks.
- Bar: speed >= +5 % prefill tok/s at 4k AND 16k, 4 + 4 paired processes, 95 % CI lower bound > 1.0, decode unchanged;
  G2 short and long within limit, admission for touched ops. Forms that keep the bits of the new defaults can be default-on
  behind a knob whose `=0` reverts; forms that change output are opt-in unless >= +15 % (then flagged for a user decision).
- Report once when it ships or is ruled out (or after about 4 hours), COMMON report format, speed table first.
- Coordinator message (~12:20 EST), after this lane reported that COMMON's build line `--load cpu --mem 10` counts 10 GB
  against the card's 32 GB that radiance holds (so a CPU build waits for a radiance window and blocks the queue): "you were
  right"; COMMON-1126.md no longer has `--mem` on tatooine; batch cancels and resubmits #12035, spec resubmits #12037 if it
  carries `--mem`; keep #12036 (this lane's window 1) in line.

## 1. Ceilings used

- bf16 WMMA: issue peak 194 TFLOPS, hipBLASLt at prefill shapes ~165-170 TFLOPS; fp8 issue peak 377, hipBLASLt ~300
  (measured on this card by lane hp, `docs/results/2026-10-22-hp-hip-prefill-roofline.md` section 1; not re-measured here).
- Bandwidth 627 GB/s (the HIP lanes' figure; the bf16 GEMVs reach 92-96 % of it).
- Dispatch stall: every dispatch of about 1 ms or more is followed by ~256 us in which the GPU starts nothing
  (`kernels/hip/ik_hip.h`, `docs/results/2026-10-19-hip-round9.md`).

## 2. Window 1: where a 2048-row prefill chunk goes (main 87c9745c0 with the new settings; fleetq #12036)

Radiance down 11:49:34 - 11:50:23 (0:49). Binary: main `87c9745c0` built in CPU job #12033 (md5 `3e70c45a…`).
`generate --backend hip --spec off --prefill-chunk 2048 --max-tokens 2`, `INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1`,
4k prompt (4228 tokens: chunks 2048 + 2048 + 132) and 16k (16915 tokens: 8 x 2048 + 531). Unprofiled in the same window:
4k 1550 ms (2727.6 tok/s), 16k 6008 ms (2815.3 tok/s); under rocprofv3 1575 / 6021 ms. Stage `raw/stage_prof.sh`, analysis
`raw/pftrace.py` (chunks split at the embedding dispatch), output `raw/w1-split.txt`, traces `raw/w1-ktrace-{4k,16k}.csv.gz`.

ms per 2048-row chunk, kernel time, at context 2048 (4k prompt, chunk 1; the 16k prompt's chunk 1 is the same within 0.5 ms)
and at context 14336 (16k prompt, chunk 7). FLOPs are 2 per MAC of the work the class must do (attention: causal, the
window on SWA layers); bytes are what the class must move at least.

| Class (kernels) | calls | ms, ctx 2k | ms, ctx 14k | Work per chunk | Achieved | Ceiling | Floor ms | Left ms (ctx 2k / 14k) |
|---|---|---|---|---|---|---|---|---|
| Experts NVFP4 gate_up (`ink_moe_wmma_kernel<0,2,true,64>`) | 44 | 187.1 (4.25 / call) | 186.5 | 173.9 GFLOP / call | 41 TFLOPS | 170 (hipBLASLt bf16, all pairs as one GEMM); 35B's `moe_wmma_kernel_r7` reaches 62 | 45 (at 170); 124 (at 62) | 142 / 142 (63 at the 35B kernel's rate) |
| Experts NVFP4 down (`<0,2,false,64>`) | 44 | 89.9 (2.04) | 89.2 | 87.0 GFLOP / call | 43 TFLOPS | 170; 35B r7 56 | 22; 68 | 68 / 67 (22) |
| Experts MXFP8 gate_up + down (`<1,..>`) | 4 + 4 | 26.6 | 25.8 | as NVFP4 | 39-40 TFLOPS | as above | 6 | 21 / 20 |
| Experts glue (gather 8.5, group 1.1, route 0.7) | 144 | 10.3 | 10.3 | ~30 MB / visit | | bandwidth | ~2.3 | 8 |
| Attention prologue (`ink_prologue_k<4>`) | 48 | 80.3 (1.67) | 80.0 | 157 MB / call (rel table `[T,H,E]` 100.7 MB of it) | 94 GB/s | 627 GB/s | 12.0 (4.3 without the rel table) | 68 / 68 |
| Flash attention (`ink_flash_k<128,3>`) | 48 | 80.5 (1.68) | 149.7 (3.12) | 1.65 / 4.12 TFLOP | 20.5 / 27.5 TFLOPS | ~100 (hipBLASLt QK^T / PV at 99-145; the 35B's flash runs 81) | 16.5 / 41.2 | 64 / 108 |
| KV / ring writes | 96 | 1.5 | 1.5 | | | | | ~0 |
| In-projection + wo (`gemm_wmma_kernel<4,2,2,4,bf16>`, 3 + 1 per visit) | 192 | 52.1 | 52.0 | 5.32 TFLOP | 102 TFLOPS | ~165 (hipBLASLt bf16 at these shapes) | 32 | 20 / 20 |
| Router (`gemm_wmma_kernel<..,float>`) | 48 | 1.7 | 1.7 | 58 GFLOP | 34 TFLOPS | | | ~1 |
| Mix: ConvMix / AttnResMix (`ink_row_kernel<3,false,true>`) | 460 | 42.8 (0.093) | 42.8 | ~60-70 MB / call | ~650-750 GB/s (est.) | 627 GB/s | ~40 | ~0 |
| Blend (`ink_ctl_gemv` 23.7, `ink_blend_gate` 16.5, `ink_blend_norm` 5.9) | 1152 | 46.1 | 46.1 | | | bandwidth | ~20 (est.) | ~25 |
| Unembed + argmax | 1 / prefill | 0 per chunk (last row only) | | | | | | 0 |
| Idle between dispatches | 2388 | 60.1 | 59.9 | | | | | |
| of which the dispatch stall (gaps of ~274 us, one after each launch of >= 1 ms) | 192 | 52.6 | 52.4 | | | 0 | 0 | 53 / 52 |
| **Chunk wall** | | **679.3** | **745.9** | | | | | |

Notes:
- The unembed runs on the last prompt row only (`head_groups` gives `(len - 1, 1)` for `LastArgmax`; the trace has one head
  dispatch after the last chunk). Not a lever.
- The 192 stall gaps per chunk are exactly the 48 visits x the four launch types over 1 ms: experts gate_up (4.25 ms) and
  down (2.04 ms), the prologue (1.67 ms) and flash (1.06 - 3.12 ms). In-chunk idle without the stall: 7.5 ms (3 us per
  dispatch). Between chunks: 0.01 ms (the host is ahead); the first chunk waits 7.2 ms (4k) for host set-up.
- The row kernel is at about the bandwidth line (estimated bytes, not counted); the GEMMs are at 62 % of hipBLASLt.

## 3. Levers picked and built (`INTARSIA_HIP_INK_PF2`, a bit mask; every bit keeps the output bits)

From section 2, ms left per chunk, largest first: experts (~230 at the hipBLASLt line, ~85 at the 35B kernel's rate), attention
(prologue 68, flash 64 / 108), the dispatch stall (53), GEMMs (20), blend (25). Picked, in the order of ms per hour of work:

1. **The prologue's rel tail as its own pass** (bit 1, `ink_rel_table_k`). `ink_prologue_k` runs one block per (kv head, row)
   and forms that block's G x E = 3072 rel entries with 16 relproj loads each: 100 MB of writes took most of its 1.67 ms. With
   the bit, the prologue writes r into the first d_rel entries of each rel row (`F_R_ROW`) and returns; `ink_rel_table_k` (8 rows
   per block, relproj columns held in registers) forms the rows with ink_rel_g's chain (from 0, j ascending, ffma) and rounds,
   the same bits. Prefill calls only (T > 16).
2. **Flash attention in pieces of query tiles under the stall line** (bit 2). A call estimated over 950 us runs as pieces of
   consecutive query tiles cut by key tiles, highest first; a piece keeps >= 128 blocks or the call stays one launch. A block of a
   piece is the block of the one launch with the same query tile (`qhi`). In practice the 40 sliding-window visits split in two;
   the full-attention visits need more pieces than the block floor allows and stay one launch (8 stalls per chunk, ~2 ms).
3. **The experts' down / gate_up in pieces of feature tiles** (bits 4 / 8; opt-in, see window 2 and 3) and the **OCC form** of
   the experts kernel (bits 16 / 32: compiled for 8 waves per SIMD, gate_up's up accumulators on the activation tiles' LDS, as
   the 35B's `moe_wmma_r7.h` kMwOcc). `INTARSIA_HIP_INK_PF2_ZP` sets the pair-tile blocks per expert (tuning).

Not levers: the unembed (last row only), the gaps between chunks (0.01 ms), the row kernel (at the bandwidth line).

## 4. Window 2: microbench, 4k A/B, bitwise tests (pf2a `4ad8a0f04`; fleetq #12046)

Radiance down 12:11:43 - 12:12:47 (1:04). Binary pf2a (CPU job #12044, md5 `a1c3aaa8…`). Log `raw/window2-12046-unit.log`,
stage `raw/stage_unit.sh`, profile split `raw/w2-split-mask7.txt`.

- Bitwise (each against `INTARSIA_HIP_INK_PF2=0`): `pf2_bitwise_prefill` 140 attention cases (sliding / full, T 17 - 2048,
  flash on and off, pieces at the default rule, at 700 us and one query tile each) and `ink_moe_pf2_pieces` 30 expert cases:
  all equal.
- Bench (2048 rows, wall per call back to back): prologue 2096 -> 920 us (sliding), 2205 -> 1026 (full); flash 2162 -> 1500
  us (sliding at 2048: the stall gone), full at 2048 / 14336 unchanged (one launch).
- 4k prefill, one process each: mask 0 2738 tok/s, 1 2891 (+5.6 %), 3 2923 (+6.8 %), 7 2835, 15 2642.
- Bits 4 / 8 lost because the experts' MAC-rate constant was 1000x low: every down / gate_up ran as 18 pieces of one feature
  tile (profile: down 792 launches of 155 us = 122.9 ms per chunk against 89.6). Fixed in pf2b (21e6 MAC per us: down 3 pieces,
  gate_up 5) and re-measured in window 3. The profile at mask 7 also shows the rest working: prologue 29.6 ms + rel table 15.2
  ms per chunk (80.1 before), flash 88 launches with no stall after the sliding ones, idle 27.3 ms (60.1).

## 5. Window 3: the bits one by one (pf2c `b61cd1315`; fleetq #12056)

Radiance down 12:29:41 - 12:33:13 (3:32). Binary pf2c (CPU job #12052), still on main `87c9745c0` with the new settings set by
env + `--prefill-chunk 2048`. Log `raw/window3-12056-ab.log`, stage `raw/stage_ab.sh`, profile split `raw/w3-split-mask3.txt`.
One process per configuration, so single samples (the two mask-0 and mask-3 repeats agree within 0.7 %):

| `INTARSIA_HIP_INK_PF2` | 4k tok/s | 16k tok/s | vs 0 (4k / 16k) |
|---|---|---|---|
| 0 (main's forms; two runs) | 2711.5 / 2691.9 | 2805.9 / 2791.5 | |
| 3 (rel pass + flash pieces; two runs) | 2900.0 / 2882.3 | 3015.6 / 3007.3 | +6.9 % / +7.6 % |
| 7 (+ down in 3 pieces) | 2923.8 | 3041.1 | +7.9 % / +8.6 % |
| 15 (+ gate_up in 5 pieces) | 2904.1 | 3010.0 | +7.2 % / +7.5 % |
| 19 (3 + the OCC gate_up) | 2613.1 | 2681.1 | -3.6 % / -4.2 % |
| 51 (3 + OCC gate_up and down) | 2498.8 | 2552.9 | -7.8 % / -8.8 % |
| 67 (3 + experts in pair-count order) | 2874.9 | 3000.1 | +6.1 % / +7.2 % |
| 3, `INTARSIA_HIP_INK_PF2_ZP=16` | 2969.7 | 3107.4 | +9.6 % / +11.0 % |
| 3, `INTARSIA_HIP_INK_MOE_WMMA=32` (32-pair tile) | 2638.6 | 2813.1 | -2.6 % / +0.4 % |

- Bitwise: `ink_moe_pf2_pieces` 72 cases (pieces, OCC, pair-count order, ZP 1 / 16) equal to one launch at mask 0.
- Dropped (no gain): the OCC form of the experts kernel (192 VGPRs at 8 waves per SIMD costs more than the occupancy buys) and
  the pair-count slot order; gate_up in pieces (its 5 pieces each pay the hot experts' tail: no better than one launch with the
  stall); the 32-pair tile.
- Kept: down in pieces (+1 % over 3) and the wider pair-tile grid (ZP 16: +2.5 % / +3.3 % over 3), now bit 16 (eight times the
  average pairs per expert over the tile, 1 to 64: 16 at a 2048-row chunk; main's rule gives 4). Default mask 23.
- Profile at mask 3 (4k, chunk 1): 633.8 ms per chunk (679.3 in window 1): prologue 30.7 + rel table 15.7 ms (80.3), flash
  82.5 ms in 88 launches, idle 37.7 ms of which 104 stalls (gate_up 48, down 48, full-attention flash 8).
