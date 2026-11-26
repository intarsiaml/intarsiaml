# Inkling-2-Mini prefill on HIP after the flip (lane ink-hip-pf2, 2026-11-26)

Branch `ink/hip-pf2` from main `87c9745c0`, merged with main `40b85a391` (lane ink-hip-pfflip's new prefill defaults, #526). One
R9700 (tatooine card 0, gfx1201). Shared rules: `.worktrees/hip-swarm/COMMON-1126.md`.

**Status (13:20 EST): PR #528 open (not merged), default-on bitwise knob `INTARSIA_HIP_INK_PF2` (default 151; `=0` is main's forms).**
Prefill +12.9 % at 4k and +13.9 % at 16k over main `40b85a391` at its defaults; greedy output identical to main (md5 at 64 and
256 tokens, short / 4k / 16k); decode unchanged.

| tok/s, one R9700 (main `40b85a391` vs branch pf2e, both at their defaults) | main | branch | change | paired 95 % CI |
|---|---|---|---|---|
| Prefill 4k (4228 tokens), 4 + 4 ABBA | 2692.2 | 3040.4 | +12.93 % | [1.1273, 1.1313] |
| Prefill 16k (16915 tokens), 4 + 4 ABBA | 2790.4 | 3177.8 | +13.89 % | [1.1361, 1.1416] |
| Decode short, 256 tokens, 2 + 2 | 76.9 | 76.9 | -0.01 % | [0.9982, 1.0015] |
| Decode 4k, 256 tokens, 2 + 2 | 74.1 | 74.0 | -0.04 % | n = 2 |

Window 6 (section 8). What moved: the prologue's 100 MB rel table as its own pass (80 -> 46 ms per 2048-row chunk), flash
attention and the experts' down launched in pieces under the R9700's ~1 ms dispatch stall line (the ~274 us idle after every
long launch), and 4x the pair-tile blocks per expert in the experts' tensor-core kernel (hot experts' tiles side by side).

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
3. **The experts' down / gate_up in pieces of feature tiles** (bits 4 / 8), and **more pair-tile blocks per expert** (bit 16)
   so a hot expert's tiles run side by side instead of 2 - 3 per block one after another.
4. **The prologue compiled for 12 waves per SIMD** (bit 128).

**Final bit map** (`INTARSIA_HIP_INK_PF2`, default 151 = 1 | 2 | 4 | 16 | 128; `=0` is main's forms; every bit keeps the bits):

| Bit | Form | Default |
|---|---|---|
| 1 | prologue's rel tail as its own pass (`ink_rel_table_k`, 8-byte stores) | on |
| 2 | flash attention in pieces of query tiles under the stall line | on |
| 4 | experts' down in pieces of feature tiles under the stall line | on |
| 8 | experts' gate_up in pieces (no gain: window 3) | off |
| 16 | experts' pair-tile grid eight times the average pairs per expert over the tile, 1 to 64 (main: twice, 1 to 8) | on |
| 128 | with bit 1, the prologue compiled for 12 waves per SIMD | on |

Tried and removed (window 3, pf2c only): the experts kernel's OCC form (then bits 16 / 32) and the experts in pair-count order
(then bit 64). In windows 2 and 3 the bit numbers above 8 meant those removed forms; window 4 on uses the map above.

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

## 5. Window 3: the bits one by one (pf2c `b61cd1315`; fleetq #12056; pf2c-era bit numbers)

Radiance down 12:29:41 - 12:33:13 (3:32). Binary pf2c (CPU job #12052), still on main `87c9745c0` with the new settings set by
env + `--prefill-chunk 2048`. Log `raw/window3-12056-ab.log`, stage `raw/stage_ab.sh`, profile split `raw/w3-split-mask3.txt`.
One process per configuration, so single samples (the two mask-0 and mask-3 repeats agree within 0.7 %):

| `INTARSIA_HIP_INK_PF2` | 4k tok/s | 16k tok/s | vs 0 (4k / 16k) |
|---|---|---|---|
| 0 (main's forms; two runs) | 2711.5 / 2691.9 | 2805.9 / 2791.5 | |
| 3 (rel pass + flash pieces; two runs) | 2900.0 / 2882.3 | 3015.6 / 3007.3 | +6.9 % / +7.6 % |
| 7 (+ down in 3 pieces) | 2923.8 | 3041.1 | +7.9 % / +8.6 % |
| 15 (+ gate_up in 5 pieces) | 2904.1 | 3010.0 | +7.2 % / +7.5 % |
| 19 (3 + the OCC gate_up; removed) | 2613.1 | 2681.1 | -3.6 % / -4.2 % |
| 51 (3 + OCC gate_up and down; removed) | 2498.8 | 2552.9 | -7.8 % / -8.8 % |
| 67 (3 + experts in pair-count order; removed) | 2874.9 | 3000.1 | +6.1 % / +7.2 % |
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

## 6. Merge of main with the new defaults, and window 4 (main2 vs pf2d; fleetq #12071)

- Lane ink-hip-pfflip's PR #526 landed (main `40b85a391`: `kWmDefault = 1`, `INK_ATTN_FLASH_DEFAULT = true`, recipe
  `[tuning.hip] prefill_chunk = 2048`). Merged into the branch (`323e079f3`; conflicts only in `tests/ink_moe.rs`, both sides'
  new tests kept, and in `docs/tuning.md`, both sides' rows kept). From here, main is `40b85a391` at its defaults (no env, no
  `--prefill-chunk`); its binary `main2` (CPU job #12061, md5 `492211cf…`). The branch's tests pin flash with `set_flash`, as
  main's tests do, instead of the env.
- A stale build caught by md5: the first pf2d (#12065) had main2's md5. `rsync -a` kept the branch files' mtimes, older than the
  main2 build, so cargo saw nothing to rebuild. `raw/touch_diff.sh` now touches every file that differs between main and the
  branch after each sync; pf2d rebuilt in #12069 (md5 `96cfcc3f…`).

Radiance down 12:52:26 - 12:56:00 (3:34). Log `raw/window4-12071-ab.log`, stage `raw/stage_ab4.sh`. One process each:

| Binary, setting | 4k tok/s | 16k tok/s |
|---|---|---|
| main2 (two runs, first and last) | 2726.2 / 2690.5 | 2819.7 / 2790.0 |
| pf2d `INTARSIA_HIP_INK_PF2=0` | 2721.2 | 2815.1 |
| pf2d 3 (rel pass, flash pieces) | 2908.1 | 3027.7 |
| pf2d 7 (+ down pieces) | 2932.4 | 3050.3 |
| pf2d 19 (3 + the wider pair-tile grid) | 2987.1 | 3119.8 |
| pf2d 23 (the pf2d default; two runs) | 3021.6 / 3015.9 | 3155.8 / 3152.9 |
| pf2d 23, ZP 8 / 32 / 64 | 2998.4 / 3015.0 / 3010.9 | 3137.0 / 3157.2 / 3151.5 |
| pf2d 151 (23 + the prologue's OCC form) | 3048.5 | 3180.3 |

- `=0` matches main2 (the revert). The default 23 is +11.5 % / +12.5 % over main2's mean; bit 128 adds +0.9 % / +0.8 % (one
  sample each, above the 0.2 - 0.7 % spread of the repeats) and is bitwise, so the final default is 151 (pf2e). The pair-tile
  count is flat from 16 to 64; 8 is 0.6 % lower.

## 7. Window 5: gate on pf2e (final defaults, mask 151; fleetq #12077)

Radiance down 13:04:34 - 13:07:35 (3:01). Binary pf2e `6fb63f30a` (CPU job #12074, md5 `2e6379f2…`); main2 = main `40b85a391`.
Log `raw/window5-12077-gate.log`, stage `raw/stage_gate.sh`. All at the defaults (no env, recipe chunk 2048).

- Greedy md5 (card 0): short 64 tokens M = B = B0 `04cdb28a…`; 4k 64 tokens M = B = B0 `2252566a…`; 16k 16 tokens M = B
  `363ebce0…` (M main2, B the branch default, B0 the branch at `INTARSIA_HIP_INK_PF2=0`). The 256-token md5s at short and 4k
  are in the speed window (section 8).
- G2 (branch default, `accuracy --mode prefill`): short 21-way KL 4.491e-3, long 4.083e-3 (limit 3.0e-2), both PASS; the same
  values lane ink-hip-pfflip recorded for main's defaults.
- Admission (branch default): `ink_kv_prologue_bf16` 117 cases, `rel_sink_attn_bf16` 63, `moe_experts_situ_bf16` 10 (and
  the bf16 / v2 / v3 / rows names), `moe_experts_mxfp8_bf16` 19 / `_rows` 14: all ADMITTED.
- hip crate tests (card 1, the branch's test binaries): `ink_attn` 5 passed (`pf2_bitwise_prefill` 196 cases bitwise `=0`,
  incl. bit 128 and the 8-byte rel table; `rows_exact_real_shapes` 63; `flash_prefill_default_and_revert`; quick admission
  sweep), `ink_moe` 7 passed (`ink_moe_pf2_pieces` 60 cases bitwise one launch; `ink_moe_wmma_default_and_revert`; rows-exact).
- `cargo test -p intarsia-cli --test docs_env_vars` (endor CPU job #12078): 2 passed (the four new `docs/tuning.md` rows).

## 8. Window 6: speed, main2 vs pf2e (fleetq #12081)

Radiance down 13:10:39 - 13:14:19 (3:40). Log `raw/window6-12081-speed.log`, stage `raw/stage_speed.sh`, analysis
`raw/speed.py` -> `raw/window6-speed-analysis.txt`. Prefill: 4 + 4 processes per prompt, ABBA over the reps, `--max-tokens 2`;
decode: 2 + 2 at 256 tokens. Every pair of processes gave the same md5 (2 tokens at 4k / 16k, 256 tokens at short / 4k:
`9a27c21b…`, `9675bbb9…`).

| Prompt | main tok/s (4 runs) | branch tok/s (4 runs) | paired ratio | 95 % CI |
|---|---|---|---|---|
| 4k prefill | 2686.5 - 2698.9 (mean 2692.2) | 3034.4 - 3047.8 (mean 3040.4) | 1.1293 | [1.1273, 1.1313] |
| 16k prefill | 2786.0 - 2793.3 (mean 2790.4) | 3174.0 - 3181.3 (mean 3177.8) | 1.1389 | [1.1361, 1.1416] |
| short decode (256) | 76.88 / 76.89 | 76.86 / 76.89 | 0.9999 | [0.9982, 1.0015] |
| 4k decode (256) | 74.05 / 74.10 | 74.02 / 74.07 | 0.9996 | n = 2 |

Decode is unchanged by construction: every bit acts on calls above 16 rows (attention) or on the experts' pair-table
tensor-core path (the rule: >= 16 pairs per expert on average), which decode never takes; the 256-token md5s match.

Bar (brief): >= +5 % at 4k and 16k with a CI lower bound > 1.0: met (+12.9 % / +13.9 %). Output: the same bits as the new
defaults, so default-on behind the knob, `INTARSIA_HIP_INK_PF2=0` reverting (window 4: `=0` 2721 / 2815 tok/s against main2's
2726 / 2820). Note: `pf2_mask` reads the knob with `atoi` and takes the default for a value that does not start with a digit,
so `INTARSIA_HIP_INK_PF2=off` means the default (on); the revert is `=0`.

## 9. Radiance downtime and jobs

| Window | fleetq | Down | Total |
|---|---|---|---|
| 1 profile of main 87c9745c0 (new settings) | #12036 | 11:49:34 - 11:50:23, 0:49 | 0:49 |
| 2 bench, 4k A/B, bitwise (pf2a) | #12046 | 12:11:43 - 12:12:47, 1:04 | 1:53 |
| 3 bits one by one + profile (pf2c) | #12056 | 12:29:41 - 12:33:13, 3:32 | 5:25 |
| 4 main2 vs pf2d A/B | #12071 | 12:52:26 - 12:56:00, 3:34 | 8:59 |
| 5 gate (pf2e) | #12077 | 13:04:34 - 13:07:35, 3:01 | 12:00 |
| 6 speed (pf2e) | #12081 | 13:10:39 - 13:14:19, 3:40 | 15:40 |

Every window restarted radiance in its EXIT trap and saw `/v1/models` answer 15 s later. CPU jobs on tatooine: #12033 (main
87c9745c0), #12044 (pf2a), #12050 (pf2b), #12052 (pf2c), #12061 (main2), #12065 (pf2d, stale: same md5 as main2), #12069
(pf2d), #12074 (pf2e); #12030 cancelled (queued with `--mem 10`, see section 0). endor CPU job #12078 (`docs_env_vars`).

## 10. Left for later (ms per 2048-row chunk at ctx 2k, after this branch)

- Experts: gate_up ~4.2 ms and down ~2.0 ms per call at ~41 TFLOPS (bf16 WMMA issue peak 194): ~280 ms per chunk, the largest
  item by far. The 35B's r7 OCC form did not carry over (slower here); the next try is the kernel's k loop itself (asm per WMMA).
- Attention: flash ~82 ms (20 - 28 TFLOPS); the rel table could go entirely if flash formed the bias from r as CUDA's RIN
  kernel does (~0.5 ms per visit: the 15 ms table pass plus the table read).
- `ink_ctl_gemv` 24 ms per chunk, a GEMV at prefill (2048 rows x rank 64): a WMMA form changes the sum order (opt-in only).
- The stall after gate_up (48 per chunk, ~13 ms): its pieces cost as much as they save (window 3).
