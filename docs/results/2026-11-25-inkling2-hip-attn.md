# Inkling-2-Mini attention on HIP (lane ink-hip-attn, package A, 2026-11-25)

Branch `ink/hip-attn` (origin/main `d8c3c7148` + `ink/hip` skeleton `6c5ee5d67`). Ops `InkKvPrologue` (70), `RelSinkAttn` (71);
stretch `InkAttnDecode` (75). Brief: `.worktrees/hip-swarm/COMMON.md`; port map: `docs/results/2026-11-25-inkling2-hip.md` (ink/hip).

## Briefs received (kept against context loss)

- 2026-11-25 ~21:45 EST, coordinator: the lead's skeleton is committed, `6c5ee5d67` on `ink/hip`; run `git merge ink/hip` now
  (done). Module shape and notes are in the last section of COMMON.md. Op 75 is optional for G2, so land 70 and 71 first; use the
  scalar split-KV `ink_attn_s1_k` + comb form first. Write this message into the results doc (this entry).

- 2026-11-25 ~22:10 EST, coordinator (from the ink-hip lead): when 70/71 pass their unit window, send "ready to merge at
  <commit>". Checkpoint on tatooine `/data/models/intarsia/Inkling-2-Mini` (dev image `/models/Inkling-2-Mini`). Copy (don't edit)
  `docs/results/2026-11-25-inkling2-hip/raw/window.sh` and `run.sh` from `ink/hip`. Set own `INTARSIA_TARGET_VOL=intarsia-cargo-target-<lane>`
  and own container name. No need to re-merge ink/hip. Lead merges lanes after their unit window, then admission + G1 + G2 in one window.
  Window rules unchanged (takes-down radiance, <= 20 min, <= 3 windows, none after 06:00, radiance up before release).

- 2026-11-25 (after 70/71 landed), coordinator, speed round: build fused `InkAttnDecode` (op 75) now, then flash prefill
  attention. Start from `ink/hip` head (has adc5b83a0 + all three lanes; whole model passes G2 there). Lead's profile
  #11945/#11947, one decode token 16.0 ms vs 8.76 ms floor: attention 1.05 ms/token at short context (prologue 0.49, s1 0.51,
  comb 0.05) in 144 dispatches; idle between all 712 dispatches 2.56 ms (~3.6 us each). Op 75 = one dispatch per visit instead
  of three (-96 dispatches). Prefill 4k (chunk 256): RelSinkAttn 12.9 %, becomes largest after the moe lane's WMMA experts.
  Rules: keep op 75's Exactness class from the port map; admission vs cpu-ref must pass; if 75 serves T=1 only, say how T>1
  stays consistent for rows-exact and verify. Op 75 needs a lead-owned line in `model/inkling2.rs` (backend-name rule, hip
  only): don't edit, report when admitted. Flash prefill: admission + G2 must pass, rows-exact stays bitwise where the op
  promises it. 2 windows left (<= 20 min, none after 06:00, radiance up before release). Decode bench before/after; whole-model
  decode number if a window allows. Report each landing as "ready to merge at <commit>" with us before/after.

- 2026-11-25 ~22:20 EST, coordinator, window rule change: no window count cap; each window keeps radiance down <= 5 min; lane
  total radiance-down from now <= 20 min; none after 06:00; radiance must answer `/v1/models` before release; log each window's
  down time here. PR #516 (ink/hip -> main) stays as is: don't merge into ink/hip; send speed heads as "ready to merge at
  <commit>", the coordinator integrates them in a follow-up branch.

- 2026-11-25 ~22:35 EST, coordinator: op 75 v2 accepted (bitwise, non-regressing even forced; lands with the `"cuda" | "hip"`
  core line). Go ahead with the <= 2048-key rule inside the launcher. Flash prefill: -12.4 % at 4k with G2 PASS, but not bitwise
  vs the #516 form, and tonight's merge rule is bitwise only with no output-changing defaults until the user says yes: (1) gate
  `rel_sink_attn_bf16_flash` behind `INTARSIA_HIP_INK_ATTN_FLASH`, default off (other name if HIP has a convention), with a
  `docs/tuning.md` row so `docs_env_vars` passes; (2) keep the default-on flip one constant; (3) record G2 with flash on (#11960)
  next to off. Then "ready to merge at <commit>" with whole-model decode short / 4k with the rule, 4k prefill flash off and on.
- 2026-11-25 ~22:25 EST, coordinator: v2 numbers noted (SWA 25.1 vs 30.1 us fused wins, full 4k 52.0 vs 45.6 pair wins; 20 SWA +
  4 full layers, so per-layer selection gets both). Rule: fused where the visit reads <= 2048 keys, the pair above. Put the rule
  in HIP kernel selection or lane files, not `model/inkling2.rs`, unless impossible (then name the exact core line). Whole-model
  check: decode short and 4k, rule on vs the pair, no regression beyond noise. Then flash prefill speed: report 4k prefill ms per
  chunk before / after.
- 2026-11-25 ~22:20 EST, coordinator: agreed, op 75 v1 is not ready to merge. Go on with v2 (8-wave prologue jobs, faster merge,
  split-size sweep), then flash prefill. If v2 still loses to the pair on full 4k-key layers, report the "fused when the visit
  reads <= 2048 keys" rule as the fallback, measured at whole-model level (short and 4k) against the pair. Integration branch
  starts after PR #516 merges (no rush for v1). Keep the window budget log.

- 2026-11-25 ~22:55 EST, coordinator: PR #518 reviewed clean, merged on green CI. Follow-up: short-prompt prefill 81 -> 100 ms
  (+19 ms; main also captures one graph). (1) Split the 19 ms into the 31-token chunk and the first-step capture (host
  timestamps or HIP trace). (2) Once per process or every request: 3 requests in one `serve` process per build. (3) If per
  request, find the kernel (at T=31 none of the fast forms should be on) and fix on a branch off main after #518 merges.
  <= 5 min radiance down (9:00 left), none after 06:00. Report as before.

## Radiance down time (this lane)

| Window | fleetq | Down | Counts toward the 20 min budget |
|---|---|---|---|
| 1 (70/71 unit) | #11939 | 21:47:58-21:48:32, 34 s | no (before the rule) |
| 2 (op 75 v1: admission, bitwise pair, bench, model decode) | #11952 | 22:10:16-22:12:09, 1 min 53 s | yes (total 1:53 of 20:00) |
| 3 (op 75 v2 + flash: admission, bitwise, benches, G2, model) | #11960 | 22:16:53-22:18:54, 2 min 01 s | yes (total 3:54 of 20:00) |
| 4 (op 75 rule + flash knob: admission x2, bitwise, benches, G2 x2, model) | #11970 | 22:26:29-22:29:07, 2 min 38 s | yes (total 6:32 of 20:00) |
| 5 (PR window `ink/hip-speed`) | #11986 | 22:45:07-22:49:35, 4 min 28 s | yes (total 11:00 of 20:00) |
| 6 (short-prefill probe) | see raw/window6-shortpf.log | 22:53:38-22:55:02, 1 min 24 s | yes (total 12:24 of 20:00; window 5 #11986 4:28 was logged in the ink/hip-speed tree) |

## Step 0

`git log --all -i --grep=hip` and a grep of every `ink/*` branch for `ink_kv_prologue` / `rel_sink_attn` under `kernels/hip` and
`crates/intarsia-backend-hip`: no earlier HIP Inkling attention work exists.

## Status

| Op | Launcher | State |
|---|---|---|
| `InkKvPrologue` 70 | `ink_kv_prologue_bf16` | ADMITTED vs cpu-ref, full sweep 117/117; rows-exact |
| `RelSinkAttn` 71 | `rel_sink_attn_bf16` (scalar split-KV, every T) | ADMITTED vs cpu-ref, full sweep 63/63; rows-exact |
| `InkAttnDecode` 75 | not built | optional (the model runs 70 + 71) |

## Design (kernels/hip/ink_attn.hip)

A straight port of CUDA's correctness form (`ink_prologue_k`, `ink_ring_write_k`, `ink_attn_s1_k<G>`, `ink_attn_comb_k`,
`ink_kv_write_k` of kernels/cuda/inkling2_attn.cu; CUDA's `INTARSIA_INK_DECODE=splitkv` path), same arithmetic and reduction
order: log2-domain scores, the rel bias (d_rel <= 32, zero at distances >= E), the sink once in the final normalisation, the
sigmoid output gate, SWA window via `ink_lo` (ring row `p % Cr`) and full layers (linear cache). HIP changes:
- wave32 shuffles `__shfl_xor(v, o, 32)`, run by every lane (the `!ok` skip stays after the score shuffle); no `__ldg`, PDL,
  cp.async; one-argument `__launch_bounds__`.
- `#pragma clang fp contract(off)` at file scope plus explicit `fadd` / `fmul` / `__builtin_fmaf` where CUDA used `_rn`
  intrinsics (hipcc's default contraction is fast-honor-pragmas), so `build.rs` needs no `NO_FP_CONTRACT_SOURCES` entry.
- The attention runs a call as slices of at most 64 rows in order (as CUDA's tile path does with 16): a slice reads the cache
  rows earlier slices wrote, so rows stay bitwise. Scratch is `min(T, 64) * H * 32 * (Dh + 2) * 4` B (25.6 MB at the real
  model), the same expression in `inkattn.rs::attn_scratch` and the launcher's check. Unsliced it would be 818 MB at a 2048 chunk.
- LDS: static `InkTg<3>` 7.8 KB + dynamic bias <= 3 x E x 4 = 12 KB (< 48 KB). Grids are sized from the capacity bound when
  `pos` is non-null (never from the host `pos0`).

## Tests (tatooine R9700, card 0; fleetq #11932 build, #11939 window, radiance down 21:47:58-21:48:32 = 34 s)

Raw log: `docs/results/2026-11-25-inkling2-hip-attn/raw/window1-11939.log`; stage script `raw/stage_unit.sh`.

| Test (`crates/intarsia-backend-hip/tests/ink_attn.rs`) | Result |
|---|---|
| `admission_full_sweep`, quick sweep (`INK_ATTN_QUICK=1`) | prologue 54/54 (max_abs 2.3e-3), attention 25/25 (max_abs 1.5e-3): ADMITTED |
| `admission_full_sweep`, full sweep (small shapes + real model: window 1024 / ring 1040, full linear to 10240, T 1/4/16/2048) | prologue 117/117 (max_abs 5.5e-3, rel_l2 <= 4.3e-4), attention 63/63 (max_abs 1.7e-3): ADMITTED at BF16_TOL (1.5e-2, 6e-3) |
| `rows_exact_real_shapes` (T 2..256 as one call vs T one-row calls, host and device positions; SWA ring wraps, full layers, pos 7900) | 63 cases bitwise |
| host tests on endor (stub build: `cargo test -p intarsia-backend-hip --lib --test quick_coverage`) | 31 + 1 pass (ABI header, coverage, quick cases) |

## Decode bench (T = 1, real shape 24 / 8 heads, Dh 128; `bench_decode`, eager, 500 launches)

KV bytes per call = 2 x 8 x keys x 128 x 2. Roofline = bytes / 627 GB/s (Neon's number, unverified). The bench re-reads the same
K / V every launch, so the 64 MB Infinity Cache serves part of it (16k keys = 67 MB): numbers are cache-warm, an upper bound
for real decode, where 24 layers' KV do not fit.

| Case | Prologue | Attention | KV GB/s | vs roofline |
|---|---|---|---|---|
| SWA, 1024 keys (4 MB) | 11.7 us | 20.7 us | 203 | 32% (latency bound: 8 x 16 x 1 blocks, 3 launches) |
| full, 4k keys | 12.3 us | 36.9 us | 454 | 72% |
| full, 8k keys | 12.1 us | 57.6 us | 583 | 93% |
| full, 16k keys | 12.6 us | 99.6 us | 674 | 107% (cache) |

Per decode token at ~4k context: 20 SWA layers x (11.7 + 20.7) + 4 full x (12.3 + 36.9) = 0.85 ms of attention. Speed levers
(after G2): fuse prologue + attention (op 75, removes a launch and the q / rel round trip), more splits on SWA rows (more
blocks for the 1024-key case), a combine-free path for one split.

## Op 75 v1 (commit 27ea64c7e; window 2 #11952, raw/window2-11952.log)

- Admission (full sweep): `ink_attn_decode_bf16` 70/70 ADMITTED (max_abs 1.0e-3); 70 117/117, 71 63/63 unchanged.
- Bitwise: R-row and 1-row op 75 calls (host and device positions) equal the pair at T 2..16 (63 cases, all pass).
- Kernel bench (eager back-to-back, cache-warm), pair = prologue + attention, us:

| Case | Pair | Op 75 v1 |
|---|---|---|
| SWA 1024 keys | 32.2 | 33.7 |
| full 4k | 49.0 | 61.5 |
| full 8k | 69.5 | 95.0 |
| full 16k | 113.7 | 164.0 |

- Whole model (`generate`, graphs, 256 tokens, `INTARSIA_INK_ATTN_FUSED=0` vs `1`): short prompt 62.50 / 62.38 -> 63.72 / 63.69
  tok/s (+1.9 %, -0.3 ms / token); 4k prompt 60.47 -> 60.28 tok/s (flat). The fused kernel is slower than the pair's kernels at
  long context (every split block redoes the q / r / k / v jobs in two rounds, one block merges all splits), which eats the
  dispatch savings on the full layers. Next: v2 (8 waves for the prologue jobs, a faster ticket merge, split-size sweep).

## Op 75 v2 + flash prefill (commit 25dccb689; window 3 #11960, raw/window3-11960.log)

- Admission (full sweep): 70 117/117, 71 split-KV 63/63, **71 flash 15/15** (max_abs 6.0e-3, rel_l2 1.6e-3, BF16_TOL), op 75 70/70.
- Bitwise: op 75 v2 R-row / 1-row, host / device position == pair, T 2..16 (63 cases pass).
- G2 with flash prefill (`accuracy --mode prefill`, oracle W4A16 goldens): short PASS, long PASS.
- Decode kernel bench (us; eager, cache-warm). v2 = 8 waves for the prologue jobs, merge over all threads, unrolled merge loops
  (the unroll also sped up the pair's combine: attention 20.5 -> 18.4 us at SWA):

| Case | Pair | Op 75 v1 | Op 75 v2 | v2, no prologue jobs (diag 1) | v2, no merge (diag 2) | v2, neither (diag 3) |
|---|---|---|---|---|---|---|
| SWA 1024 keys | 30.1 | 33.7 | **25.1** | 21.1 | 22.4 | 18.0 |
| full 4k | 45.6 | 61.5 | 52.0 | 44.9 | 48.3 | 41.1 |
| full 8k | 64.9 | 95.0 | 80.2 | 73.3 | 76.2 | 69.5 |
| full 16k | 108.9 | 164.0 | 149.1 | 142.6 | 144.2 | 136.9 |

  Split size sweep (`INTARSIA_INK_SPLIT_KEYS`, both paths): 128 / 256 are worse at SWA (31.7 / 43.5 us fused), equal or worse
  elsewhere; the default 64 stays. The diag rows show the fused body itself is slower than stage 1 at long context (41 vs 33 us
  at 4k without prologue and merge): the 8-wave blocks with the prologue code and the K / V LDS hold fewer waves in flight,
  and the long-context body is latency bound. Hence the per-layer rule below.
- Prefill attention bench, one 256-row chunk (us): SWA at 3840 split-KV 1349 -> flash 185 (7.3x); full at 3840 5207 -> 445
  (11.7x); full at 0 250 -> 58 (4.3x). Per 4k-context chunk (20 SWA + 4 full layers): 47.8 ms -> 5.5 ms.
- Whole model (`generate`, graphs, 256 tokens; flash prefill on in both arms):

| Prompt | Pair (`INTARSIA_INK_ATTN_FUSED=0`) | Op 75 v2 on every visit (`=1`) |
|---|---|---|
| short (31 tokens) decode | 62.48 tok/s | 63.79 tok/s (+2.1 %) |
| 4k (4228 tokens) decode | 61.06 tok/s | 61.69 tok/s (+1.0 %) |
| 4k prefill (chunk 256, 17 chunks) | 7030 ms | 7060 ms |

  Flash prefill vs window 2 (split-KV prefill): 4k prefill 8026 / 8049 -> 7030 / 7060 ms (-12.4 %), 472 -> 414 ms per chunk.
  Short prompt "prefill" 83 -> 92 ms with fused = 1 (it includes the first decode step's capture; to be checked).

## Op 75 with the <= 2048-key rule + flash behind a knob (commit 71158ebf8; window 4 #11970, raw/window4-11970.log)

Op 75's launcher picks its path (`INTARSIA_INK_DEC_FUSED_KEYS`, default 2048): a visit that reads at most 2048 keys (every window
layer; a full layer at a host position up to 2048 context) takes the one-dispatch fused kernel; others (full layers past 2048,
and full layers under a device position, where the launcher sees only the capacity bound) take the split path: the prologue
writing r instead of the [T,H,E] rel table, stage 1 forming its split's bias from r, the combine. Both paths are bitwise the
pair (63 rows-exact cases incl. both paths, host and device positions; admission 70/70). The rule lives in lane files only; the
one core line op 75 needs on HIP is the selection in `crates/intarsia-core/src/model/inkling2.rs` (about line 1203):
`self.backend.name() == "cuda"` -> `matches!(self.backend.name(), "cuda" | "hip")` (measured below with
`INTARSIA_INK_ATTN_FUSED=1`, which forces the same).

Flash prefill now runs inside `rel_sink_attn_bf16` (calls of more than 16 rows) when `INTARSIA_HIP_INK_ATTN_FLASH=1`; default off
(`INK_ATTN_FLASH_DEFAULT` in inkattn.rs, the one-line flip). Off is bitwise the #516 form. `docs/tuning.md` rows added for
`INTARSIA_HIP_INK_ATTN_FLASH`, `INTARSIA_INK_DEC_FUSED_KEYS`, `INTARSIA_INK_DEC_DIAG`. (`docs_env_vars` still fails on
`INTARSIA_HIP_INK_MOE_PRE` and `INTARSIA_HIP_SELECTION_DUMP`, other lanes' knobs already on ink/hip, not added here.)

- Admission (full sweep): flash off: 70 117/117, 71 63/63 (max_abs 1.7e-3), 75 70/70. Flash on: 71 63/63 (max_abs 6.0e-3).
- Decode kernel bench (us; host position), op 75 by path:

| Case | Pair | Op 75, rule (2048) | Op 75, split path only (`=0`) | Op 75, fused only |
|---|---|---|---|---|
| SWA 1024 keys | 30.8 | **25.1** (fused) | 30.3 | 25.1 |
| full 4k | 46.2 | **41.9** (split) | 42.2 | 51.3 |
| full 8k | 65.8 | **61.8** (split) | 61.9 | 79.4 |
| full 16k | 108.4 | 110.1 (split) | 111.3 | 148.6 |

- Prefill attention bench, one 256-row chunk (us): SWA at 3840 split-KV 1510 -> flash 180; full at 3840 5610 -> 437; full at 0
  262 -> 58.
- G2 (`accuracy --mode prefill`, oracle W4A16 goldens; same build, flash off vs on):

| Golden | Flash off: KL21 / top-1 | Flash on: KL21 / top-1 | Verdict |
|---|---|---|---|
| short (569 rows) | 2.566e-3 / 98.95 % | 4.491e-3 / 99.47 % | PASS / PASS (floor 3.0e-2) |
| long (1569 rows) | 3.251e-3 / 99.30 % | 4.521e-3 / 99.04 % | PASS / PASS |

  Flash raises KL21 by 1.3-1.9e-3 (bf16 WMMA operands, P rounded to bf16 against the running max). Within the class floor; the
  short value is above the 4e-3 figure in the lead's gate-script comment, so the user's flip decision should see it.
- Whole model (`generate`, graphs, 256 tokens):

| Run | Short decode | 4k decode | 4k prefill |
|---|---|---|---|
| pair (`INTARSIA_INK_ATTN_FUSED=0`), flash off | 62.44 tok/s | 60.83 tok/s | 8093 ms |
| op 75 with the rule (`=1`), flash off | **63.62 tok/s (+1.9 %)** | **61.79 tok/s (+1.6 %)** | 8143 ms |
| op 75 with the rule, flash on | | 61.62 tok/s | **7082 ms (-12.5 %; 476 -> 417 ms per 256-token chunk)** |

## Handoff (wiring; HANDOFF.md was removed by the ink/hip merge bcfbcf066, so it lives here)

| Launcher (TABLE, src/inkattn.rs) | Op | ABI symbol (ik_hip_abi_inkattn.h) | Inputs -> outputs | Scratch |
|---|---|---|---|---|
| `ink_kv_prologue_bf16` | 70 | `ik_ink_kv_prologue_bf16` | i(0..8) -> o(0..6), pos | none |
| `rel_sink_attn_bf16` | 71 | `ik_rel_sink_attn_bf16` | i(0..7) -> o(0..3), sc, pos | `attn_scratch`: min(T, 64) * H * 32 * (Dh + 2) * 4 B |
| `ink_attn_decode_bf16` | 75 | `ik_ink_attn_decode_bf16` | i(0..10) -> o(0..5), sc, pos | `decode_scratch`: T * H * 32 * (Dh + 2) * 4 + 2 * T * Hk * Dh * 2 + T * H * (Dh + 32) * 2 B (T <= 16) |

- Registration: bf16, `Exactness::Deterministic`, `BF16_TOL (1.5e-2, 6e-3)`; no `.rows_exact()` / `.row_table()` / `.priority()`.
  Op 75: `decode_ok` (attention shapes, T <= 16, Hk <= 256).
- Core line for op 75 on HIP: `crates/intarsia-core/src/model/inkling2.rs` about line 1203, `self.backend.name() == "cuda"` ->
  `matches!(self.backend.name(), "cuda" | "hip")`. Nothing else outside lane files.
- Op 75's merge uses a self-resetting device ticket array (`ink_dec_tick`): launches on one stream (decode is single-stream;
  graph replay is fine). No `build.rs` change: `ink_attn.hip` has `#pragma clang fp contract(off)` and build.rs passes no
  `-ffp-contract` / `-ffast-math`.
- Flash prefill default: `inkattn::INK_ATTN_FLASH_DEFAULT = false` (the one-line flip, user's call).

## Short-prompt prefill +19 ms on #518: once per process (HIP lazy code-object loading), not capture, not per request

Window 6 (raw/window6-shortpf.log): main `4fc9901d4` vs the #518 tree, short prompt (31 tokens), 32 tokens, one process per line.

| Probe | main prefill | #518 prefill |
|---|---|---|
| first request in the process | 81 / 85 ms | 95 / 122 ms |
| third request in one process (`--repeat 3`, sequence reset, no prefix reuse) | 70 ms | **69 ms** |
| no graphs (`INTARSIA_SOLO_GRAPHS=0`: 0 captures, 31 eager steps) | 81 ms | 94 ms |
| all code objects loaded at start-up (`HIP_ENABLE_DEFERRED_LOADING=0`) | 77 ms | **74 ms** |
| attn + mix fast forms off (`INTARSIA_INK_MIX_FAST=0 INTARSIA_INK_ATTN_FUSED=0`) | | 95 ms |

- Capture: 0 of it. The prefill clock stops at the first token (`engine.rs` `accept_first`), before the first decode step
  captures, and with graphs off the gap stays (81 vs 94).
- Per request: none. A warm request is equal (70 vs 69 ms).
- Cause: the first prefill pays HIP's deferred loading of the code objects whose kernels it launches first. With
  `HIP_ENABLE_DEFERRED_LOADING=0` the gap is gone (77 vs 74). The #518 tree's Inkling code objects are larger (op 75, the
  flash kernel, the moe forms, the mix fast forms); turning only the attn / mix forms off does not remove it (95 ms), as a code
  object loads whole. Whole-process time is the same (5.3 s both, either loading mode).
- First-request prefill varies 94-122 ms on the branch (81-85 on main), so the size of the once-per-process cost is 13-37 ms.
- `serve` was not run: `generate --repeat 3` is the same engine and process with the sequence reset between requests.
  `serve` on HIP runs admission cases at start-up (lazy admission is off on HIP), which likely loads these code objects
  before the first request; not measured.
- No fix branch: nothing per request to fix. Optional: setting `HIP_ENABLE_DEFERRED_LOADING=0` (or a warm-up pass) moves the
  cost to start-up; not proposed as a default here.

## Next

- Coordinator: integrate `ink/hip-attn` (HANDOFF.md) with the one core line above; the flash default flip is the user's call.
