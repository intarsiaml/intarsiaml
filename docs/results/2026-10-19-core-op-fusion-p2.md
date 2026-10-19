# Core-op fusion, phase 2: P4a (gate split folded into the prologue) and P3 (GDN conv folded into the delta rule), 2026-10-19

Branch `perf/core-op-fusion-p2`, final rebase on origin/main `8d1f16a5` (CUDA round 5 merged; HIP round 7 `49973a51` and
phase 1 `f3cbba21` included). The HIP numbers below are measured on the branch's final kernels against `49973a51` in the same
windows. The base tree is `49973a51` plus the bench-client commit `06b5237d` (`bench speed --at-most`, no engine code).
`49973a51..8d1f16a5` is CUDA, CLI and docs only, so the HIP base numbers stay valid; HIP was not re-measured after the rebase. Plan: `docs/proposals/2026-10-19-core-op-fusion.md`
sections 5 (P4a) and 6 (P3). P2' and P1 are not started. Raw logs: `docs/results/2026-10-19-core-op-fusion-p2/raw/`.

## Headline

* **Qwen3.6-35B-A3B NVFP4 decode on one R9700: 149.0 -> 150.9 tok/s at a 128-token prompt (+1.28 %), +1.25 to +1.31 % at
  every prompt size up to 16384.** Prefill +0.38 to +0.53 % at 1024 to 16384 (-0.09 % at 128, inside the round spread).
  Interleaved A/B over 3 rounds, same window (W4). Every round of B is above every round of A in decode at every size.
* **40 fewer launches per 35B decode step: 415 -> 375** (`plan show --json`): 10 gated-attention layers lose `gate_split`,
  30 GatedDeltaNet layers lose one of `gdn_conv` / `gdn_delta_rule`. Qwen3-8B: 364 -> 364, no change.
* **Bitwise.** Both kernels give the bits of the chains they replace: on-device bitwise tests, including the conv state and the
  delta state after every launch of a chain of launches with chunk boundaries, 220 fresh draws, and a race seam. The 35b-short
  and 35b-long KL21 digits equal base. The 256 greedy ids after the 4408-token prompt equal base. The prefix-reuse gate passes.
  The full admission sweep has 0 FAIL. All 17 GPU mutants and 1 lib mutant are caught on the final build, including the 4
  re-enabled round-6 `moe_wmma.h` mutants.
* **Default on, on HIP only.** CUDA and Metal register neither op, so `fused_ok` is false and they run the chains they ran
  before. Pin tests on both backends prove it. On spark1 the CUDA `plan show` JSON is byte-identical for the 35B and the 8B, and
  admission is identical.
* One real defect was found and fixed on the way. The first P3 kernel flipped one bf16 rounding in about 3e5 conv outputs: the
  compiler contracted a different product into the first fma (section 3). The fix spells out the unfused kernel's order.

## 1. Results vs base (HIP, tatooine R9700 gfx1201, W4, interleaved A1 B1 A2 B2 A3 B3)

35B (`raw/bench/pf-35b-A*.json` vs `pf-35b-B*.json`, serve bench with graphs, 5 reps per size, `--at-most` prompts;
`raw/bench/ab-35b.txt`):

| prompt | base prefill tok/s | P2 prefill tok/s | delta | base decode tok/s | P2 decode tok/s | delta |
|---|---|---|---|---|---|---|
| 128 | 1746 / 1760 / 1772 (1759) | 1732 / 1770 / 1770 (1758) | -0.09 % | 148.8 / 149.0 / 149.0 (149.0) | 150.9 / 150.9 / 150.9 (150.9) | +1.28 % |
| 1024 | 6463 / 6455 / 6431 (6450) | 6466 / 6446 / 6511 (6474) | +0.38 % | 148.1 / 148.3 / 148.2 (148.2) | 150.0 / 150.1 / 150.0 (150.1) | +1.27 % |
| 4096 | 8975 / 8985 / 8934 (8965) | 9011 / 8978 / 9047 (9012) | +0.53 % | 142.8 / 143.0 / 142.9 (142.9) | 144.6 / 144.4 / 145.2 (144.8) | +1.31 % |
| 16384 | 7634 / 7651 / 7653 (7646) | 7686 / 7677 / 7670 (7678) | +0.42 % | 132.5 / 131.8 / 132.2 (132.2) | 134.2 / 134.2 / 133.0 (133.8) | +1.25 % |

8B (`raw/bench/pf8x-A*` vs `pf8x-B*`, 3 reps per size; `raw/bench/ab-8b.txt`):

| prompt | prefill delta | decode delta |
|---|---|---|
| 1024 | -1.01 % (4536 -> 4490) | +0.04 % |
| 4160 | -0.22 % | -0.05 % |
| 16384 | +0.07 % | +0.14 % |

* The 8B runs no new code: it has neither gated attention nor GatedDeltaNet layers, so neither `try_fused` site builds a call,
  and its plan rows are identical. The -1.01 % at 1024 is noise. The `--at-most` prompts drew 953 to 959 tokens on B against 971
  to 985 on A, and a shorter prompt has a lower prefill rate at this size. This is inferred from those counts, not measured
  separately.
* **Prompt sizes:** every 35B row stays under its size (`--at-most`, section 2). The "4096" prompts drew 3977 to 4032 tokens on
  both sides, so none paid the second 4096-token prefill chunk that round 7 hit (4079 to 4120 tokens). The "128" row draws
  108 to 111 tokens: 2 % under the size, minus the fixed instruction.

**Launches per decode step** (`plan show --json`, sum of `calls_per_decode_step`; `raw/plans/diff.txt`, W3 rerun in
`raw/logs/w3.log`):

| model | base | P2 | rows |
|---|---|---|---|
| 35B | 415 | 375 (-40) | `gate_split` 10 -> 0 and `qk_norm_rope_kv` 10 -> 0, `gate_split_qk_norm_rope_kv` 10; `gdn_conv` 30 -> 0 and `gdn_delta_rule` (v2) 30 -> 0, `gdn_conv_delta` 30 |
| 8B | 364 | 364 | 14 rows identical |

Scratch is unchanged: 167905792 bytes for the 35B, 748800 for the 8B.

## 2. What was built

**Core (`crates/intarsia-core`), the same template as phase 1:**

* Two `Op`s appended at the end of the enum (`ops.rs`): `GateSplitQkNormRopeKv` (5 in, 4 out) and `GdnConvDelta` (6 in, 3 out).
  Their checks reuse the checks of the ops they compose, with the same messages:
  - `OpCall::gate_split_inner` gives P4a as the `QkNormRopeKv` call it ends in.
  - `gdn::conv_delta_parts` gives P3 as its (`GdnConv`, `GdnDeltaRule`) pair.
  - P4a takes a device position (graph replay) like `QkNormRopeKv`. P3 takes none, like the GDN ops.
* `plan.rs` `PER_ROW_OPS` now lists `GateSplitQkNormRopeKv` (the `check_batch_ops` trap). It holds the per-row cache append
  like `QkNormRopeKv`. `Model::enable_batch` rejects gated attention today, so no batched plan can contain it.
* **P4a call site** (`model.rs`):
  - `Model::gate_qkv_call` builds the fused call. It reads the q+gate projection and writes q to `bufs.q[1]`, the gate to the
    output gate's buffer, and k / v to the caches. It returns `None` for a block that is not gated or lacks a norm.
  - `attn_block` decides with `fused_ok`, before the q projection, because the split it replaces runs right after that
    projection.
  - On false, `attn_project` runs `GateSplit` and `attn_rows` runs `qkv_fold` or the six-op chain, exactly as before.
  - The batched path (`batch.rs`) passes "not folded".
* **P3 call site** (`gdn_block.rs`): `try_fused(GdnConvDelta, ...)`. On false, `GdnConv` then `GdnDeltaRule` run as before.
* **CPU oracle** (`intarsia-backend-cpu/src/fused_ref.rs`): each op is the composition of the ops it replaces, with the
  intermediate stored in the activation dtype as the unfused path stores it.
* **Admission** (`admission.rs`):
  - Sweep cases. P4a uses the `QkNormRopeKv` style: pos0 0 / 37, partial rotary, both norm offsets. P3 uses T 1 / 2 / 5 / 17 /
    32 with non-zero conv and delta states, so a kernel that ignores either state fails.
  - Plan-time labels and exact-shape case builders for both ops.
  - `gdn_conv_delta_case` is public, for the device tests.

**HIP (`kernels/hip`, new files):**

* `gate_qkr.hip`, `gate_split_qk_norm_rope_kv_bf16` (P4a): `qk_norm_rope_kv_kernel` statement for statement, with two
  changes.
  - A query wave reads its head's D query values from the gated row (stride 2D), where `gate_split_kernel` would have copied
    them bit for bit.
  - It also copies the head's D gate values to the gate output.
  - Same constraints as the prologue: D % 8 == 0, D <= 256, N / Nkv <= 8. bf16 or fp8 cache.
* `gdn_fused.hip`, `gdn_conv_delta_bf16` (P3): `gdn_delta_rows_v2_kernel<128, 2, PD = 1>` statement for statement. Each lane
  computes the q / k / v values it reads from the raw projection, using `gdn_conv_kernel`'s per-thread loop with the output
  rounded to bf16 as `y` is stored. The constraints are dk = 128, 4 taps and T <= 32 (the conv's single tile: every decode
  step and short prefill tails).
  - **The conv-state hazard** (proposal 6f). The old conv state must be read by every block that needs it before the new one
    is stored.
    - A v channel has one reader, the block of its value row. That block stores the channel itself at the end.
    - The q / k channels of key head kh are read by all of kh's blocks (16 at the 35B's shape). Each block reads them, then
      takes a ticket on kh's counter (one atomic, acq_rel, agent scope; after a `__syncthreads`).
    - The block that draws ticket nblk - 1 knows that every sibling has read the old state. It stores the new state after its
      token loop and puts the counter back to 0.
    - No block waits for another, so there is no spin and no deadlock whatever the dispatch order.
    - The counters are one 64-entry slot of `g_gdn_cd_sync` per (device, stream), like `moe.hip` v3's `g_moe_sync`. Without a
      free slot (more than 16 streams, or Hk > 64), the fused kernel stores no conv state and a second small launch stores it,
      with the same bits.
  - Resources (gfx1201, hipcc -O3): 232 VGPRs, 0 scratch, occupancy 6, 4096 bytes of LDS. The q / k conv history lives in LDS
    and the taps are packed bf16 pairs.
* Registration (`src/kernels.rs`) at priority 10, `Deterministic`. FFI and dispatch (`ffi.rs`, `device.rs`), and the ABI
  header (`ik_hip_abi.h`).

**Tests:**

* `crates/intarsia-backend-hip/tests/fusion_p2.rs` (GPU), against the unfused launches on the same device, bitwise:
  - **P4a.** The admission sweep at 16/2x256, 32/8x128 and 8/8x128, bf16 and fp8 cache: q, gate, k cache and v cache. Over 100
    cases.
  - **P3, chains of launches.** Each path carries its own states. T = 5, 1, 32, 1, 17, 2, 1, 31, 3 (chunk boundaries) on five
    shapes: the 35B's, idle lanes, one block per key head, and three value heads per key head. `o`, the delta state and the
    conv state are compared after every launch, all three reported.
  - **P3, race seam.** `INTARSIA_HIP_GDN_CD_SKEW` delays every other block's state reads. Each step also runs unseamed on the
    same inputs, so a timing effect is told apart from an arithmetic one.
  - **P3, without a slot.** The two-launch form.
  - **P3, many draws.** 200 T=1 and 20 T=32 fresh draws at the 35B shape.
  - The unfused side runs at `INTARSIA_HIP_GDN_PD` 1, 2 and 4. Plus the cpu reference within the admission budget.
* `crates/intarsia-models/tests/fusion_p2.rs` (CPU), on the tiny hybrid fixture (GatedDeltaNet layers plus 2 gated attention
  layers), against a backend without the two ops:
  - Every logit and the recurrent state (conv + delta state) are byte-identical after a 12-token prefill (chunks of 8 + 4), 5
    decode steps, a snapshot restore (the prefix-reuse path) and 5 other decode steps.
  - The fold counts in a step are checked, and the fallback runs exactly the chain.
  - A second test runs P4a in the same blocks as phase 1's P5 and P4b (fp8 attention, W8A16 and W8A8).
* `crates/intarsia-models/tests/graphs.rs`: the recorded graph now shows `GateSplitQkNormRopeKv` (with a device position) in
  place of `GateSplit` + `QkNormRopeKv`.
* `ops.rs` `check_phase2_folds`: the checks, shape facts and `gate_split_inner`.
* Pin tests:
  - HIP `phase2_fold_selection_is_pinned`.
  - CUDA and Metal `phase2_folds_are_absent_and_the_unfused_selection_is_pinned`: neither op is registered, and the unfused
    chain's selection at the 35B's shapes (m = 1 and 32) is pinned to origin/main's kernel names.

**Scripts:**

* `scripts/hip-decode/mutate.py`:
  - An occurrence index for patterns found twice; the 4 round-6 mutants now apply.
  - 13 `p2-*` mutants.
  - The 4 obsolete `mw-*` moe.hip mutants are dropped (section 6).
* `fusion-p2.sh` (stages `mutbuild`, `mutrun`, `tests`, `adm`, `idsoff`, `plans`). `fusion-window.sh` takes `FUSION_SCRIPT`.
  `mac-checks.sh`.
* `scripts/cuda-decode/fusion-noop-p2.sh`. `fusion-noop{,-up}.sh` take `NOOP_A` / `NOOP_B`.
* `intarsia bench speed --at-most` (`crates/intarsia-bench`, CLI, `remote.sh AT_MOST=1`): prompts aim 2 % under the size, and a
  measured rep whose prompt the server counted longer is redone with fewer sentences and a fresh nonce. Rows record `at_most`
  and `reps_redone_over_size` (0 to 4 redos per row in W4).

## 3. Tried and dropped

* **P3 conv with implicit accumulation (`acc = t3*cur; acc += t0*h0; ...`)**, the first version. W1's race-seam test failed:
  15 of 4096 `o` elements in one key head were off by one bf16 ulp. W2 (`raw/logs/w2.log`) told the two causes apart.
  - The fused kernel matched itself with and without the seam: no timing effect.
  - A new many-draws test failed at draw 34: 1 `o` element off, both states identical.
  - Cause: with K known, the first add sees two products, and the contraction may fuse either one. `fma(t3, cur, t0*h0)` rounds
    differently from `fma(t0, h0, t3*cur)`. `gdn_conv_kernel`'s runtime-K loop compiles to one `v_mul_f32` and one
    `v_fmac_f32` per history tap (read from the ISA).
  - Fix: `cd_conv` writes that order explicitly under `fp contract(off)`. Mutant `p2-cd-contract` restores the old form, and the
    many-draws test catches it.
* **P3 with taps in f32 registers and prefetch depth 2 / 4 (v2's default 2):**
  - Taps in f32 registers: 255 / 256 VGPRs with 124 / 228 bytes of scratch per lane.
  - Taps read from memory: still spilling.
  - Packed bf16 taps: 52 / 160 bytes at PD 2 / 4, 28 bytes at PD 1.
  - Kept: the q / k history in LDS and PD = 1, which gives 0 scratch. The fused kernel ignores `INTARSIA_HIP_GDN_PD`. The prefetch
    depth changes no bit (v2's own claim; the tests run the unfused side at 1 / 2 / 4).
* **P3 for chunks longer than 32 tokens:** not attempted. The unfused conv tiles at 32 tokens (`kConvTile`), and the fused
  kernel keeps the single-tile form. Longer prefill chunks keep `gdn_conv_bf16` + `gdn_delta_rule_bf16_v2`.
* **Round-6 mutants `mw-def64`, `mw-def32`, `mw-tile`, `mw-range`:** round 7 replaced the `moe.hip` lines they mutate. The
  round-7 entries `r7-def*`, `r7-tile` and `r7-range` cover the new lines, so the 4 old ones are dropped.

## 4. Accuracy (W3 final build; base `49973a51` in W1 on the same base tree; `raw/logs/w3.log`, `w1.log`)

| check | base (W1) | P2 final (W3) |
|---|---|---|
| 35b-short golden, KL21 prefill / decode | 7.417e-3 / 1.172e-2, PASS | 7.417e-3 / 1.172e-2, PASS |
| 35b-long golden (chunk 4096), KL21 prefill / decode | 4.356e-2 / 9.273e-3 | 4.356e-2 / 9.273e-3 |
| 35b-long prefix-reuse gate | PASS | PASS |
| 4408-token prompt, 256 greedy ids | - | IDENTICAL to base |
| same, kill switch `INTARSIA_HIP_DISABLE=gate_split_qk_norm_rope_kv_bf16,gdn_conv_delta_bf16` | - | IDENTICAL to base |
| 8B fp32 goldens, short / long, full-vocab KL prefill / decode | 2.335e-4 / 3.626e-4, 3.966e-4 / 3.573e-4 (PASS) | the same digits, PASS |

* The base digits equal the task's f3cbba21 base (short 7.417e-3 / 1.172e-2, long 4.356e-2 / 9.273e-3). Round 7 changed no
  bit.
* The 35b-long prefill floor "FAIL" (4.356e-2 against the absolute 3.0e-2 floor) is pre-existing and identical on base. Its
  decode and its prefix-reuse gate pass.
* The kernels the W1 accuracy ran with had the rare flip of section 3, and W1 gave the same digits. The W3 numbers are from the
  final kernels.

## 5. Bitwise tests and mutants (HIP, W3 GPU 1, final build, `raw/logs/w3.log`)

* `fusion_p2` at `INTARSIA_HIP_GDN_PD` 1 / 2 / 4: 5 of 5 pass each time. "P3: 220 draws bitwise".
* The suites of the kernels the folds mirror all pass: `fusion_p1` (P5 24, P2a 48, P4b 14 cases), `gdn_v2` (19 shapes),
  `qk_rope_kv` (240 cases, 60 draws), `moe_pairs`, `quantmoe`, `fused_decode`, `decode_v2`, `device_pos`, and the lib (16).

**Mutants**, built in a CPU slot and run on the final build. Every one is caught:

| mutant | what it breaks | caught by |
|---|---|---|
| `p2-ga-gate` | gate copied from the query half | P4a bitwise |
| `p2-ga-qstride` | query read with stride D, not 2D | P4a bitwise |
| `p2-ga-woff` | zero-centred norm's +1 dropped | P4a bitwise |
| `p2-ga-fp8` | fp8 cache value halved | P4a bitwise (fp8 cache) |
| `p2-ga-vpos` | every token's v to the first new row | P4a bitwise |
| `p2-cd-writer0` | FIRST block of a key head stores the new q / k state | race-seam test (and chain) |
| `p2-cd-noreset` | key-head counter not reset | race-seam test (and chain) |
| `p2-cd-novstate` | v channels' new conv state not stored | conv-state compare |
| `p2-cd-round` | conv output not rounded to bf16 | chain |
| `p2-cd-tap` | current input times the first tap | chain |
| `p2-cd-contract` | conv accumulation left to contraction (section 3) | many-draws test |
| `p2-cd-hist` | conv history not shifted | chain |
| `p2-cd-slotstate` | no-slot form skips its state launch | no-slot test |
| `p2-sel-m33` (lib, Mac) | fold admitted at 33 tokens | `phase2_fold_selection_is_pinned` (`raw/logs/mut-lib.log`) |
| round 6 `moe-z-step` | z-step of the pair-tile loop | `moe_pairs` |
| round 6 `pairs-sred` | reduction stride | `moe_pairs` |
| round 6 `pairs-xrow` | x row of the 64-pair tile | `moe_pairs` |
| round 6 `pairs-out` | output loop of the tile | `moe_pairs` |

* The 4 round-6 mutants matched twice since round 6, so they never applied. `mutate.py` now takes `(index, count)`. They mutate
  `moe_wmma_kernel_v`, the second copy.
* `moe-z-step` used to be checked by `quantmoe`, whose default-env case runs `moe_wmma_r7.h` since round 7. It moved to
  `moe_pairs`, which runs `INTARSIA_HIP_MOE_MW` 1 / 4 / 5 against 0.
* All 134 mutations in `mutate.py` apply on the final tree (checked on the Mac).
* In W1 the clean build failed the race-seam test (section 3). That made every `p2-cd-*` kill in W1 meaningless, so W1 is not
  cited. The table is W3.

## 6. Admission per backend

| backend | base | P2 |
|---|---|---|
| HIP, full sweep (not `--quick`), 35B shapes | 49 kernels, 853 cases, 0 FAIL | 51 kernels, 899 cases, 0 FAIL (`gate_split_qk_norm_rope_kv_bf16` 36 ADMITTED, `gdn_conv_delta_bf16` 10 ADMITTED) |
| HIP, full sweep, 8B shapes | 49 kernels, 866 cases, 0 FAIL | 51 kernels, 912 cases, 0 FAIL |
| CUDA, `kernels test --quick`, 35B / 8B (spark1, rebased trees) | 50 kernels, 149 cases, 0 FAIL; exit 0 with round 5's "incomplete: 9 kernel(s) had no cases" line | identical apart from timing lines |
| CPU | the new ops' oracle is the composition of the unfused ops | - |
| Metal | no new kernel; pin test only (section 11) | - |

## 7. Memory

* Scratch per plan is unchanged (35B 167905792, 8B 748800 bytes on HIP; CUDA 83963392 / 748800).
* New device memory: `g_gdn_cd_sync`, 16 slots x 64 counters x 4 bytes = 4 KiB of module globals.
* P3 uses 4096 bytes of LDS per block (v2: 1024). P4a uses the prologue's LDS.

## 8. Every shared file touched, and what changes per backend

| file | CPU | HIP | CUDA | Metal |
|---|---|---|---|---|
| `crates/intarsia-core/src/ops.rs` (2 appended `Op`s, `gate_split_inner`, checks, shape facts, test) | new ops known | new ops known | known, unregistered | known, unregistered |
| `crates/intarsia-core/src/gdn.rs` (`GdnConvDelta` check / shape, `conv_delta_parts`) | used by the oracle | facts it selects on | unused | unused |
| `crates/intarsia-core/src/plan.rs` (`PER_ROW_OPS`) | no batched plan holds the op | same | same: byte-identical plans | same |
| `crates/intarsia-core/src/admission.rs` (sweep cases, plan-time labels and cases) | cases run on the oracle | cases run | unregistered op: no case runs | same |
| `crates/intarsia-core/src/model.rs` (`gate_qkv_call`, `attn_project` / `attn_rows` parameters, `attn_block` decision) | folds run (oracle) | folds run | `fused_ok` false: the old chain | same |
| `crates/intarsia-core/src/model/batch.rs` (passes "not folded") | unchanged behaviour | same | same | same |
| `crates/intarsia-core/src/model/gdn_block.rs` (`try_fused(GdnConvDelta)`) | fold runs | fold runs | false: the old chain | same (no GDN model loads) |
| `crates/intarsia-backend-cpu/src/{fast,reference,fused_ref}.rs` | oracle | - | - | - |
| `crates/intarsia-backend-cuda/src/kernels.rs` (exemptions, pin test) | - | - | test only | - |
| `crates/intarsia-backend-metal/src/kernels.rs` (pin test) | - | - | - | test only |
| `crates/intarsia-models/tests/{fusion_p2,graphs}.rs` | tests | - | - | - |
| `crates/intarsia-bench/src/speed.rs`, `crates/intarsia-cli/src/cmd/bench.rs` (`--at-most`, default off) | bench client | same | same | same |
| HIP only: `kernels/hip/{gate_qkr,gdn_fused}.hip`, `ik_hip_abi.h`, `crates/intarsia-backend-hip/src/{kernels,ffi,device}.rs`, `tests/fusion_p2.rs` | - | new kernels | - | - |
| scripts: `scripts/hip-decode/{mutate.py,fusion-p2.sh,fusion-window.sh,mac-checks.sh,remote.sh}`, `scripts/cuda-decode/{fusion-noop.sh,fusion-noop-up.sh,fusion-noop-p2.sh}` | - | measurement only | measurement only | - |

## 9. CUDA no-op (spark1, `fp2-base` vs `fp2-dev` = this branch)

**vLLM-up slot on the rebased trees** (job 2714: `fp2-base` = `8d1f16a5` export, `fp2-dev` = HEAD `60847700`;
`fusion-noop-p2.sh up`; `raw/logs/cuda-up-rebased.log`, `raw/cuda/adm/`):

* **`plan show --json`, 35B and 8B:** byte-identical A / B (md5 prefixes `26379bc9beb2`, `d1847a362f17`). This is the
  `check_batch_ops` / `PER_ROW_OPS` trap check with real plan loads. 35B: 525 launches per decode step on both; 8B: 544 on both.
  Scratch unchanged (83963392 / 748800).
* **`kernels test --backend cuda --quick`, 35B and 8B:** 50 kernels, 149 cases on both trees, identical apart from timing lines.
  Since round 5 the CLI exits 0 with an "incomplete: 9 kernel(s) had no cases" line. The scripts compare logs, not exit codes.
* The same checks on the pre-rebase trees (job 2711: `49973a51` + bench commit vs the branch then) gave the same two md5
  prefixes and identical admission logs (`raw/logs/cuda-up.log`).

**vLLM-down window** (job 2712, pre-rebase trees: `fp2-base` = `49973a51` + bench commit, `fp2-dev` = the branch at `9665bf35`;
`fusion-noop-p2.sh` = `w-noop.sh fp2-base fp2-dev nospeed`; `raw/logs/cuda-window.log`, `raw/cuda/`):

* **Greedy ids, 256 tokens, 35B and 8B:** B-graph, A-eager and B-eager all match A-graph (6 of 6 comparisons "IDS ... == A-graph").
* **35b-short golden (prefill chunk 256), A and B:** the same KL21, 6.271e-3 prefill / 1.556e-2 decode, PASS on both. These are
  CUDA's own digits.
* **Self-golden dump compare:** 60 tensors per tree, 0 differ, per-position TSV identical.
* **8B fp32 goldens on B:** short 2.897e-4 / 3.238e-4, long 4.004e-4 / 3.332e-4 (full-vocab KL prefill / decode), PASS.
* This window ran before the rebase onto `8d1f16a5`. It was not repeated: round 5's fp8 KV path is opt-in, the default CUDA
  plan is what the rebased `plan show` compare covers, and it is byte-identical.

## 10. Metal no-op check to run (Metal spike session, endor)

Base origin/main `8d1f16a5` (the branch's merge-base) vs this branch's HEAD (section 14).

**Getting the branch:** `perf/core-op-fusion-p2` is local and unpushed. Someone pushes it after review, or the Metal session takes
a bundle: `git bundle create core-op-fusion-p2.bundle 8d1f16a5..perf/core-op-fusion-p2` on this Mac, then
`git fetch core-op-fusion-p2.bundle perf/core-op-fusion-p2` on endor.

1. `cargo test --workspace` on endor. It includes `intarsia-models` `batch_metal` and `fusion_p2`, plus the Metal
   `phase2_folds_are_absent_and_the_unfused_selection_is_pinned` and phase 1's `kernel_list_is_unchanged_by_core_op_fusion_p1`.
2. `intarsia plan show --backend metal --model <Qwen3-8B> --json` on both trees. The JSON must be byte-identical. This is the
   `check_batch_ops` / `PER_ROW_OPS` trap check. The 35B NVFP4 checkpoint does not load on Metal before or after, since there
   are no GatedDeltaNet, fp8, nvfp4 or MoE kernels.
3. `intarsia kernels test --backend metal --model <Qwen3-8B>`: kernel and case counts equal on both trees.
4. 8B greedy ids (256 tokens) and the two 8B fp32 goldens (`acc8`) on both trees: identical ids, identical KL digits.
5. With `--max-batch 8` (batched decode): the batch identity test. `cargo test -p intarsia-models --test batch_metal` covers it.

## 11. Caveats and env vars

* **Kill switch:** `INTARSIA_HIP_DISABLE=gate_split_qk_norm_rope_kv_bf16,gdn_conv_delta_bf16` (verified: 4408 ids identical to
  base). `INTARSIA_NO_FUSION` turns off every fused op, as before.
* **P3 counter slots:** one per (device, stream), 16 in all, never freed, like `moe.hip` v3. A process that launches on more
  than 16 streams falls back to the two-launch form, with the same bits. The scheme assumes what v3 assumes: launches that
  share a slot (one stream) do not overlap. A graph captured on one stream and replayed concurrently on two streams would
  break that.
* **P3 scope:** dk = 128, 4 taps, T <= 32, bf16. The decode step and prefill tails up to 32 tokens; the bulk of a long prefill
  keeps the unfused pair. The +0.4 % prefill comes mostly from P4a, which has no token limit. This is not separated by
  measurement.
* **Test seams (default off):** `INTARSIA_HIP_GDN_CD_SKEW=<n>` (every other block reads its state late) and
  `INTARSIA_HIP_GDN_CD_NOSLOT=1` (two-launch form). Both are read at each launch.
* **`INTARSIA_HIP_GDN_PD`** still sets the unfused v2 kernel's prefetch depth. It does not apply to `gdn_conv_delta_bf16` (PD 1).
* **Bench:** `bench speed --at-most` / `remote.sh AT_MOST=1` (default off). Sizes in W4 are the prompts' true counts, which run
  1 to 16 % under the label (section 1). Older rounds' JSONs are not comparable row for row.
* The 8B's -1.01 % prefill at 1024 is attributed to prompt length, not code (section 1). This is inferred.

## 12. Downtime and slots

| service | window | job | down | up | minutes |
|---|---|---|---|---|---|
| radiance (tatooine) | W1: admission, KL x2 trees, ids, kill switch, 8B; tests, mutants, plans | 2689 | 05:23:45 | 05:32:19 | 8.6 |
| radiance | W2: P3 discriminating test | 2694 | 05:44:08 | 05:44:51 | 0.7 |
| radiance | W3: final build, accuracy, tests, 17 mutants, plans | 2702 | 05:58:12 | 06:04:25 | 6.2 |
| radiance | W4: interleaved A/B x3, 35B and 8B | 2707 | 06:04:57 | 06:19:43 | 14.8 |
| **radiance total** | | | | | **30.3 of 45** |
| spark1 vLLM | CUDA no-op (`window.sh`; `/v1/models` on 100.120.88.104:8000 answered afterwards) | 2712 | 06:20:40 | 06:32:11 | 11.5 (of 20) |

* CPU-only slots (radiance up): tatooine 2683, 2693, 2695 to 2698, 2699. spark1 with vLLM up: 2688 (build), 2711 and 2714
  (vLLM-up no-op).
* Radiance answered `/v1/models` after every window, and spark1's vLLM after its window. Every slot was released.

## 13. Next phases (not done here)

P2' (the user's gate) and P1 (proposal sections 3 and 7) are not started.

## 14. Final HEAD

* Branch `perf/core-op-fusion-p2`, rebased on origin/main `8d1f16a5`, not pushed. Code HEAD `60847700`; the commit that adds
  this document sits on top of it.
* Commits, oldest first:
  1. the `mutate.py` occurrence index (the 4 round-6 mutants);
  2. core: the two `Op`s, call sites, oracle, admission, pins;
  3. HIP kernels, tests, `p2-*` mutants, `fusion-p2.sh`;
  4. P4a composed with phase 1 (model test), the P3 barrier, `mac-checks.sh`;
  5. P3 without spill;
  6. the obsolete `mw-*` mutants dropped;
  7. `bench speed --at-most`;
  8. the CUDA no-op tree names;
  9. the W1 log and the `plans` stage fix;
  10. P3 without contraction (the W2 fix);
  11. `fusion-noop-p2.sh`;
  12. docs.
* `kernels/hip`, `crates/intarsia-backend-hip`, `crates/intarsia-core` and `crates/intarsia-models/src` on HEAD are identical
  to the tree W3 and W4 measured (`git diff 9665bf35 HEAD` over those paths is empty; `9665bf35` is the pre-rebase commit
  whose kernels were rsynced for W3).
* Mac checks on HEAD `60847700` (`raw/logs/mac-checks-final.log`):
  - `cargo test --workspace`: 103 binaries, 494 passed, 0 failed (CUDA, Metal and HIP pin tests included).
  - `cargo check --workspace`, `cargo check -p intarsia-cli --features hip`, `cargo check -p intarsia-cli --features metal` and
    `cargo build -p intarsia-backend-metal` all exit 0.
  - All 134 `mutate.py` mutations apply.
