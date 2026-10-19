# Core-op fusion P2' (`MoeBlock`): the whole MoE block after the router logits in one launch, 2026-10-19

Branch `perf/p2prime-moeblock` on origin/main `6ef5faaa` (P2' step 0 merged). Not pushed. HIP backend, AMD R9700 (gfx1201,
tatooine). Proposal: `docs/proposals/2026-10-19-core-op-fusion.md` section 5. Gate this work passed:
`docs/results/2026-10-19-p2prime-step0.md`. Raw data: `docs/results/2026-10-19-p2prime-moeblock/raw/`.

## Headline

* **Qwen3.6-35B-A3B NVFP4 decode on one R9700, with `INTARSIA_HIP_MOE_BLOCK=1`: 150.6 -> 164.9 tok/s at a 128-token prompt
  (+9.49 %).** At 1024 / 4096 / 16384 the gain is +9.38 / +9.06 / +8.76 %. Prefill is unchanged: -0.36 to +1.28 %, inside the
  round spread. Prefill chunks of 3+ tokens never take the op; a 1- or 2-token tail chunk does, bitwise. These are 3 interleaved rounds in one window (W5) with
  `bench speed --at-most` on both sides. In decode, every round of the opt-in is above every round of the default at every size.
* **120 fewer calls per 35B decode step: 375 -> 255** (`plan show --json`). Each of the 40 MoE layers replaces
  `moe_topk`, `moe_experts` (2 kernels), `matmul_nvfp4_glu` and `shared_down_combine` with one `moe_block`. That is 5 GPU
  kernels down to 1, so 160 fewer kernel launches (from the `moe.hip` source: both v2 decode branches launch gate_up +
  down_reduce, xstage defaults off; not a profiler count). Qwen3-8B: 364 -> 364.
* **Bitwise the default chain.** Evidence:
  - 44 on-device cases, including ties, E = 512, K = E and T * K = 16;
  - 200 back-to-back launches and graph replay;
  - 90 launches with scrambled ticket order;
  - 6000 sentinel-stress launches over both cross-block edges at T = 1, 2 and 4.

  The 35b-short and 35b-long KL21 digits equal the default's, digit for digit. The 256 greedy ids after the 4408-token prompt
  are identical. The prefix-reuse gate passes. The full admission sweep has 0 FAIL. The mutants:
  - all 11 non-equivalent mutants are killed (10 GPU, including the wrong ticket order, which deadlocks, plus 1 lib);
  - 3 survive, and the ISA diff shows each is equivalent on gfx1201 (section 5).
* **v3 alone (`INTARSIA_HIP_MOE_V3=1`), fresh A/B in the same window: +0.65 % at 128** (150.6 -> 151.6), +0.34 to +0.65 % at
  every size. The old +0.8 % no longer applies. P2' subsumes v3 on the shapes it takes.
* **M0:** a multi-token stress now kills `v3-noorder` at T = 2..8, including the 35B's T = 2 shape. The trick is to make
  token 0's last gate_up role late, not the grid's last role. Kill rates: 3/3 runs on each GPU, plus 2/2 at PRE = 1.
* **Nothing changes by default.** P2' and v3 both stay opt-in on HIP. CUDA and Metal register no `MoeBlock` kernel, and pin
  tests prove it. Making P2' (or v3) default is the user's decision (section 12).

## 1. Milestone status

| milestone | status | where |
|---|---|---|
| M0 T >= 2 stress variant | **done**, kills `v3-noorder` at T = 2..8 (own commit `1d0dafef`) | section 2 |
| M1 microbench gate | **passed**: projected +4.3 % (needed +1 %); measured end to end +9.5 % | section 3 |
| M2 P2' opt-in build | **done**: `Op::MoeBlock`, CPU oracle, admission cases, HIP kernel, `try_fused` site, env `INTARSIA_HIP_MOE_BLOCK=1`, off under TP > 1 | section 4 |
| M3 prove it | **done**: both edges at T = 1 / 2 / 4, 13 GPU mutants + 1 lib mutant, bitwise device tests, KL21 short/long, ids, prefix reuse, 8B goldens, full admission | sections 5, 6 |
| M4 measure | **done**: 35B A/B x3 (default / P2' / v3), 8B A/B x3, launches per step | section 7 |
| M5 CUDA no-op | see section 9 (spark1 plan JSON, pin tests); Metal checklist in section 10 | sections 9, 10 |

## 2. M0: the T >= 2 stress (`v3_sentinel_stress_multitoken`, `tests/moe_v3.rs`)

Step 0's narrow stress makes the last gate_up role of the whole grid late. At T >= 2 that role writes the last token's h, and
that token's down roles hold the last down tickets. They are mostly not resident yet when the store lands, so they cannot hold
stale lines. That is why step 0 never killed at T = 2.

The new default-off seam `INTARSIA_HIP_MOE_V3_SKEW_TOK=n` (bits 18..24 of the v3 skew word, `kernels/hip/moe.hip`) makes the
last gate_up role of token n late, or of every token (`=all`). Token 0's down roles take the first down tickets: they are
resident, pre-read, and spinning when the late store lands. Same poison, PREREAD and bitwise comparison as step 0. The shapes
are T = 2..8 at P = T * K <= 16, the shapes v3 serves.

| binary (W1, job 2720) | GPU 0 | GPU 1 |
|---|---|---|
| base | 0 / 3600, 0 / 3600 | - |
| `v3-noorder` | 2171, 2171, 2176 / 3600 (3/3 killed) | 2174, 2174, 2174 / 3600 (3/3 killed) |
| `v3-noorder`, `INTARSIA_HIP_MOE_PRE=1` | - | 2560, 2560 / 3600 (2/2 killed) |
| `v3-nofence` (equivalent, step 0) | - | 0 / 3600, 0 / 3600 |

Per configuration at the 35B T = 2 shape (e = 256, k = 8, h = 2048, i = 512), `v3-noorder`:
- token 0 late: 40 / 40 at skew 20 and 40 / 40 at skew 40;
- every token's last role late: 40 / 40 and 40 / 40;
- the old last-role skew (control): 0 / 120, as in step 0.

The other T = 2..8 shapes also die under the token-0 skew. Logs: `raw/logs/m0-w1*.log`.

## 3. M1: microbench gate (`moe_block_timing`, `tests/moe_block.rs`)

Setup:
- 16 MoE layers at the 35B decode shape (T = 1, E = 256, K = 8, H = 2048, I = 512, IS = 512);
- distinct weights, routing and activations per layer;
- each variant captured as one HIP graph;
- median of 50 replays, 3 interleaved rounds; us per layer.

Final kernel (W3, `raw/logs/m3-w3.log`):

| graph | round 1 | round 2 | round 3 |
|---|---|---|---|
| default chain: `moe_topk_v2`, `moe_experts_v2` (2 kernels), `nvfp4_glu`, `shared_down_combine` | 64.29 | 60.39 | 60.55 |
| v3 chain (`moe_experts_v3`: 1 kernel) | 59.21 | 59.53 | 59.56 |
| **`moe_block` (fused)** | **53.43** | **53.52** | **53.42** |
| one tiny launch (`moe_combine`, 1 block): the price of a launch boundary in a replay | 3.90 | 3.94 | 3.93 |
| `moe_topk_v2` alone | 6.66 | 6.71 | 6.70 |
| `moe_experts_v2` alone | 35.56 | 35.57 | 35.50 |
| `nvfp4_glu` alone | 6.75 | 6.78 | 6.74 |
| `shared_down_combine` alone | 14.73 | 14.86 | 14.66 |

The gate question: do the in-block top-k and shared-expert work grow the critical path enough to eat the saving?
- **"Free fusion" bound.** The chain is about 60.5 us. It has 4 launch boundaries (5 kernels), 4 x 3.9 = 15.6 us. So a fused
  kernel that added nothing would take about 45 us.
- **What the in-block work costs.** The fused kernel takes 53.45 us, so the recomputed top-k (every routed and down wave
  reselects its token's experts), the shared roles in the grid, and the one wait add about 8.5 us. The standalone top-k
  (6.7 us with its launch) and glu (6.75 us) are off the critical path.
- **Net saving vs the default chain: 7.05 us per layer.** With 40 MoE layers (plan), that is 282 us per step. At 150.9 tok/s
  (6627 us per step) it is +4.3 %, about 157 tok/s. That clears the +1 % gate, so M2 went ahead.
- **Net saving vs the v3 chain: 6.0 us per layer.**
- The first build (before the occupancy work, `df2bf50d`) measured 57.0 us fused. Three changes raised occupancy from 10 to 12
  waves/SIMD and cut 3.6 us:
  - scalar (readfirstlane) top-k ids and weights;
  - a 1-iteration prefetch of the shared-down row;
  - `amdgpu_waves_per_eu(12)`.

**End to end (section 7) the gain is +9.5 %, about twice the projection.** That is 576 us per step, or 14.4 us per MoE
layer, against 7.05 us in the microbench. This was not investigated. The likely causes are what a 16-layer MoE-only graph
leaves out: the chain's 5 grids drain and fill next to the attention / GDN kernels around them, and the chain's small
grids (top-k: 1 block; glu: 64 blocks) idle most of the device. The decode ids and all KL digits are identical, so the extra
speed is not less work done.

## 4. What was built

**Core (`crates/intarsia-core`)**
* `Op::MoeBlock` is appended to the `Op` enum (`"moe_block"`). It has 17 inputs:
  - `n`, router `logits`, shared-gate logit `sl`;
  - the 5 routed-expert tensors, in `MoeExperts` order;
  - the 3 NVFP4 parts of each of shared gate, up and down.

  One output `out` [T, H]; attrs `top_k`, `norm_topk`. `check` composes the checks of the ops it replaces.
* `ShapeInfo` facts: m = T, k = H, n = I, experts, top_k, and **`head_dim` = IS** (the shared intermediate; no new field, so
  CUDA plan JSON cannot change). `OpCall::with_rows` knows the op (batch-row probe, `check_batch_ops` trap). It is not in
  `PER_ROW_OPS`.
* `quant::ops::moe_block_scratch` (h [T * K, I] + sa [T, IS] bf16, each rounded up to 256 B), and `quant::cases`
  (`moe_block_case`, sweep cases at the 35B shape for T = 1 and 2, small and odd shapes, both normalisations).
* Call site, `MoeWeights::forward_folded` (`quant/moe.rs`):
  - Tried only when the router pair already produced the shared-gate logit and the shared expert is NVFP4 throughout.
  - `run.run_fused(Op::MoeBlock, ...)`; if it returns false, the old chain runs unchanged.
  - `fused_ok` refuses every fused op under TP > 1. A TP load of this model family is refused today anyway (GatedDeltaNet
    blocks are not sharded). `moe_block_is_off_under_tp2` pins both.

**CPU oracle** (`intarsia-backend-cpu/src/fused_ref.rs`): `MoeTopk` -> `MoeExperts` -> gate and up `MatmulNvfp4` -> `SiluGlu`
-> `SharedDownCombine`. Each intermediate is stored in the activation dtype, as the chain stores it.

**HIP kernel** (`kernels/hip/moe_block.hip`, `ik_moe_block_bf16`)
- **Grid.** One grid of `NS + NG + ND` blocks of 256 threads (8 waves). The role comes from a ticket (thread 0: atomic add on a
  per-stream counter), in dependency order:
  1. `NS = T * ceil(IS / 8)` shared glu roles (`nvfp4_glu_kernel`'s blocks), writing sa [T, IS];
  2. `NG = T * K * ceil(I / 8)` routed gate_up roles (v3's), writing h [T * K, I];
  3. `ND = T * ceil(H / 8)` down roles: v3's down + reduce, then as the row's epilogue `shared_down_combine_kernel<1>`'s
     shared-down row on sa and its combine.
- **Top-k is recomputed, so it has no edge.** Every gate_up wave reselects its token's experts up to the pair's rank. Every
  down wave reselects all K with the weights. Both use `moe_topk_v2_kernel`'s statements: the same key packing, tie order
  and softmax expression.
- **The only cross-block edges are h and sa.** Each producer stores, `__threadfence()`, barrier, then thread 0 does an agent
  release add on `sync[0]`. Each down role's thread 0 spins (acquire, agent) until `sync[0] >= NS + NG`. Then the barrier and
  a per-wave `__builtin_amdgcn_fence(acquire, "agent")` before any read of h or sa.
- **Counters.** The last down role resets all three counters. There is one 64-byte slot per (device, stream), 16 slots. A
  17th stream on one device makes the launch fail with `IK_ERR_UNSUPPORTED`. It does not fall back. Serve uses one stream.
- **Bitwise.** Every expression is the source kernel's, in the same order. That includes the routed row rounded to bf16
  before the combine, as the routed store rounds it. Section 6 has the evidence.
- **Shapes.** T * K <= 16, K <= 8, E <= 512, H / I / IS multiples of 16. These are the shapes the default chain serves with its
  decode kernels. At T = 3+ on the 35B the chain runs, as before.
- **Test seams, all default off, read per launch:**
  - `INTARSIA_HIP_MOE_BLOCK_SKEW` / `_SKEW_ALL` / `_SKEW_ROLE`: which producer is late;
  - `_PREREAD`: down waves read their h and sa lines before the wait;
  - `_SCRAMBLE`: blocks sleep `(blockIdx % 7) * SKEW` before drawing a ticket, which scrambles ticket order against
    dispatch order.

**Deadlock-freedom for any dispatch order.** A block draws its ticket only when it is already running. So whenever a down role
(ticket >= NS + NG) exists, every producer ticket is held by a block that is running or finished. Producers never wait. They
all finish, `sync[0]` reaches NS + NG, and every waiting down role proceeds. This holds for any order in which the hardware
dispatches or the blocks reach the atomic, because the role is a function of the draw, not of `blockIdx`. Residency does not
enter the argument: a spinning down role only occupies a slot that a producer would otherwise need if that producer had not
drawn yet, which is impossible, since all producer tickets are drawn before any down ticket. This is the rocPRIM / CUB
tile-ticket argument, as for v3.

Tested by `moe_block_dispatch_order`: 90 launches with scrambled draw order and late producers, bitwise, no hang. The inverse
is shown by mutant `p2p-ticket` (consumers first): it hangs at the 35B shape and is killed by the timeout.

**HIP registration** (`quantmoe.rs`): `moe_block_bf16`, priority 10, scratch `moe_block_scratch`.
- `when` = `moe_block_enabled() && moe_block_shape_ok(s)`.
- `moe_block_enabled()` is `INTARSIA_HIP_MOE_BLOCK` == `"1"`, read once.
- Off, `fused_ok` finds no kernel and the chain runs.
- `moe_block_is_opt_in_and_its_shapes_are_pinned` pins the env parse, the T limit (1, 2 yes; 3, 4096 no), K, E, IS and the
  scratch bytes.

## 5. Edges and mutants (W3, job 2730, final build; `raw/mut/*.log`, ISA in `raw/isa/`)

Two cross-block edges. Both are stressed at T = 1, T = 2 (the M0 shape, token 0's role late) and T = 4.
- **routed:** gate_up -> h -> down;
- **shared:** glu -> sa -> down/combine.

| config (`moe_block_sentinel_stress`) | late producer | shapes |
|---|---|---|
| routed last late | the last routed role | T = 1 x3 shapes, T = 2, T = 4 |
| shared last late | the last shared role | same |
| routed tok0 last late (+ side) | token 0's last routed role | same, with and without a stream-1 instance |
| shared tok0 last late (+ side) | token 0's last shared role | same |
| every producer late, side | all, 1..4 x skew | same |

Base: 6000 / 6000 launches bitwise (twice: W2 on the first build, W3 on the final one).

The mutant table below covers every mutant and its outcome. To summarize:
- **Killed (10 GPU + 1 lib):** `p2p-norefresh`, `p2p-noorder`, `p2p-ticket`, `p2p-wait-short`, `p2p-wait-routed`,
  `p2p-swap`, `p2p-noreset`, `p2p-topk-rank`, `p2p-sdc-noround`, `p2p-topk-wround`, and the lib mutant `p2p-sel-p24`.
- **Survived, each equivalent by ISA diff (3):** `p2p-gu-nofence`, `p2p-sh-nofence`, `p2p-nofence`.

| mutant | what | edge | result | stress launches that differ |
|---|---|---|---|---|
| `p2p-gu-nofence` | producer `__threadfence()` dropped, routed | routed | **survives: equivalent** (below) | 0 / 6000 |
| `p2p-sh-nofence` | producer `__threadfence()` dropped, shared | shared | **survives: equivalent** (below) | 0 / 6000 |
| `p2p-nofence` | consumer per-wave agent acquire fence dropped | both | **survives: equivalent** (as `v3-nofence`, step 0) | 0 / 6000 |
| `p2p-norefresh` | barrier without acquire and no fence: no per-wave refresh (step 0's `nobaracq`) | both | killed (sentinel stress) | 3161 / 6000 |
| `p2p-noorder` | all consumer ordering removed (relaxed spin, no barrier acquire, no fence) | both | killed (sentinel stress) | 3163 / 6000 |
| `p2p-ticket` | wrong ticket order: down roles take the first tickets | both | killed: deadlock at the 35B shape, SIGKILL after 90 s | - |
| `p2p-wait-short` | wait for NS + NG - 1 producers | both | killed (bitwise, graph, dispatch, stress) | 2644 / 6000 |
| `p2p-wait-routed` | wait for NG only (the shared edge unwaited) | shared | killed (same 4 tests) | 3398 / 6000 |
| `p2p-swap` | down reads h as the shared activation | shared | killed (5 tests) | 6000 / 6000 |
| `p2p-noreset` | counters not reset by the last down role | both | killed (4 tests) | 3400 / 6000 |
| `p2p-topk-rank` | gate_up takes the next rank's expert | routed | killed (5 tests) | 6000 / 6000 |
| `p2p-sdc-noround` | routed row not rounded to bf16 before the combine | - | killed (5 tests) | 6000 / 6000 |
| `p2p-topk-wround` | weights' bf16 rounding dropped in the recomputed top-k | - | killed (5 tests) | 6000 / 6000 |
| `p2p-sel-p24` (lib, Mac) | admitted at T * K <= 24 | selection | killed (`moe_block_is_opt_in_and_its_shapes_are_pinned`) | - |

Per-config kills (from `raw/mut/p2p-noorder.log` and `p2p-norefresh.log`):
- every configuration fails, at 40 / 40 skewed launches per skew value;
- that includes the T = 2 35B shape with token 0's routed role late and with token 0's shared role late;
- the side instance on stream 1 also differs, 80 / 120.

**Equivalence by ISA diff** (`raw/isa/*.k4.diff`, `moe_block_kernel<4, 8>`, gfx1201):
* **`p2p-gu-nofence` / `p2p-sh-nofence`.** Each diff removes exactly `global_inv scope:SCOPE_DEV; s_wait_loadcnt 0x0` from the
  producer's tail. That is the acquire half of the seq_cst `__threadfence()`. The release half stays as it was: the barrier's
  `s_wait_storecnt 0x0` before `s_barrier_signal`, then thread 0's `global_atomic_add_u32 ... scope:SCOPE_DEV`. A producer
  reads nothing after its store, so the removed invalidate has nothing to order. Stores on gfx12 complete at the device-coherent
  L2 (no `global_wb` is emitted at agent scope in either build). Same instructions on the store -> flag path, so equivalent.
* **`p2p-nofence`.** Removes `s_wait_loadcnt 0x0; global_inv scope:SCOPE_DEV` after the wait. The barrier's
  `global_inv scope:SCOPE_SE` is left, the same shape as `v3-nofence`, and step 0 showed it equivalent on this hardware. Removing
  that refresh too (`p2p-norefresh`) is killed, so the stress does see the per-wave refresh.

## 6. Accuracy and bitwise evidence

| check | default | `INTARSIA_HIP_MOE_BLOCK=1` |
|---|---|---|
| on-device bitwise vs the chain (`moe_block_is_bitwise_the_chain`): 44 cases, T = 1..4, ties, K = E, E = 512, H / IS past the windows, both normalisations | - | 44 / 44 bitwise |
| 200 back-to-back launches, two shapes, one scratch (`moe_block_counters_reset_between_launches`) | - | bitwise |
| graph replay x 20 with new activations (`moe_block_graph_replay_is_bitwise_eager`) | - | bitwise |
| scrambled ticket order, 90 launches (`moe_block_dispatch_order`) | - | bitwise, no hang |
| CPU model test: logits of a 12-token prefill + 6 decode steps, tiny MoE fixture, f32 and bf16 (`fusion_p2prime.rs`) | - | byte-identical to the P2a chain and to the unfused chain |
| 35b-short golden, KL21 prefill / decode | 7.417e-3 / 1.172e-2, PASS | 7.417e-3 / 1.172e-2, PASS |
| 35b-long golden (chunk 4096), KL21 prefill / decode | 4.356e-2 (floor FAIL, see below) / 9.273e-3 | 4.356e-2 / 9.273e-3 |
| prefix-reuse gate | PASS | PASS |
| 256 greedy ids after the 4408-token prompt | - | **identical** to the default (`IDS4K IDENTICAL`) |
| 8B fp32 goldens, short / long, full-vocab KL prefill / decode | 2.335e-4 / 3.626e-4, 3.966e-4 / 3.573e-4, PASS | the same digits, PASS |
| full admission sweep, 35B / 8B shapes (`kernels test`, no `--quick`) | 52 kernels, 899 / 912 cases, 0 FAIL | 52 kernels, 910 / 923 cases, 0 FAIL; `moe_block_bf16` 11 cases ADMITTED, max rel err 8.8e-4 vs the cpu oracle |

The 35b-long prefill floor FAIL (4.356e-2 against 3.0e-2) is pre-existing. The p2 doc reports the same digits for its base and
branch.

Prefill chunks of 3+ tokens do not take `MoeBlock`. A 1- or 2-token tail chunk does. At T = 2 the default chain is:
- `moe_topk_bf16_v2` and `moe_experts_bf16_v2`;
- gate and up `matmul_nvfp4_gemv_v2` (priority 20, m <= 8; `matmul_nvfp4_glu_gemv` takes m = 1 only), then `silu_glu_bf16`;
- `shared_down_combine_nvfp4_gemv` (m <= 8).

That is exactly the chain the device tests compare against at T = 2, 3 and 4. Those cases are bitwise.

Logs: `raw/logs/m3-w3*.log` (tests, ids, plans, admission), `raw/logs/m3-w4*.log` (accuracy), `raw/adm/`, `raw/plans/`.

## 7. A/B (W5 and W6, interleaved, `bench speed --at-most` on both sides; JSON in `raw/bench/`)

The arms are the same binary with different env in the serve container: default, `INTARSIA_HIP_MOE_BLOCK=1`, and
`INTARSIA_HIP_MOE_V3=1`. W5 order: def1 blk1 v3-1 blk2 v3-2 def2 v3-3 def3 blk3. Each cell is round 1 / 2 / 3 (median).

**35B, P2' vs default**

| prompt | default prefill | P2' prefill | delta | default decode | P2' decode | delta |
|---|---|---|---|---|---|---|
| 128 | 1752 / 1720 / 1778 (1750) | 1794 / 1790 / 1732 (1772) | +1.28 % | 150.2 / 150.8 / 150.8 (150.6) | 164.8 / 165.0 / 164.8 (164.9) | **+9.49 %** |
| 1024 | 6504 / 6415 / 6472 (6464) | 6457 / 6446 / 6417 (6440) | -0.36 % | 149.6 / 149.9 / 149.9 (149.8) | 163.8 / 164.1 / 163.7 (163.9) | **+9.38 %** |
| 4096 | 9073 / 8987 / 8964 (9008) | 9013 / 9011 / 8993 (9006) | -0.02 % | 144.2 / 144.3 / 144.3 (144.2) | 157.6 / 157.2 / 157.2 (157.3) | **+9.06 %** |
| 16384 | 7730 / 7671 / 7672 (7691) | 7684 / 7678 / 7674 (7679) | -0.16 % | 134.0 / 133.6 / 134.2 (133.9) | 145.5 / 146.0 / 145.6 (145.7) | **+8.76 %** |

**35B, v3 alone vs default** (fresh; the earlier +0.8 % predates phases 1 and 2)

| prompt | default prefill | v3 prefill | delta | default decode | v3 decode | delta |
|---|---|---|---|---|---|---|
| 128 | 1752 / 1720 / 1778 (1750) | 1753 / 1794 / 1799 (1782) | +1.85 % | 150.2 / 150.8 / 150.8 (150.6) | 151.5 / 151.6 / 151.6 (151.6) | +0.65 % |
| 1024 | 6504 / 6415 / 6472 (6464) | 6478 / 6463 / 6435 (6459) | -0.07 % | 149.6 / 149.9 / 149.9 (149.8) | 150.7 / 150.7 / 150.8 (150.7) | +0.57 % |
| 4096 | 9073 / 8987 / 8964 (9008) | 9016 / 8967 / 8977 (8986) | -0.24 % | 144.2 / 144.3 / 144.3 (144.2) | 145.6 / 144.8 / 145.1 (145.2) | +0.65 % |
| 16384 | 7730 / 7671 / 7672 (7691) | 7683 / 7688 / 7672 (7681) | -0.13 % | 134.0 / 133.6 / 134.2 (133.9) | 133.8 / 134.4 / 134.9 (134.4) | +0.34 % |

Prefill: the 128-token prefill moves +1.3 / +1.85 % on both opt-ins, but the default's own rounds spread 1720..1778 (3.4 %). It
is noise. Prefill chunks of 3+ tokens take neither op (T * K > 16 at K = 8).

**8B, P2' vs default** (W6; no MoE layer, so the op is never tried; expected 0)

| prompt | default prefill | P2' prefill | delta | default decode | P2' decode | delta |
|---|---|---|---|---|---|---|
| 1024 | 4503 / 4565 / 4498 (4522) | 4455 / 4510 / 4412 (4459) | -1.38 % | 38.4 / 38.4 / 38.4 (38.4) | 38.4 / 38.4 / 38.4 (38.4) | -0.10 % |
| 4160 | 4558 / 4564 / 4560 (4561) | 4550 / 4562 / 4562 (4558) | -0.07 % | 37.1 / 37.0 / 37.0 (37.0) | 37.0 / 37.0 / 37.0 (37.0) | -0.07 % |
| 16384 | 3429 / 3420 / 3430 (3426) | 3425 / 3420 / 3424 (3423) | -0.11 % | 33.2 / 33.2 / 33.3 (33.2) | 33.2 / 33.2 / 33.3 (33.3) | +0.03 % |

The 8B is unchanged. The 1024 prefill -1.38 % is inside the rounds' own spread (4412..4565). W6 order: def1 blk1 blk2 def2 def3 blk3.

**Launches per decode step** (`plan show --json`, `raw/plans/diff.txt`):

| model | default | P2' | change |
|---|---|---|---|
| 35B: plan calls | 375 | 255 | -120 (40 x 3) |
| 35B: GPU kernels (experts v2 is 2 kernels at decode) | 415 | 255 | -160 |
| 35B: scratch | 167905792 B | 167905792 B | 0 |
| 8B | 364 | 364 | 0 |

## 8. Memory, scratch, LDS, registers

| | `moe_block_kernel` (final) | for comparison |
|---|---|---|
| VGPRs | 120 (`amdgpu_waves_per_eu(12)`; 123 to 136 without the hint) | `raw/isa/resource-usage.txt` |
| scratch (spills) | 0 bytes per lane | at waves_per_eu 16: 76 B per lane of spills, so not used |
| occupancy | 12 waves per SIMD | 10 before the occupancy work |
| LDS | 68 B per block (fp4 table 64 B + ticket 4 B) | - |
| SGPRs | 99 (E <= 256 instance), 107 (E <= 512) | v3 `<4>`: 76 SGPR, 107 VGPR, occupancy 12 |
| op scratch | h T*K*I + sa T*IS bf16, each rounded to 256 B: 9216 B at the 35B T = 1, 18432 B at T = 2 | inside the plan's existing workspace (unchanged total) |
| device globals | `g_moe_block_sync`: 16 slots x 64 B = 1 KiB | v3's `g_moe_sync` is separate |

## 9. CUDA no-op (spark1)

CUDA_SECTION

## 10. Metal no-op check to run (Metal spike session, endor)

Base origin/main `6ef5faaa` vs this branch's HEAD.

**Getting the branch:** `perf/p2prime-moeblock` is local and unpushed. Either someone pushes it after review, or the Metal
session takes a bundle:
- on this Mac: `git bundle create p2prime-moeblock.bundle 6ef5faaa..perf/p2prime-moeblock`;
- on endor: `git fetch p2prime-moeblock.bundle perf/p2prime-moeblock`.

1. `cargo test --workspace` on endor. It includes:
   - `intarsia-models` `fusion_p2prime` and `batch_metal`;
   - the Metal `moe_block_is_absent_and_the_unfused_selection_is_pinned`;
   - the phase-1/2 Metal pins.
2. `intarsia plan show --backend metal --model <Qwen3-8B> --json` on both trees. The JSON must be byte-identical. This is the
   `check_batch_ops` / `PER_ROW_OPS` trap check: `MoeBlock` is in `with_rows`, not in `PER_ROW_OPS`. The 35B NVFP4
   checkpoint does not load on Metal before or after (no GatedDeltaNet, fp8, nvfp4 or MoE kernels).
3. `intarsia kernels test --backend metal --model <Qwen3-8B>`: kernel and case counts equal on both trees.
4. 8B greedy ids (256 tokens) and the two 8B fp32 goldens (`acc8`) on both trees: identical ids, identical KL digits.
5. With `--max-batch 8` (batched decode): the batch identity test. `cargo test -p intarsia-models --test batch_metal` covers it.

## 11. Every shared file touched, and what changes per backend

| file | CPU | HIP | CUDA | Metal |
|---|---|---|---|---|
| `crates/intarsia-core/src/ops.rs` (appended `Op::MoeBlock`, name, arity, check, shape facts with IS in `head_dim`, `with_rows`, test) | op known | op known | known, unregistered | known, unregistered |
| `crates/intarsia-core/src/quant/ops.rs` (check, `moe_block_scratch`) | used by the oracle | scratch + predicate facts | unused | unused |
| `crates/intarsia-core/src/quant/cases.rs` (`moe_block_case`, sweep and plan-time cases) | cases run on the oracle | cases run | unregistered op: no case runs | same |
| `crates/intarsia-core/src/quant/moe.rs` (`try_fused(MoeBlock)` before `MoeTopk`) | fold runs (oracle) | fold runs only with the env | `fused_ok` false: the old chain | same (no MoE model loads) |
| `crates/intarsia-backend-cpu/src/{fast,reference,fused_ref}.rs` | oracle | - | - | - |
| `crates/intarsia-backend-cpu/tests/quant_hf.rs` | counts updated | - | - | - |
| `crates/intarsia-backend-cuda/src/kernels.rs` (pin test) | - | - | test only | - |
| `crates/intarsia-backend-metal/src/kernels.rs` (pin test) | - | - | - | test only |
| `crates/intarsia-models/tests/fusion_p2prime.rs` | tests | - | - | - |
| HIP only: `kernels/hip/moe_block.hip`, `ik_hip_abi.h`, `kernels/hip/moe.hip` (M0 seam, default off), `crates/intarsia-backend-hip/src/{quantmoe,kernels,ffi,device}.rs`, `tests/{moe_block,moe_v3}.rs` | - | new opt-in kernel, test seam | - | - |
| scripts: `scripts/hip-decode/{mutate.py,p2prime-moeblock.sh,p2prime-step0.sh,remote.sh}` | - | measurement only | - | - |

## 12. Decisions for the user

1. **Make P2' default on HIP?** It is +8.8 to +9.5 % decode at every size and bitwise the default output. Every non-equivalent
   mutant is killed. The risk is the new cross-block wait, the same mechanism as v3. To make it default, flip
   `moe_block_from` to "on unless 0", and do it after one more stress window on a second day.
2. **v3 as default:** it is +0.34 to +0.65 % now. If P2' becomes default, v3 matters only at shapes P2' declines (none in
   the 35B's decode at T = 1, 2). Recommendation: leave v3 opt-in; consider retiring it once P2' is default.
3. **Push and Metal check:** push the branch, or hand the bundle to the endor session for section 10.

## 13. Caveats

* **The end-to-end gain is twice the microbench projection** (section 3). It was not root-caused. Both numbers favour the
  change, and the ids and KL digits rule out skipped work.
* **One day, one box, one A/B window per model.** The 3 interleaved rounds are tight (decode spread <= 0.4 %), but this is not
  a multi-day measurement.
* **Hardware facts are inferred from ISA plus stress, not from vendor documentation.** This covers the cache-scope equivalence
  arguments in section 5. Step 0 has the same caveat.
* **Stress is probabilistic.** Survivors are argued equivalent by ISA. Kills are 40 / 40 per skew value on the killing configs.
* **More than 16 streams** launching `moe_block` on one device fail with `IK_ERR_UNSUPPORTED` instead of falling back (same
  limit as v3's slots). Serve and the tests use at most 2.
* **Shapes.** At T * K > 16 (verify / spec rows >= 3 on the 35B, prefill) the chain runs. With batching on, `fused_ok`
  refuses the op when any row count up to `batch_rows` lacks a kernel. HIP has no batched step today.

## 14. Env vars

| var | default | effect |
|---|---|---|
| `INTARSIA_HIP_MOE_BLOCK=1` | off | admits `moe_block_bf16` (P2') |
| `INTARSIA_HIP_MOE_V3=1` | off | v3 one-launch experts (unchanged) |
| `INTARSIA_NO_FUSION=1` | off | every fused op off (existing kill switch). The generic `INTARSIA_HIP_DISABLE=moe_block_bf16` hides the kernel too; that mechanism exists but was not separately tested for this kernel |
| `INTARSIA_HIP_MOE_BLOCK_{SKEW,SKEW_ALL,SKEW_ROLE,PREREAD,SCRAMBLE}` | off | test seams (section 4) |
| `INTARSIA_HIP_MOE_BLOCK_BENCH=1`, `_BENCH_LAYERS`, `_STRESS_ROUNDS` | off / 16 / 120 | microbench and stress size |
| `INTARSIA_HIP_MOE_V3_SKEW_TOK=n\|all` | off | M0 seam: token n's (or every token's) last gate_up role late |

## 15. Downtime and slots

| window | job | radiance stopped | radiance answers | minutes | what |
|---|---|---|---|---|---|
| W1 | 2720 | 09:27:52 | 09:28:42 | 0.83 | M0: v3 multitoken stress, base / `v3-noorder` / `v3-nofence` |
| W2 | 2723 | 09:41:03 | 09:41:28 | 0.42 | M1: first build, bitwise + microbench + stress |
| W3 | 2730 | 09:59:47 | 10:04:23 | 4.60 | M3: final build tests + bench, plans, full admission, 13 GPU mutants, 4k ids |
| W4 | 2731 | 10:09:56 | 10:12:36 | 2.67 | M3: accuracy both arms (two GPUs) |
| W5 | 2732 | 10:18:01 | 10:25:25 | 7.40 | M4: 35B A/B x3, three arms |
| W6 | 2733 | 10:30:44 | 10:41:03 | 10.32 | M4: 8B A/B x3 |
| **total** | | | | **26.2 of 45** | every window < 20 min, >= 5 min of serving between windows, `/v1/models` answered before each release |

Per-window log: `raw/downtime.log`.

CPU-only slots (radiance up), all on tatooine: 2719 (M0 build), 2721 (compile check), 2722 (build), 2725 (resource usage,
mutant builds, ISA), 2729 (`topk-wround` build, ISA), 2734 (final resource usage). spark1 (vLLM up, never taken down): see section 9.
