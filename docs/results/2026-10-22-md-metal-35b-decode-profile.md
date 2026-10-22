# Metal Qwen3.6-35B-A3B NVFP4: where the decode step goes (measurement only)

Lane md, 2026-10-22. Machine: endor (Apple M5 Max, 128 GB). Model: `Qwen3.6-35B-A3B-NVFP4` (FP8 W8A8 projections, NVFP4 experts and lm_head). Code: origin/main `62c759b7` built with `--features metal` (worktree `/opt/code/wt-md`, branch `lane-md/decode-profile`). No code changed.

All numbers below come from **exclusive** fleetq slots on endor (#4007, #4013, #4017); builds ran in non-exclusive slots (#4006, #4015). GPU clock was 1617 MHz and GPU utilisation 98-100 % in every run (macmon, `data*/**.mac`). Prompt: 973 tokens (`raw/2026-10-22-md-metal-35b-decode-profile/prompt-1k.txt`), 129 forced tokens, temperature 0, `--profile-every 0`, `--max-seq 8192`.

## Key findings

1. **Decode on main is 124 tok/s, not 103.** Served (`bench speed`, 1k prompt, 128 tokens, 5 reps) 124.0-124.6 tok/s; `intarsia generate` 123.9-125.0 (6 runs, median 124.2). The "about 103" figure is the state before PR #78 (lane m35/sp-attn, `d1189e93`, split-KV decode attention for head_dim 256 / 8 q heads per kv head). In the same slot, interleaved, base `d99cfa73` (the base lane m35 sp-gemv measured) gives 101.2-103.4 tok/s (generate, 6 runs, median 102.9) and 102.5-103.1 served. The whole difference is `attn_decode`: 1939 us per token at base, 267 us now (section 4). The sp-gemv premise ("decode is 9.7 ms") was true for its base; main's step is **8.05 ms**.
2. **The step is GPU-bound, not host-bound.** Per token: GPU busy 7.84 ms, GPU idle between steps 0.22-0.23 ms, idle inside a step 0.01 ms. CPU encode of all 764 dispatches takes 0.71-0.82 ms per step and is hidden behind the GPU (9 % duty). Served and `generate` have the same trace (served: busy 7840 us, idle 227 us), so the serving path adds nothing.
3. **The step is 55 % of the byte roofline.** 2.40 GB of weights, KV and recurrent state stream per token (4.28 ms at 560 GB/s). GPU busy time is 7.84 ms. The 3.55 ms gap splits into about 1.9 ms of under-roofline streaming in the GEMV / MoE kernels and 1.67 ms of small kernels that move almost no bytes (21 % of GPU time, 383 of 764 launches).
4. **No Metal comparator can load this model**, so there is no comparator tok/s (section 6).

## 1. Method

| Probe | How | Used for |
|---|---|---|
| plain | `intarsia generate ... --max-tokens 129 --ignore-eos`, engine-reported decode tok/s | throughput; decode is 128 steps after the prefill token |
| `cb` | `INTARSIA_METAL_PROFILE=cb:FILE` (default 128 dispatches per command buffer, 7 buffers per step) | GPU busy, GPU idle, host encode / commit times per command buffer |
| `ce1` | `INTARSIA_METAL_PROFILE=cb:FILE INTARSIA_METAL_COMMIT_EVERY=1` | per-launch GPU time in a continuously loaded stream (no host waits); same method as the cb-m45 profile |
| `kernel` | `INTARSIA_METAL_PROFILE=kernel:FILE` | cross-check only: every launch is committed and waited on, so each row carries Metal's per-command-buffer cost and the run decodes at 8.7 tok/s |

Decode steps are split at `argmax_f32` (kernel, ce1) or by walking back 764 dispatches per step (cb); the last 128 steps are used and medians over steps are reported. Roles are assigned by position: every layer has 19 launches (GDN layer: qkv, z, a/b pair, conv, delta rule, gated rmsnorm, silu_glu, out, rmsnorm_add, then the MoE block; full-attention layer: q+gate, gate_split, k, v, qk_norm_rope_kv, attn_decode, sigmoid_mul, o, rmsnorm_add, MoE block). Metal profile labels carry no n / k / bytes for fp8 / nvfp4 matmuls on main, so shapes come from the launch position and the model config, and bytes from the config (fp8 1 B per weight, NVFP4 0.5625 B per weight, bf16 2 B; fp8 scale tensors ignored). The second dispatch of `moe_experts` (down projection plus weighted sum) shows up as an unlabelled `other` record in the traces.

`ce1` costs about 0.82 us per launch (764 launches: `ce1` attributed sum 8462 us minus `cb` GPU busy 7837 us); the table subtracts this per launch. In `ce1` consecutive independent launches overlap on the GPU (for example `gdn z` starts before `gdn qkv` ends), so the table merges overlapping neighbours into one row instead of crediting the overlap to one of them.

Scripts and raw data: `raw/2026-10-22-md-metal-35b-decode-profile/` (`run.sh`, `run2.sh`, `run3.sh`, `analyze.py`, `table.py`; `data/`, `data2/`, `data3/`; traces gzipped).

**Roofline.** 614 GB/s is the M5 Max memory spec (docs/results/2026-10-15-metal-longctx-profile.md). Practical peaks measured: `lm_head` streaming 286 MB at 587 GB/s in this run (table), the 65536-row fp8 microbench asymptote of 564 GB/s (docs/results/2026-10-22-m35-sp-gemv.md on `lane-m35/sp-gemv`), and 481 GB/s for the 8B GEMVs (cb-m45 profile). The table uses 560 GB/s; lm_head sits slightly above it, so the roofline is conservative by about 5 %.

## 2. Step breakdown (exclusive slots #4007 and #4017, 1k context, 128 decode steps)

Wall time per step (`generate`, plain): 8.05 ms = 124.2 tok/s.

| Quantity | us per token | Source |
|---|---:|---|
| GPU busy (sum of command-buffer GPU times, 7 buffers) | 7837 / 7825 | `cb` runs 1 and 2 |
| GPU idle between steps (last GPU end to next first GPU start) | 234 / 220 | `cb` runs 1 and 2 (served: 227) |
| GPU idle inside a step (between the 7 command buffers) | 8 / 5 | `cb` |
| Host encode of the step (summed over 7 buffers, overlapped with GPU) | 732 / 710 | `cb` (served: 820) |
| Host `commit` calls | 16 | `cb` |
| First encode to its GPU start (first 4-dispatch buffer) | 109 | `cb` |
| Dispatches per token | 764 | 40 layers x 19 + embedding, first norm, lm_head, argmax |

Why the 0.23 ms gap exists: the next step is encoded about 120 us after the previous one ends. Device-side token feedback (T1, on by default on Metal) would remove it, but `Model::feedback_ok` (`crates/intarsia-core/src/model/feedback.rs`) requires `!st.has_recurrent()`, and 30 of the 40 layers are GatedDeltaNet, so the 35B never takes the feedback path.

### Per-phase table (`ce1` per-launch GPU time, launch cost subtracted, % of GPU busy 7837 us)

Roofline 560 GB/s. "Excess" is measured minus bytes / 560 GB/s.

| Phase | Launches / token | us / token | % of step GPU | MB / token | Achieved GB/s | At roofline us | Excess us |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8 GEMV gdn qkv 8192x2048 + z 4096x2048 | 60 | 1743 | 22.2 % | 755 | 433 | 1348 | 395 |
| MoE experts gate/up (8 of 256, NVFP4) | 40 | 926 | 11.8 % | 378 | 408 | 674 | 252 |
| fp8 GEMV gdn out 2048x4096 | 30 | 855 | 10.9 % | 252 | 294 | 449 | 405 |
| MoE experts down + weighted sum | 40 | 655 | 8.4 % | 189 | 288 | 337 | 318 |
| Shared expert (gate, up, silu_glu, down; NVFP4) | 160 | 621 | 7.9 % | 71 | 114 | 126 | 495 |
| lm_head NVFP4 248320x2048 | 1 | 487 | 6.2 % | 286 | 587 | 511 | -23 |
| fp8 GEMV attn q+gate 8192x2048 + k, v 512x2048 | 30 | 450 | 5.7 % | 189 | 420 | 337 | 113 |
| rmsnorm / rmsnorm_add (all layers) | 81 | 352 | 4.5 % | 0 | - | 0 | 352 |
| MoE topk (256 experts, top 8) | 40 | 281 | 3.6 % | 0 | - | 0 | 281 |
| gdn delta_rule (f32 state read + write) | 30 | 276 | 3.5 % | 126 | 456 | 225 | 51 |
| fp8 GEMV attn o 2048x4096 | 10 | 271 | 3.5 % | 84 | 310 | 150 | 121 |
| attn_decode split-KV (1k context, KV bf16 assumed) | 10 | 259 | 3.3 % | 20 | 79 | 37 | 223 |
| MoE router + shared-gate pair (bf16 257x2048) | 40 | 162 | 2.1 % | 42 | 260 | 75 | 87 |
| gdn gated rmsnorm + silu_glu | 60 | 111 | 1.4 % | 0 | - | 0 | 111 |
| gdn conv | 30 | 103 | 1.3 % | 0 | - | 0 | 103 |
| attn qk_norm_rope_kv + gate_split + sigmoid_mul | 30 | 98 | 1.3 % | 0 | - | 0 | 98 |
| gdn a/b pair bf16 | 30 | 86 | 1.1 % | 8 | 92 | 14 | 72 |
| MoE combine | 40 | 80 | 1.0 % | 0 | - | 0 | 80 |
| embedding + argmax | 2 | 20 | 0.3 % | 0 | - | 0 | 20 |
| **Total** | **764** | **7837** | 100 % | **2399** | **306** | **4283** | **3554** |

Groupings:

- MoE router + topk + routed experts + combine + shared expert: 2725 us, **34.8 %** of GPU busy (router 162, topk 281, routed experts 1581, combine 80, shared expert 621). Decode has no separate gather launch: the routed `down` dispatch applies the weights in k order (the unlabelled second dispatch of `moe_experts`).
- fp8 projections (gdn qkv, z, out, attn q+gate, k, v, o): 3319 us, 42.3 %, 1.28 GB at 386 GB/s.
- Bandwidth-class kernels (all of the above with bytes): 6170 us, 78.7 %, 2.23 GB at 362 GB/s.
- Small kernels with no weight bytes (norms, topk, silu, combine, conv, gate_split, sigmoid_mul, qk_norm_rope_kv, embedding, argmax): 1667 us, 21.3 %, 383 launches.
- The kernel-mode (`kernel1`) per-role times agree for non-overlapping rows (qkv 44 us, out 29 us, lm_head 496 us, experts gate/up 24 us plus down 17 us per layer) and sum to 9205 us per step; the extra over 7837 us is Metal's per-command-buffer cost.
- `attn_decode` at 1k context is latency-bound (27 us per launch), but its cost grows with context; at 4k and above it is the only op whose bytes do.
- GPU idle between steps is 2.8 % of the 8.07 ms period.

## 3. Cross-checks

- Wall step 8.05 ms (124.2 tok/s) against cb-trace period 7837 + 227 = 8064 us: agrees to 0.2 %.
- `cb` mode and plain agree on speed (cb runs 119.5 and 124.0 tok/s, plain 120.3-124.0), so the profile does not perturb the stream. `ce1` slows decode to 113.9 tok/s (host encode becomes the limit with one buffer per dispatch), so its host timings are not used.
- Three arms in two slots give the same trace: `generate` (#4007), served (#4013 `servecb`) and a second `generate` (#4017). The served arm adds no GPU idle.

Per-run decode tok/s (exclusive slots):

| Slot | Run | Decode tok/s |
|---|---|---|
| #4007 | plain 1-3, 4, 5 | 123.8, 120.3, 124.0, 122.4, 123.3 |
| #4007 | cb 1, cb 2, ce1 | 119.5, 124.0, 113.9 |
| #4013 | generate A-D | 124.1, 124.1, 123.3, 121.5 |
| #4013 | served A (5 reps) | 120.2-122.9 |
| #4013 | served + cb profile (5 reps) | 122.3-124.0 |
| #4013 | served B (5 reps) | 120.8-122.9 |
| #4017 | main generate 1-6 | 124.4, 123.9, 124.1, 124.3, 125.0, 124.2 |
| #4017 | main served (5 reps) | 124.0-124.6 |

## 4. Why 103 became 124: main against base `d99cfa73` (slot #4017, interleaved)

| Arm | Generate decode tok/s (6 runs) | Served decode tok/s (5 reps) | Prefill tok/s (generate) |
|---|---|---|---|
| base `d99cfa73` | 101.2, 102.6, 103.4, 102.2, 103.1, 103.0 (median 102.9) | 102.5-103.1 | 261-267 (one 211) |
| main `62c759b7` | 124.4, 123.9, 124.1, 124.3, 125.0, 124.2 (median 124.2) | 124.0-124.6 | 811-930 |

`ce1` traces of both builds (`data3/baseE`, `data3/mainE`, per-role medians): the only role that moves is `attn_decode`, 1939 us to 267 us per token (-1672 us). Every other role is within 12 us except `shared silu_glu` (+124 us, an overlap-attribution artefact in `ce1` that the merged shared-expert row absorbs). That is PR #78 (`lane-m35/sp-attn`). The m35 MoE prefill kernels (#70, #90) are the prefill jump (about 260 to 900 tok/s on generate, served TTFT 3.4 s to 0.72 s).

So the m35 sp-gemv finding stands (the big GEMVs were not the problem). The step shrank from about 9.7 ms to 8.05 ms, and the 1.65 ms difference is the 1.67 ms `attn_decode` saving (194 us to 27 us per launch).

## 5. Levers, ranked by expected decode tok/s gain

Baseline 8.05 ms per step (124.2 tok/s). Gains are estimates from the table, not measurements; none was prototyped. "Bitwise" means the lever can keep every rounding point of the current kernels, so it can ship default-on with the usual gates (256 greedy ids, goldens, KL21 unchanged).

| # | Lever | Evidence | Expected gain | Numerics |
|---|---|---|---|---|
| 1 | Re-pack the fp8 projections to 4-bit weights (opt-in) | fp8 GEMVs are 3319 us (42 %) and stream 1.28 GB at 386 GB/s; at 4 bits per weight (NVFP4: 0.5625 B) they stream 0.72 GB, about 1.9 ms if the GEMV rate holds, i.e. -1.4 ms | about +20 % (+25 tok/s) | **Changes numerics**: the checkpoint keeps these layers in FP8 on purpose (ModelOpt mixed precision); needs a KL21 / golden gate and would be opt-in only |
| 2 | Cut launches: fuse the small kernels and fold elementwise work into neighbours | 383 launches, 1667 us (21 %) move almost no bytes; largest excess: rmsnorm / rmsnorm_add 352 us, topk 281 us, gdn conv + gated rmsnorm + silu 214 us, qk_norm_rope + gate_split + sigmoid 98 us, combine 80 us. HIP runs the same 35B decode in 375-415 plan launches after core-op fusion (docs/results/2026-10-19-core-op-fusion-p2.md); Metal registers none of those fused ops (`fused_ok` false) and issues 764 dispatches | -0.6 to -1.0 ms: **+8 to +12 %** | **Bitwise** if each fused kernel spells out the unfused arithmetic order (the HIP P2 precedent, with the one rounding slip it found and fixed) |
| 3 | MoE block efficiency: shared expert as one fused GLU kernel overlapped with the routed experts, and routed experts closer to the roofline | Shared expert is 160 launches, 621 us, 114 GB/s for 71 MB. Routed gate/up reaches 408 GB/s, routed down 288 GB/s (each at 8 x 0.6-1.2 MB runs picked by router ids) against 560. Excess 252 + 318 + 495 us | -0.5 to -0.8 ms: **+6 to +9 %** (overlaps lever 2 on `silu_glu`) | **Bitwise** if the reduction order inside each dot product and the k-order weighted sum stay; retiling that reorders sums would not be |
| 4 | Token feedback for recurrent (GatedDeltaNet) models | 0.22-0.23 ms idle per step (2.8 %), exactly the gap T1 removed on the 8B (0.263 to 0 ms); disabled here by `!st.has_recurrent()` in `Model::feedback_ok` | +0.1 to +0.23 ms: **+1.5 to +3 %** | **Bitwise.** A late stop must not leave a half-run step in the recurrent state: the held step must stop before the first GDN state write (launch index 5 of layer 0) instead of the default 16 dispatches, or the state needs a snapshot (126 MB) |

Not levers (measured):

- `lm_head` is at the roofline (587 GB/s).
- The fp8 GEMV kernel itself: sp-gemv already showed split-K is flat; the GEMVs sit at the per-launch ramp / tail cost, which is why launch count (levers 2 and 3), not bandwidth tuning, is the lever on these shapes.
- `attn_decode` at 1k context: 259 us (3.3 %), at most about 0.2 ms of excess. Revisit at long context.
- `gdn delta_rule` (456 GB/s on its f32 state): only lowering the state precision would help, which changes numerics.
- CPU encode and commit (hidden, 9 % duty) and serve-path overhead (none).

Not measured here and therefore not ranked: concurrent dispatch of independent launches (qkv with z, k with v, shared expert with routed experts). In `ce1` those pairs overlap on the GPU (`gdn z` costs 332 us instead of 823 us), but the serial encoder in normal operation does not, and `ce1` per-buffer cost hides the net effect. A prototype with explicit barriers would settle it; the 2026-10-14 estimate for the 8B was 0.1 to 0.2 ms net. Speculative decoding (verify M = 2..8 rows) also amortises the weight stream but is a separate measurement.

## 6. Comparator

None. mlx-lm cannot load the ModelOpt NVFP4 checkpoint, and endor has no MLX 4-bit or GGUF copy of Qwen3.6-35B-A3B (the HF cache entries `mlx-community/Qwen3.6-35B-A3B-4bit` and `Qwen/Qwen3.6-35B-A3B` hold no weights; `docs/results/2026-10-22-m35-metal-moe.md` §4 reaches the same conclusion). The only other Metal engine with this model on disk is Splash (`incoai/Qwen3.6-35B-A3B-Splash`, 20 GB): it is a different 4-bit quantisation (not the same weights), ships a speculative-decoding draft (DFlash, so decode tok/s is not comparable to greedy single-token decode), and `/opt/homebrew/bin/splash` is already the live LaunchAgent serving Qwen3.8-27B on :8000 (`endor-splash` target), which this lane does not stop. Not run. A like-for-like comparator needs a download of an MLX 4-bit or GGUF copy, which is the user's decision.

## 7. fleetq jobs

| Job | What | Slot |
|---|---|---|
| #4006 | build main (`--features metal`) | non-exclusive |
| #4007 | plain x5, cb x2, kernel, ce1 profile runs (`run.sh`) | **exclusive** |
| #4013 | generate vs served vs served + cb (`run2.sh`) | **exclusive** |
| #4015 | build base `d99cfa73` | non-exclusive |
| #4017 | main vs base A/B, `ce1` of both (`run3.sh`) | **exclusive** |

## 8. Caveats

- Byte counts are from the model config and ignore fp8 scale tensors (small) and any activation traffic; the KV read assumes a bf16 cache.
- Where `ce1` shows overlapping neighbours (gdn qkv with z, attn q+gate with k and v, the shared-expert chain), time is reported for the merged group, not per kernel; the GB/s of a merged row is a rate over the group.
- Single machine, one prompt (973 tokens, 1k context). Decode attention grows with context; this profile does not cover 4k and above.
- endor was under memory pressure during every run (macmon: RAM 104-112 GB used, swap 41.5 GB used) because standing services (`endor-kev`, `endor-tensorfold`) hold memory. Speed was stable at 120-125 tok/s across all slots, so no effect was visible, but it is a risk for runs near the memory budget.
- Lever gains are arithmetic on this table, not prototypes. Lever 1 assumes the GEMV rate does not fall when the bytes halve; sp-gemv's finding that these shapes sit at a per-launch floor suggests it will fall, so treat +20 % as an upper bound.
