# DeepSeek-V4.1 CUDA TP=2 prefill: attention + hyper-connection gap to TensorFold (lane attnfuse, lever 6)

Date 2026-11-03. Branch `dsv41/moe-other` from origin/main `47af18777` (after #284, #285, #286, #287). Raw logs and sources: `2026-11-03-dsv41-attn-fusion/` (`raw/`, `tools/`, `prompts/`; nothing over 1 MB except the three gzipped nsys traces).

## Result

**Negative: nothing lands.** Three bitwise-identity-intended changes aimed at this class gap (vector staging of the `HcSublayer` partial kernel with the prefill HcPost fold, a row-vector `rope_tail`, a register-prefetch `attn_sparse_mma`) each measured **zero or negative** in an isolated kernel bench on the real shapes, and the same in-situ at chunk 512. The kernel changes are reverted from the branch and kept as a patch (`raw/experiments-kernels.patch`, opt-outs `INTARSIA_DSV41_HC_VEC`, `_ROPE_VEC`, `_ATTN_PF`, `_HC_FOLD_ROWS`). The branch carries only docs, tools, the bench and a small `run-full.sh` addition (`pflong`, `PF_WARM`, `PF_REPS`). Recommendation: **drop** these three, keep the profile below as the starting point.

Secondary question (why `combine_kernel` takes 4.5 ms in the real pass against 1.47 ms alone): **answered in part, and the answer is a setting, not a kernel.** In the current default window (`INTARSIA_TP_SPIN_MS=100`, `run-full.sh`'s default since 2026-11-02) `combine_kernel` takes **1.48 ms** per 2048-row launch, the standalone figure. With `INTARSIA_TP_SPIN_MS=0` (the engine's own default, "unset: off") the same launch takes **3.90 ms**, and the whole 4K prefill at chunk 2048 goes from 4.44 s to 4.69 s (+5.6 %). Details in the last section. The memory-contention guess of the moegate doc is not what the data says.

## 1. Current class gap (task: measure first)

Rank 0, `nsys` cuda_gpu_trace, median of 3 traced requests per prompt, binary `bin/intarsia.main` (clean build of main `47af18777`, job 8181; window job 8185 configs 0 and 1). Class method as in `2026-11-02-dsv41-prefill.md` (excl-time sweep), TF trace from job 7840 of that doc. Two fixes to the class rules, in `tools/class_table.py`: TensorFold's attention regex (`_finish$|_partial`) caught `_hc_finish` and `_hc_partial`, so TF's hyper-connection time was understated by 52 ms and its attention overstated by the same (the sum was right); and ours: `grouped_rows_kernel<(bool)1` no longer exists after the staged-A gate/up (#284), so our "MoE gate/up" now lands in "MoE other". Attention plus hyper-connection is unaffected by both.

4106-token prompt, ms of GPU time:

| class | ours chunk 512 | ours chunk 2048 | TensorFold chunk 2048 |
|---|---|---|---|
| attention (sparse, rope, index, kv) | 501.8 | 520.0 | 149.5 |
| hyper-connection | 428.9 | 389.6 | 197.7 |
| **attention + HC** | **930.7** | **909.6** | **347.2** |
| GPU busy (all classes) | 5863 | 4387 | 2071 |

Gap to TF: **583 ms at chunk 512 (9.7 % of the 6.0 s wall, 10.0 % of busy), 562 ms at chunk 2048 (12.8 % of busy)**. The pre-#284 doc had 570 ms; the PRs since did not move this class. The doc's estimate for lever 6 (70 % of the gap, about 0.4 s) therefore stands as a ceiling, not as a result.

Per kernel (4K prompt, ms; ours chunk 512 / chunk 2048 / TF):

| kernel | ours 512 | ours 2048 | TF equivalent | TF |
|---|---|---|---|---|
| `attn_sparse_mma` (360 / 120 calls) | 389.2 | 390.1 | `_sparse_attn_part` + merge | 115.5 |
| `hc_sub_partial` | 221.2 | 163.5 | `_hc_mix_pf` (also folds HcPost) | 139.5 |
| `hc_post_kernel` (646 / 166 calls) | 149.4 | 142.7 | `_hc_post` (15 calls) + `_hc_partial` | 4.9 + 7.3 |
| `rope_tail_kernel` | 87.1 | 103.3 | `_rope_heads` + `_kv_norm_rope` | 16.0 |
| `hc_sub_finish` | 53.4 | 78.2 | `_hc_finish` | 45.0 |
| index score / select / qdq / window | about 25 | about 28 | | about 15 |

Attention kernel alone is half of the gap (275 ms), HC 197 ms, rope 71 ms. TF folds HcPost into its HC mix kernel at prefill (15 `_hc_post` calls against our 646); our design notes say the fold is slower at 4096 rows.

## 2. What I tried and measured

All three keep the old kernel selectable and were written to be bitwise identical to it.

1. **`hc_sub_partial_vec`**: 8-byte staging loads, next chunk's loads issued ahead of the dots, and the prefill fold (HcPost inside the partial kernel, vectorised, `HC_FOLD_ROWS` knob in `graph.rs`). Same sums in the same order.
2. **`rope_tail_rows`**: one block per row, sin/cos computed once per frequency and shared by the heads, 16-byte vectors.
3. **`attn_sparse_mma<PF>`**: the next 32-key tile is gathered into registers while the current tile runs QK, softmax and PV.

### Kernel bench (real shapes, `raw/bench/hc_rope_bench.cu`, the repo's own objects, spark2 next to vLLM, best of 5 runs of 20)

| kernel / shape | old | new | bits |
|---|---|---|---|
| `AttnSparse` CSA, T=512, Nc 1024, Kw 512 | 1.305 ms | **1.599 ms (+22 %)** | identical |
| `AttnSparse` CSA, T=2048 | 5.041 ms | **6.197 ms (+23 %)** | identical |
| `AttnSparse` HCA, T=512, Nc 32 | 0.371 ms | 0.406 ms | identical |
| HC sublayer unfolded op, T=512 / 2048 (vec staging) | 0.418 / 1.436 ms | 0.417 / 1.448 ms | identical |
| HcPost + unfolded op (today's prefill), T=512 / 2048 | 0.636 / 2.309 ms | 0.633 / 2.327 ms | identical |
| HC folded op (post inside the partial), T=512 / 2048 | 0.706 / 2.527 ms (scalar staging) | 0.890 / 2.846 ms (vector staging) | identical to HcPost + unfolded, all 8 shapes incl. S = 1..3, D = 5000 |
| `rope_tail` q, T=512 / 2048, H=32, hd=512 | 0.135 / 0.572 ms (248 / 235 GB/s) | 0.145 / 0.623 ms | **not identical** (hash differs) |
| `rope_tail` index_q T=512 H=64 hd=128 | 0.049 ms | 0.013 ms | not identical |

The old attention bench time matches the in-situ trace (1.09 ms mean per call over CSA and HCA layers at 512 rows), as the calibration rule of the moegate lane asks. `cuobjdump -res-usage`: PF 123 registers, old 107, no spills; 73 KB of shared memory, so one 8-warp block per SM either way.

Reading, with the assumption that failed named each time:
- **Rope**: the old kernel already moves data at 235 to 250 GB/s in isolation, the DRAM stream rate of this box. The assumption "rope is slow because of per-element `sincosf` and scalar loads" is wrong; the 87 to 103 ms in the trace is 1251 / 417 small calls at the memory floor, and only fusing the rotation into the neighbouring kernels (never writing `q_rope` / `o_derot`) would remove traffic. The new kernel is also not bit-identical: nvcc contracted `re * cs - im * sn` differently in the new kernel (index_q and the q/kv shapes differ). Rewriting the arithmetic with explicit `__fmaf_rn` would fix the bits, not the speed. Not pursued.
- **HC**: vector staging changes nothing (0.417 vs 0.418 ms), so staging is not what bounds `hc_sub_partial`; the fold, now 5 times cheaper per element to stage, is still slower than HcPost plus unfolded (0.89 vs 0.63 ms at 512 rows), because it reads every x row four times (once per stream k, from L2) where HcPost reads it once. The assumption "the fold loses on instruction count in the staging loop" is wrong. A fold that wins needs the four k-chunks of a d-range in one block, which changes the accumulation order of the dots, i.e. the bits (opt-in numerics, golden and KL needed). TF's `_hc_mix_pf` is such a kernel.
- **Attention**: prefetching the next tile makes it **slower**, with bit-identical output. The assumption "the kernel is exposed to global-memory latency at the gather" is wrong: per tile it is bound by the dependent chain after the gather (32 dependent QK `mma` per warp, four serial warp-reduction softmax rows, barriers), with only 8 warps per SM to hide it. `ncu` is not usable on these boxes (`ERR_NVGPUCTRPERM`, needs an admin setting I did not change), so I have no stall breakdown; a knockout bench for the phases is written (`raw/bench/knock.sh`, dry-run of its source patch checked locally) but **was not run**: its job (8192) waited behind other lanes' Spark windows and I cancelled it.

### In-situ A/B/A/B (window #8185, one binary, chunk 512, vLLM down and restored once; `raw/window/ab-walls.txt`)

Wall seconds of a `max_tokens 1` request with a unique first line, 3 reps each (spread = min to max). "old" = `HC_VEC=0 ROPE_VEC=0 ATTN_PF=0`, "new" = defaults of the experimental binary with `ROPE_VEC=0` (rope output differs), "fold" = new + `HC_FOLD_ROWS=100000`.

| config (order in window) | 1K prompt s | 4K prompt s | tok/s of the median 4K |
|---|---|---|---|
| old (ab3) | 1.573 1.637 1.717 | 6.055 6.074 5.969 | 678.1 |
| new (ab4) | 1.661 1.592 1.610 | 6.039 6.105 6.053 | 678.3 |
| old (ab5) | 1.586 1.583 1.647 | 6.091 5.959 5.965 | 688.3 |
| new (ab6) | 1.688 1.751 1.613 | 6.048 6.055 6.082 | 678.1 |
| fold (ab7) | 1.649 1.647 1.630 | 6.203 6.202 6.210 | 661.9 |

Old and new are the same within the spread (medians 6.055 / 5.965 against 6.053 / 6.055); fold is 2.5 % slower, in line with the bench. **No config is 3 % faster, so the stop rule applies.** End to end the new kernels are bitwise the old ones: `longp` (16 greedy tokens, top-5 logprobs) on the 4K and 8K prompts is bit-identical between ab3 (old), ab4 (new) and ab7 (fold) (`cmp_longp.py`: 16/16 equal, max abs diff 0).

### Gates

Nothing landed, so the landing gates are not applicable: golden 640/640, 32K/65K bit check, `plandiff.sh`, decode kinds and the Metal no-op were **not run**. For the record: `cargo test --workspace --no-fail-fast` on the experimental tree (endor, job 8190): 226 `test result: ok`, exit 0, no FAILED. The 32K and chunk-2048 cells of the requested A/B grid were not run either (the 4K/512 result was flat or negative, so a longer grid would not change the decision). Lowest MemAvailable in the window: 6.1 GiB (spark1) / 6.2 GiB (spark2), no memfine kill.

## 3. Combine: it is not the kernel and not (visibly) memory contention

Same binary (main `47af18777`), chunk 2048, the 4K and 1K prompt traced 3 times each, rank 0 (`raw/window/ab1-*` and `ab2-*`; job 8185 configs 1 and 2):

| config | `combine_kernel` per launch (median over all launches of the three 4K requests, including the 10-row tail chunk) | 4K request span (3 requests) | other kernels |
|---|---|---|---|
| `INTARSIA_TP_SPIN_MS=100` (ab1, as `run-full.sh` ships it) | **1.48 ms** | 4437 4439 4446 ms | |
| `INTARSIA_TP_SPIN_MS=0` (ab2) | **3.90 ms** | 4685 4681 4750 ms | |

Per-kernel excl time of the 4K request, spin 100 vs spin 0 (ms): `combine_kernel` 121 vs 322 (x2.66), `rot_in` 128 vs 147 (x1.15), `epilogue` 60 vs 66 (x1.09), everything else within 4 % (gate/up 1188 vs 1204, `attn_sparse_mma` 390 vs 390, `hc_post` 143 vs 148, NCCL all-reduce 489 vs 451). Total 4387 vs 4644 ms busy.

So: (a) the standalone 1.47 ms and the in-situ 1.48 ms agree, the kernel is at its floor; (b) the 3x of the earlier nsys traces (the 2026-11-02 traces average 3.07 ms per launch) comes from running without the rank-1 spin, which the earlier lanes did; (c) only combine, `rot_in` and `epilogue` care, not the weight-streaming GEMMs, so "memory-bound kernels slow down under contention" is not it (they would all slow). Common to the three affected kernels: small blocks that index per-expert scale vectors (`svh`, `suh`, `sud`) by routed expert id. A guess, untested: those vectors sit in memory the GPU reaches through a CPU-coherent path whose speed depends on the host's power state; the spin keeps the host busy. Not tested: spin on rank 0 only, a CPU burner with spin 0, where `svh_d` is allocated. One pair of configs at one chunk size, so the effect size (up to 5.6 % of a 4K prefill, the doc's "up to 5 % at chunk 2048") is one measurement. The engine default for `INTARSIA_TP_SPIN_MS` is off ("unset: off" in `docs/tuning.md`) and the setting costs two busy cores on spark2 during decode (the user's call of 2026-11-02), so this is a decision for the coordinator, not something I changed.

## Not checked

- Why `attn_sparse_mma` is 3.4x TF's attention kernel (275 ms of the 583). The knockout bench (`raw/bench/knock.sh <repo src>`, builds six variants of the kernel with the key gather, the QK mma, the softmax or the PV mma removed, about 5 minutes on a Spark next to vLLM) was queued but not run. No stall breakdown (`ncu` denied).
- A bitwise-identical rope fix (explicit `__fmaf_rn` in both kernels) and a rope fused into `attn_sparse_mma` (q rotation on load, inverse rotation of `o` in the epilogue; the epilogue thread holds each (even, odd) pair): not built.
- A reduced-bit-exactness HC fold (all four k-chunks of a d-range per block, TF's `_hc_mix_pf` shape): changes the dot accumulation order, so it would be opt-in numerics needing golden and KL; not built.
- The combine mechanism (host power state vs coherent access) and whether the spin should be the default for prefill; chunk 512; 1K prompts (the 1K wall was 3 % *faster* with spin 0 at chunk 2048, 1.362 vs 1.404 s, one run of 3).
- 32K and chunk-2048 timing cells, golden, plandiff, Metal no-op (nothing landed).

## Jobs and commands

| what | fleetq job |
|---|---|
| baseline build of main `47af18777` into `~/intarsia-dsv41-attnfuse` on spark2 | 8181 |
| `cargo test --workspace --no-fail-fast` (endor, experimental tree) | 8190 |
| window A: kernel bench, branch build, one `run-full.sh` engine run (14 min), vLLM restored 17:27 | 8185 (spark1+spark2) |
| knockout bench of `attn_sparse_mma` (spark2, vLLM up): queued, cancelled unrun | 8192 |
| remove `src/target` on spark2 (`~/intarsia-dsv41-attnfuse` keeps `bin/`, `bench/`, `logs/`, sources) | 8193 |

Commands: bench `raw/bench/build-run.sh <src>` (compiles `dsv41_misc/hc/hc_sub/attn.cu` with the `build.rs` flags, runs the bench with each opt-out, diffs the hashes; needs the experimental kernels from `raw/experiments-kernels.patch` applied); engine window `raw/window/step3-engine.sh` (the `AB` list is in `raw/window/ab-list.txt`), `NSYS=1 STEPS=pfnsys` for the traces; tables with `tools/class_table.py` (`ours512=… ours2048=… tf=… --prompts 1034,4106`), `tools/kcls.py` (per kernel), `tools/abtable.py` (A/B walls). Spark directory `~/intarsia-dsv41-attnfuse`: build trees removed at the end, `bin/` and `bench/` kept.
