# G4X: Gemma-4-26B-A4B-NVFP4 decode on CUDA, spark2 (2026-10-27 night)

Lane G4X, branch `lane-g4x/gemma4-decode` (PR with the CUDA graph and the head_dim 512 split-KV decode attention), fleetq jobs on spark2
(`--quiet gpu,cpu` for every timing run, `--load cpu` for builds, next to `spark2-vllm`, 38 GB, never touched). Builds: origin/main `1f8764f92`
(base, binary `base2`), the graph commit alone (`g1b`), this branch (`g2b`). Times are EDT. Continues
`2026-10-27-g4d-gemma4-drill.md` (G4D: 34.5 tok/s decode at 136 tokens, 13.6 at 16.7k, "decode runs eagerly, no CUDA graph, at about half the byte floor").

## 1. Result

| Gemma-4-26B-A4B-NVFP4, spark2, localeval `speed`, gen 128, thinking off, median of @@ROUNDS@@ interleaved rounds | base (main) | graph only (`g1b`) | this branch (`g2b`) |
|---|---|---|---|
@@SPEEDTABLE@@

| gate | result |
|---|---|
| decode, median over the four sizes, at least +10% | @@DECODEGATE@@ |
| prefill at least 0.99x of base at every size | @@PREFILLGATE@@ |
| CPU tests (`cargo test --workspace`, endor) and the docs/env tests | pass (section 6) |
| `plan show` G0, Qwen3-8B and Qwen3.6-35B, CUDA (spark2) and Metal (endor), text and `--json` | byte-identical to main for both builds (8758 / 15095 B JSON on CUDA) |
| greedy ids vs base, 7 prompts (3 short, 1k, 4k, 8k code, 16k) | graph only: **7 of 7 identical**; this branch: @@IDSGATE@@ (a numerics-changing kernel, section 5) |
| accuracy vs the committed f32 golden (`gemma4-26b-a4b-nvfp4-fp32`, 10 prompts, prefill and decode) | prefill KL 2.384e-1 = base; decode KL **1.660e-1 vs 2.024e-1 base** (better); floor PASS (limit 2.598e-1) |
| MMLU, 5-shot, n = 1425, thinking off | @@MMLU@@ |
| kernel admission + planted mutants (new D = 512 split-KV kernel) | admission 51 cases ADMITTED; **4 of 4 mutants killed**, section 4 |
| rollback knob | `INTARSIA_CUDA_SPLITKV_512=0` (D = 512 back to the SIMT kernels); `--no-graphs` for the graph |
| `max_batch` | not raised (section 7) |

## 2. Why Gemma-4 decoded at half of what Qwen3.6-35B does

nsys (`scripts/g4x/profile.sh`, `--cuda-graph-trace=node`, 64 greedy tokens after a 1.2k and a 19.8k-token prompt), per decode step:

| per decode step | eager base, 19.8k prompt | this branch, 1.2k prompt | this branch, 19.8k prompt |
|---|---|---|---|
| span | 81.7 ms (12.2 tok/s) | 26.9 ms (37.1 tok/s) | 30.1 ms (33.2 tok/s) |
| bf16 dense gemv (235 launches: q/k/v/o, dense MLP, router) | 14.6 ms | 14.4 ms | 14.3 ms |
| `lm_head` gemv (f32 logits) | 6.1 ms | 5.8 ms | 5.8 ms |
| MoE decode kernels (30 layers) | 4.2 ms | 4.1 ms | 4.1 ms |
| sliding-window attention (25 layers, split-KV) | 1.1 ms | 0.9 ms | 1.1 ms |
| global attention (5 layers, head_dim 512) | SIMT `attn_kernel`: **52.5 ms** (10.5 ms per launch, 65.7% of the step) | split-KV 0.15 ms | split-KV **3.3 ms** (0.67 ms per launch) |
| host gaps inside the step | 2.6% | 0.4% | 0.3% |

* **Closed cause 1, long context: head_dim 512 attention ran the one-block-per-q-head SIMT kernel.** The global layers (16 q heads over 2 kv heads,
  head_dim 512) had no split-KV kernel (`splitkv_ok` stopped at head_dim 256), so each of the 5 global layers launched 16 blocks that walked the whole
  context alone: 10.5 ms per layer at 20k tokens, 66% of a step. That is the whole fall from 34.9 to 13.7 tok/s in the base column. A
  split-KV kernel for head_dim 512 (`splitkv512_kernel`, below) takes it to 0.67 ms per layer.
* **Closed cause 2, short context: decode ran eagerly.** `GraphState::new(opts.graphs && dims.gemma.is_none())` and a
  `captured decode steps of a Gemma 4 layer` error kept Gemma off the CUDA graph: 964 launches per step with 2.6% host gaps. The Gemma layer now
  takes the device-side position (`dpos`) through rope, kv append and attention like the other architectures, and the soft-capped logits are
  capped on the host after the graph's logits download. Graph only: +5.8% median (+3.0% at 16k, +7.2% at 128).
* **Not closable here: the bytes.** The checkpoint quantises only the experts. Attention, the dense MLP, the router and the tied `lm_head` stay bf16:
  per token 3.31 GB dense bf16 + 1.48 GB `lm_head` + 0.80 GB of 8 active experts (fp4 + e4m3 scales) = **5.59 GB**, against about 1.6 GB for
  Qwen3.6-35B-A3B. At the measured 253 GB/s (the `lm_head` gemv reaches 253 GB/s, the 235 dense gemvs 229 GB/s) the floor is 22 ms, **45 tok/s**;
  this branch decodes at 37 (1k) and is at 80% of it. The wrong assumption in G4D was "similar active bytes as Qwen3.6-35B (84 tok/s)". The next lever
  is fewer bytes (quantise the dense bf16 weights at load, W8A16 or NVFP4: roughly +40% to +80% decode, a numerics change that needs its own gate),
  not kernels. At 16k the global-layer kernel still reads at only about 100 GB/s (335 MB in 3.3 ms); more splits would give back up to 2.5 ms of 30.

## 3. What changed

* `crates/intarsia-core/src/model/gemma.rs`, `model.rs`, `model/graph_step.rs`: `gemma_layer` takes `dpos` and calls `exec_pos` for rope, kv append
  and decode attention (attention rows of a prefill chunk keep the host position); `GraphState` is no longer disabled for Gemma;
  `softcap_logits` is applied to the graph's downloaded logits (the device argmax path is unaffected: cap is monotonic).
* `kernels/cuda/attention_splitkv.cu`: `splitkv512_kernel` + `launch_split512`. Block = 256 threads = 8 key lanes; one warp spans a 512-wide key row (two
  16 B loads per thread, dims `c*8` and `256 + c*8`), so a score is a 5-step shuffle reduction; the 8 q heads of a kv head run in 2 passes of 4
  (grid.y = kv heads x 2) with the second pass re-reading K/V from L2; the lane merge, window, split policy (`splitkv_count`/`chunk`), the partial layout
  and `combine_kernel<512>` are the existing ones, so the kernel takes the device position (`pos_dev`, graph capture) and the sliding window like the
  D <= 256 kernels. `crates/intarsia-backend-cuda/src/kernels.rs`: `splitkv_ext_ok` accepts head_dim 512 with at most 8 q heads per kv head;
  `INTARSIA_CUDA_SPLITKV_512=0` removes that case (rollback; row in `docs/tuning.md`).
* `scripts/g4x/`: `ab.sh`, `le.sh` (localeval rounds), `profile.sh`, `ids.sh`, `acc.sh`, `selfgold.sh`, `g0.sh`, `mutate.sh`/`mutate.py`, `summarize.py`;
  `scripts/cuda-decode/nsys_steps.py` cuts Gemma steps at the scaled embedding kernel.

## 4. Gates, in detail

**Speed** (`scripts/g4x/le.sh`, localeval `speed`, `--sizes 128,1024,4096,16384 --gen 128 --reps 2` plus one warmup, a fresh `intarsia serve` per
arm, one fleetq slot per round, arms in the order base, g1b, g2b each round):

@@SPEEDDETAIL@@

**Kernel admission and planted mutants** (`scripts/g4x/mutate.sh`, a copy of the tree with one planted bug, rebuilt, `intarsia kernels test --backend cuda
--op attn_decode --kernel attn_decode_bf16_splitkv_ext`; the unmutated tree is ADMITTED with 51 cases first): all four killed.

| mutant | planted bug | result |
|---|---|---|
| `sk512-pass` | the second q-head pass restarts at head 0 | KILLED (T=1 pos0=7 N=16 Nkv=2 D=512) |
| `sk512-hidim` | high 256 dims read from the wrong offset | KILLED (rel_l2 0.75) |
| `sk512-merge` | the lane merge writes only the first 256 dims | KILLED (rel_l2 0.71) |
| `sk512-window` | sliding window off by one key | KILLED (window=5, rel_l2 0.84) |

The admission sweep already held the Gemma shapes (16/2 x 512, windows 0 / 5 / 1024, prefix lengths 7 / 40 / 1100 / 1500); the registry unit test
`attention_selection_by_kv_dtype` pins the pick, including that more than 8 q heads per kv head stay on the SIMT kernel.

**Accuracy** (`scripts/g4x/acc.sh`, `selfgold.sh`; golden `gemma4-26b-a4b-nvfp4-fp32`, `--max-seq 8192 --prefill-chunk 1024`):

| build | prefill KL(21-way) | decode KL(21-way) | note |
|---|---|---|---|
| base | 2.384e-1 | 2.024e-1 | the limit is 2.598e-1 (ref noise 2.362e-1 x 1.1) |
| graph only | 2.384e-1 | 2.024e-1 | bit-identical logits path |
| this branch | 2.384e-1 | **1.660e-1** | prefill untouched; decode closer to the f32 golden |

Against the base build's own logits (`--self-golden`): prefill mean KL 1.3e-12 (identical), decode mean KL 6.78e-2 = 0.29 x the reference noise (limit
1.1), 10 of 160 teacher-forced positions flip their top-1 (a different rounding order in the head_dim 512 attention). One baseline-gate metric misses by a
hair: decode `|tail excess|` 1.788e-2 against a limit of 1.749e-2 (the baseline's own 1.550e-2 plus the 0.002 fixed margin); mean KL, top-1, top-5 and
max KL are all better than base. For scale: the model is numerically chaotic (HF's own bf16 forward is 0.236 KL away from its f32 forward), and the same
base build at prefill chunk 256 moves decode self-KL to 1.17e-1.

## 5. Why the greedy ids change, and why that is allowed here

The ids gate ("identical to the pre-change build") cannot hold for a new attention kernel: the split-KV kernel sums the keys in a different order than the
one-block SIMT kernel it replaces, so near-ties flip. Measured on the 7 raw prompts (`scripts/g4x/ids.sh`, 96 tokens, `--ignore-eos`): @@IDSDETAIL@@
The graph alone is bitwise (7 of 7). The coordinator decision of this night applies the numerics-changing gate path (the Metal NAX precedent, #193):
changed ids are allowed when decode KL against the f32 golden is inside or better than base (1.660e-1 vs 2.024e-1), MMLU holds (@@MMLUSHORT@@), the
other gates pass, and the path stays behind a rollback knob (`INTARSIA_CUDA_SPLITKV_512=0`).

## 6. Tests

* CPU: `cargo test --workspace --locked` on endor (all `test result: ok`), `cargo test -p intarsia-cli --test docs_env_vars` (the new knob has its row),
  `cargo test -p intarsia-backend-cuda --lib` (registry picks, 19 tests).
* GPU (spark2): `cargo test --release -p intarsia-backend-cuda --test admission --test attention --test quantmoe` all pass.
* `cargo fmt --check` reports differences in 2009 places across the workspace on main itself; the touched files were not reformatted.

## 7. Not done, and what was tried and not shipped

* **`max_batch` stays 1.** Batched decode and packed prefill exclude Gemma (`pack_capable` requires `gemma.is_none()`, `batch.rs` line 1028): the Gemma
  layer is solo-only (no row-table forms of rope_prop, the window attention, kv append or the gelu MoE). Raising it needs those kernels with admission
  and mutants, then the batched-equals-solo gate; it does not fit this lane.
* **Prefill, branch `lane-g4x/gemma4-prefill-ref` (pushed, not merged, no PR).** Prefill at 1k is 523 tok/s and 211 at 16k. The profile of a 19.8k prefill
  (105 s) is 74% SIMT attention (`attn_kernel`: 78 s, 62 s of it in the 5 global layers) and, at 1k, 21% attention + 20% bf16 gemv + the SIMT MoE kernels
  (`moe_gate_up_kernel`, `moe_down_kernel`) because the tensor-core grouped MoE kernels required I % 128 == 0 and Gemma has I = 704. The branch has
  (a) the grouped gelu MoE kernels accepting I = 64 mod 128 (a half raw unit at the end of the down projection), (b) a tensor-core flash prefill
  kernel with the sliding window for head_dim 256 / 512 (`attention_flash_ext.cu`), admission cases for both and a Gemma-shape quantmoe test. Every unit
  test and the admission sweep pass, but the accuracy gate does **not**: against the base build's logits the prefill self-KL is 0.26 with 14.8% top-1
  flips (limit 0.26), decode 0.39; each half alone is 0.21 to 0.30. Against the f32 golden the prefill KL is 2.586e-1 and the decode KL 3.563e-1 (limit
  2.598e-1; each half alone moves prefill KL 2.384e-1 to 2.455e-1 (flash) or 3.461e-1 (MoE tensor cores)). The kernels match the CPU reference on random
  and Gemma-shaped data, so the divergence is real-data numerics (bf16 P and the bf16 roundings of the tensor-core epilogue on Gemma's large activations),
  not a shape bug, but I could not show that within the lane and did not ship it. Speed of that branch was not measured (the run was cancelled for time).
* The same bisection is the reason the PR carries no prefill change: prefill stays at 1.00x.
* `INTARSIA_CUDA_W4A16` etc. are untouched; the w4a16 gelu kernel is not selected by default and so was not exercised by the model-level gates.

## 8. Method notes and hazards

* Everything from a clean `1f8764f92` build; `scripts/deploy.sh` syncs with `--delete`, so run output lives in `replay-out/` of the deployed tree.
* Memory: no HF f32 reference generation was run on spark2 (the committed goldens were used). Every serve ran with `--mem-budget 30` after a
  `MemAvailable >= 45 GB` check, next to `spark2-vllm`'s 38 GB.
* A first MMLU run of the base arm was killed by a SIGTERM from outside (exit 143) five minutes in; it was repeated in a fresh slot.
* The aborted first localeval round (base/g1 on `2859c1a89`-based binaries, labels `g4x-base-r1`, `g4x-g1-r1`) is logged in unanimis and should be ignored: the
  PR builds were rebuilt on the then-current main.
