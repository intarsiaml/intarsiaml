# NAB: new-architecture drill on LFM2.5, stopped at the premise check (2026-10-27 morning)

Lane NAB, branch `lane-nab/lfm2-block`, main `e7e0845ec`, endor (Metal). Clock: box clock (EDT), T0 = 07:38 (task start; weights were
already in the HF cache). Goal of the lane: add LFM2's short-conv block as a reusable block and run `onboard` to PASS on Metal, to test
the second half of requirement 1 ("a model that needs a new block: basic test in 1-2 h, in-depth in 2-4 h").

**Result: the drill did not start. No code was written.** The only LFM2.5 checkpoints on endor are three MLX quantisations of one 24B
MoE model, and the short-conv block is the smallest of the gaps between that checkpoint and a PASS.

## What is on disk

`LiquidAI/LFM2.5-24B-A2B-MLX-{4bit,6bit,8bit}` (13 / 18 / 24 GB), `Lfm2MoeForCausalLM`: 40 layers, hidden 2048, 30 `conv` layers and 10
`full_attention` layers (GQA 32/8 heads, qk layernorm), layers 0-1 with a dense FFN (`intermediate_size` 11776) and layers 2-39 with a
64-expert top-4 MoE (`moe_intermediate_size` 1536, no shared expert). The smallest is the 4-bit one (snapshot `667408d4`); no bf16 or
smaller (1.2B / 2.6B) LFM2 was found on endor (`find` for lfm2 / liquidai names to depth 6, plus the HF cache listing) and nothing may be downloaded.

## Stage timing

| Stage | Start | End | Wall | Note |
|---|---|---|---|---|
| Read docs, prior drill, checkpoint config and tensor index | 07:38 | 07:43 | 5 min | found MLX-only checkpoint, MoE |
| Advisor check of the premise, offline-reference search, HF router and conv code | 07:43 | 07:46 | 3 min | |
| `intarsia recipe new` on the 4-bit checkpoint (release build, fleetq #6210, 22 s) | 07:44 | 07:44 | under 1 s | **13 TODOs, exit 1**, raw output in `raw/2026-10-27-nab/` |
| Decision: stop, report (spec premise broken) | 07:46 | | | |

Targets: basic test 1-2 h, in-depth 2-4 h. **Neither was measured.** The honest reading is that for this checkpoint the targets are
not reachable by adding one block: see the list below.

## What stands between this checkpoint and `onboard` PASS, ranked by cost

`recipe new` names items 1, 2 (partly), 5, 6, 7 and 9; the rest come from reading `quant/moe.rs` and the HF `modeling_lfm2_moe.py`.

| # | Gap | Evidence | Size | One-line fix |
|---|---|---|---|---|
| 1 | **No MLX affine format.** Weights are `U32` packed 4-bit, group 64, with `.scales` and `.biases` (embedding 6-bit); `recipe new` prints "U32 weight without a quantisation note" for every linear. The 10-26 drill already lists this as "blocked: no MLX format" for MiniCPM-V, Nemotron-H and Kolibri; it blocks every MLX checkpoint on the disk. | `raw/.../recipe-new-lfm2-24b-mlx4.txt` items 10-13 | days if done as kernels; 1-2 h as a load-time dequant to bf16 | `quant/mlx_affine.rs`: dequantise to bf16 at load (`w = q*scale + bias`, group 64, 4 and 6 bit), kernels later; or `tools/mlx_to_bf16.py` for a bf16 copy (about 48 GB). |
| 2 | **Routed experts are NVFP4-only.** `quant/moe.rs` stacks per-expert NVFP4 tensors for `Op::MoeExperts`; there is no bf16 expert path, so even a dequantised copy would not load. | `quant/moe.rs` header and `load` (NVFP4 checks) | a new bf16 MoE-experts kernel on cpu-ref and Metal | bf16 `MoeExperts` (cpu-ref first, Metal grouped GEMM) or requantise to NVFP4 and take that numerics change. |
| 3 | **Router is sigmoid + `expert_bias`.** LFM2 selects top-k on `sigmoid(logit) + expert_bias`, weights are the unbiased sigmoids normalised by `sum + 1e-6`, times `routed_scaling_factor`; the engine's `Op::MoeTopk` is softmax then normalise. No shared expert (the MoE block assumes one for Qwen). | HF `Lfm2MoeTopKRouter`; router `gate.weight` and `expert_bias` are F32 | one router variant, cpu-ref + Metal | `scoring = "sigmoid"` and an `expert_bias` input on `MoeTopk`. |
| 4 | **Experts are stacked** (`switch_mlp.{gate,up,down}_proj` `[64, 1536, 256]`), the MoE loader reads `experts.{e}.*`. | tensor index | small | slice the stacked tensor in the loader (the MTP loader already does it for `gate_up_proj`). |
| 5 | **Dense-then-MoE layers** (`num_dense_layers = 2`) with a `feed_forward.*` namespace; the recipe maps FFN kind by layer *type*, not by layer index. | `recipe new` item 3 | small | a per-layer FFN kind (`layer.ffn_by_layer = "dense_below:num_dense_layers"`). |
| 6 | **The short-conv block** (the lane's actual goal). `in_proj` splits B, C, x; `y = out_proj(C * conv(B * x))`; depthwise causal conv with `conv_L_cache = 3`, **no activation** (GDN's `gdn_conv` applies SiLU, so it needs an activation-off variant); state is the last 2 columns of `B * x`. Checkpoint conv weight is `[2048, 3, 1]` (MLX layout) against PyTorch `[C, 1, K]`. | HF `Lfm2MoeShortConv` | the planned 2-3 h, not started | as in the brief. |
| 7 | **Config and naming mismatches** that `recipe new` reports but does not propose a fix for: `norm_eps` instead of `rms_norm_eps` (the `[config]` rename table exists, the scaffold does not use it), final norm is `model.embedding_norm` (not `model.norm`), per-layer norms `operator_norm` / `ffn_norm`, attention `q_layernorm` / `k_layernorm` / `out_proj`, `model_type` rope under `rope_parameters`. | `recipe new` items 1, 2, 9 | minutes each, by hand | have `recipe new` try the nearest-name candidate (`embedding_norm` for `norm`, `norm_eps` for `rms_norm_eps`) and print it as the TODO's suggestion. |
| 8 | **No golden can be made from this directory.** `transformers` with `lfm2_moe` is installed (`tools/.venv`, `models/lfm2_moe`), but HF cannot load MLX-quantised weights, so the auto-made short golden (PR #203) needs the dequantised bf16 copy from item 1. An fp32 reference of a 24B model is about 96 GB and does not fit next to `endor-kev` and `endor-tensorfold` (60 GB free); it has to be a bf16 reference, which the QGC calibration then sees as its own spread. | `fleetq status`, `tools/pyproject.toml` | 30-60 min of CPU golden for 4k, unmeasured | a bf16-reference mode in `tools/golden.py` for models over the memory budget. |
| 9 | f32 router and `expert_bias` against bf16 kernels: the same class as the NMD `A_log` gap. | `gate.weight` F32 | minutes | extend `gpu_f32_to_bf16` rounding to router tensors or keep f32 in the router op. |

Not checked: whether `qualify` makes the auto short golden for a MoE hybrid at all (PR #203 text says hybrid; not run), CUDA / HIP
refusal text for the missing kernels (nothing was added), Metal memory for 13 GB 4-bit weights as bf16 (about 48 GB, fits in 128 GB).

## Friction list (what makes adding a block or a new family slow), ranked

1. **Checkpoint format coverage is the real gate, not block count.** Four of the six families listed in the 10-26 drill, and all three
   LFM2.5 copies here, stop at the quantisation format. A load-time MLX-affine dequant (item 1) would unblock the cached MLX models
   (LFM2.5, Nemotron-H lightning, Kolibri, Qwen3.8 MLX) in one change.
2. **No bf16 MoE experts**: the MoE path is NVFP4-only, so a bf16 or dequantised MoE checkpoint cannot load (item 2).
3. **MoE router variants are hard-wired** to softmax + shared expert (item 3).
4. **`recipe new` finds problems but proposes few fixes** (item 7); every rename is a manual look at the tensor index.
5. **Nothing tells you before downloading or picking a model that a family needs a new block plus a new format.** A `recipe new --list-gaps`
   over all HF cache entries (22 s release build, under 1 s per model) would have shown this on 10-27 07:00.

## Next step (decision needed)

Either (a) allow a download of a dense `LiquidAI/LFM2.5-1.2B` bf16 (or LFM2-2.6B): that is the intended "one new block" drill (conv block,
GQA, dense FFN, no MoE, no quantisation) and would take the planned 2-3 h of engineering; or (b) widen the lane to items 1-5 on the
24B MLX checkpoint (dequant at load, bf16 MoE experts, sigmoid router, stacked experts, dense-then-MoE layers) before the block, which
is a multi-day change and not a block drill.

## Commands run

* `cargo build --release --features metal -p intarsia-cli` via `fleetq run --host endor --expect 12 --load cpu` (#6210): pass, 22 s.
* `./target/release/intarsia recipe new --model <LFM2.5-24B-A2B-MLX-4bit snapshot> --out raw/2026-10-27-nab/lfm2-scaffold.toml`: exit 1, 13 TODOs (expected; this is the finding).
