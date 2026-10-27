# NAB: LFM2.5 new-architecture drill (stopped at the premise check) and the MLX affine reader it led to (2026-10-27 morning)

Lane NAB, branch `lane-nab/lfm2-block`, PR #205, main `e7e0845ec`, endor (Apple M5 Max, Metal). Times are the box clock (EDT). Requirement 1:
a basic test within 1-2 h of the weights, in-depth within 2-4 h, also for a model that needs something new.

**Part 1 (07:38-07:46): the LFM2.5 drill could not start.** The only LFM2.5 on endor is `LFM2.5-24B-A2B-MLX-{4,6,8}bit`, an MoE in a
quantisation format the engine did not read, so the short-conv block was one of nine gaps. Stopped, nothing written.
**Part 2 (07:47-08:31): the coordinator opened gap 1, the MLX affine reader.** Implemented as a `WeightSource` wrapper that dequantises to
bf16 at load, validated on `mlx-community/Qwen3.5-2B-MLX-4bit` (dense, a size of the supported `qwen3_5_vl` family): `onboard` PASS on Metal
through the quick and the in-depth tier. The LFM2 block itself (gaps 2-6) is still not written.

## Part 2: MLX affine quantisation, validated on Qwen3.5-2B-MLX-4bit

### Result

| Gate | Result |
|---|---|
| (a) cpu-ref unit test against a numpy reference of `w = scale * q + bias`, bitwise to bf16 | **PASS.** `quant::mlx_affine::tests::dequant_matches_the_numpy_reference_bitwise`: 18 cases (2, 3, 4, 5, 6, 8 bits x group 32, 64, 128), expected values from numpy (`tools/gen_mlx_affine_fixture.py`, which also asserts numpy == `mx.dequantize` on every case, packed words from `mx.quantize`). 5 unit tests in all (also config parsing with per-module overrides, layout rejection, name / conv mapping, the norm shift). |
| real checkpoints, beyond the fixture | **PASS.** Qwen3.5-2B-MLX-4bit through `SchemedSource` against the numpy dequantiser's bf16 directory (`tools/mlx_to_hf_bf16.py`): all 320 text tensors identical bit for bit, 3.76 GB (`tests/mlx_affine_real.rs`, ignored by default). LFM2.5-24B-A2B-MLX-4bit (4-bit body, **6-bit embedding via the config override**, stacked 3-D experts `[64, 1536, 256]` words) opens with the implied shapes, `[128000, 2048]` and `[64, 1536, 2048]`, and finite values; 6-bit rows of its embedding match `mx.dequantize` (0 of 131072 differ). |
| (b) `intarsia onboard` PASS on Metal, auto golden | **PASS** both tiers (`raw/2026-10-27-nab/onboard2-*.md`): 4 min 16 s whole pass, 11 of 15 stages PASS, 4 skipped (below), **0 fail**. Reference used: **the MLX-dequantised HF reference** (no bf16 original: `Qwen/Qwen3.5-2B` is not in the HF cache; the 4B and 9B-Base are, the MLX 2B has no sibling). So the golden measures the engine against HF on the same dequantised weights (rounding), **not the 4-bit loss against the original**. Short golden made by `qualify` itself in 57 s (HF fp32, CPU, bf16 spread recorded). |
| (c) `plan show` byte-identical, Qwen3-8B and Qwen3.5-4B, Metal | **PASS**, text and `--json`, base `e7e0845ec` binary against this branch (sha256 in `raw/.../plan-show/sha256.txt`). |
| (d) Mac `cargo test --workspace` | **PASS** locally: 1061 passed, 0 failed (`--release --locked --no-fail-fast`, fleetq #6245). CI: see the PR. |

Numbers of the in-depth `onboard` (Qwen3.5-2B-MLX-4bit, Metal, bf16 after dequantisation, reference = same weights dequantised in HF):

| Stage | Result |
|---|---|
| acc-short (441 prefill / 128 decode rows) | top-1 99.55 % / 100.00 %, KL 1.39e-4 / 9.74e-5 |
| acc-long (HF fp32 golden, long set) | prefill top-1 98.75 % KL 6.23e-4, decode top-1 99.40 % KL 1.69e-4 |
| acc-4k (HF fp32 top-20 golden, 3 x 4k tokens) | prefill top-1 99.30 % KL 2.89e-4, decode top-1 99.48 % KL 1.61e-4 |
| gate | perplexity 4.58, 5/5 prompts, no non-finite |
| speed (`bench`, eager decode, 3.76 GB per token, byte floor 163 tok/s) | decode 129.6 / 132.2 / 131.2 tok/s at 128 / 1k / 4k (79-81 % of the floor), 127.5 at 16.7k; prefill 5.5k / 10.2k / 10.2k tok/s at 128 / 1k / 4k, 6.5k at 16.7k (TTFT 2.6 s) |
| skipped | `acc-format` and `acc-oracle` (no format reference / CPU oracle golden), `ab` (reference is a directory), `tp` (not asked) |

The first (quick-only) pass read 107 tok/s decode; the second, with the GPU quiet, 130. Not compared with mlx-lm (no time); not run: MMLU / GSM8K
(not asked in this scope).

### Timeline (from the coordinator's message at 07:47)

| Stage | Start | End | Wall |
|---|---|---|---|
| Pick the checkpoint (cache scan: `Qwen3.5-2B-MLX-4bit`, 1.7 GB; its bf16 original is absent) and read the safetensors header | 07:47 | 07:49 | 2 min |
| Fixture generator + `mlx_affine.rs` + 4 unit tests green (cpu-ref, first run) | 07:49 | 07:54 | 5 min |
| Wire into `SchemedSource`, `recipe new` view (`covered`, 45/45 templates) | 07:54 | 07:55 | 1 min |
| First load: **fails** `bf16 tensor at byte 757762825 is not 2-byte aligned` (MLX writes an unpadded header, data starts at an odd byte); fix in `load_tensor` | 07:55 | 07:56 | 1 min |
| Second load: fluent nonsense; found the cause in 1 min (norm weights have mean 1.1, the converter baked in the `+1`); fix + test | 07:56 | 07:57 | 2 min |
| **First correct completion** ("Paris.") | | **07:57** | **+10 min** |
| numpy reference directory (`tools/mlx_to_hf_bf16.py`, 10 s) + real-checkpoint bit test, commit | 07:57 | 07:58 | 1 min |
| `onboard` 1 (golden made by qualify): quick.md public | 07:58 | 08:01 | **+14 min** from the message; quick 2 m 30 s, PASS in 3 m 26 s |
| long + 4k HF fp32 goldens by hand (CPU; the 4k one 22 min, as the NMD drill's 34 min for the 4B) | 08:02 | 08:25 | 23 min |
| `cargo test --workspace`, `plan show` A/B (base build 28 s) in parallel with the goldens (approximate times) | 08:05 | 08:15 | |
| `onboard` 2 with all goldens: in-depth PASS | 08:27 | 08:31 | 4 min 16 s |

Against the targets: **basic 1-2 h: 14 min** from the decision to run; **in-depth 2-4 h: 44 min** (08:31), 22 of them the hand-made 4k golden.
Part 1 is not counted: before the reader existed an MLX checkpoint stopped at `recipe new` (13 TODOs, 4 of them about the format).

### What was built

* `crates/intarsia-core/src/quant/mlx_affine.rs`: `MlxAffineConfig` (config `quantization`, per-module overrides, `mode` other than `affine` refused by
  name), `dequantize_bf16` (little-endian bit stream, 2/3/4/5/6/8 bits, f32 `scale * q + bias`, one RNE rounding to bf16, threaded over rows) and
  `MlxAffine<S: WeightSource>`: packed modules read as bf16 `.weight` (lazily, cached), the bits checked against the tensor shapes and the config,
  `mlx-vlm` names mapped, `[C, K, 1]` conv shaped `[C, 1, K]`, the converter's `+1` on zero-centred norms undone when the probe norm has mean above 0.5.
* `intarsia-models`: safetensors `U32` carried as `I32` (same bytes, one reader), `SchemedSource` picks `MlxAffine` for a config with `quantization` and no `quant_method`.
  `intarsia-core` `load_tensor` copies instead of mapping a tensor whose file offset is not a multiple of its element size.
* `intarsia-cli`: `recipe new` shows the bf16 view and a note with the 4x memory cost; `qualify`'s byte floor counts the bf16 bytes the engine reads;
  the gate class of an MLX checkpoint is its lowest bit width (`<= 5` bits the 4-bit class, else FP8) instead of "unrecognised quantization metadata".
* `tools/mlx_to_hf_bf16.py`, `tools/gen_mlx_affine_fixture.py`, `docs/adding-a-model.md` (MLX paragraph and the "new quantisation format" note).

Cost, stated plainly: **dequantising to bf16 takes 4x the stored size in memory** (Qwen3.5-2B-4bit: 1.7 GB on disk, 3.76 GB loaded; the 24B LFM2 4-bit would
be about 48 GB) and reads bf16 bytes per token, so decode runs at bf16 speed. A native MLX-quant GEMV is a later lane.

### Friction list of part 2, ranked

1. **The reference.** No bf16 original of an MLX checkpoint is guaranteed to exist, and HF cannot load one: the gate then measures rounding, not the 4-bit loss. One-line fix: `qualify --reference <mlx dir>` calls `tools/mlx_to_hf_bf16.py` itself (10 s) and labels the golden `reference = MLX dequantised`.
2. **The golden for long and 4k prompts is manual and took 22 of the 44 minutes** (HF's GatedDeltaNet loop on CPU). Fix: install `flash-linear-attention` / `causal_conv1d` in `tools/pyproject.toml` or run the golden on the GPU (`golden.py --device mps`).
3. **Three silent converter edits in `mlx_lm`** (names, conv layout, the `+1` on the norms) each cost a failed run or a minute of diagnosis; the `+1` produced fluent nonsense, not an error. Fix: `recipe new` could print the mean of `input_layernorm` for zero-centred families (it already knows the family) and say "shifted".
4. **An odd-byte data offset** in MLX safetensors crashed the Metal import deep in generation. Fix done (`load_tensor` copies when `offset % element size != 0`); `Backend::wrap_host` should reject it itself with the tensor name.
5. **`cargo test --release` can leave `target/release/intarsia` built without `--features metal`**: onboard 2's first attempt died at smoke with "built without the Metal backend" (the error names the fix, `cargo build ... --features metal`, and a rebuild took seconds). One-line fix: keep the Metal binary in its own `--target-dir`, or have `onboard` check the feature before it takes a fleetq slot.
6. **An env-var test**: the docs table test fails on any `INTARSIA_*` name in a test file; renamed. Fix: let `docs_env_vars` skip `tests/`.
7. `qualify` treated a quantised checkpoint as "no golden can be made" and `weight_class` bailed on unknown metadata: both needed an MLX case (done). Fix for the next format: one `QuantFamily` enum instead of three string probes (`recipe.rs`, `facts.rs`, `accuracy.rs`).

Not done: the LFM2 block and gaps 2-6 below; CUDA / HIP (the wrapper is host-side and backend-independent, nothing backend-specific was added); the MLX
`mxfp4` / `nvfp4` / `mxfp8` modes (refused by name); stacked MoE experts (the MoE block is NVFP4-only); MMLU / GSM8K; mlx-lm speed comparison.
Uncertain: the gate class for 6-bit checkpoints (FP8 class, chosen by bit width, unmeasured on a 6-bit model); `Qwen3.5-2B` has a vision tower, left unread.

## Part 1: the LFM2.5 premise check (07:38-07:46, nothing written)

### What is on disk

`LiquidAI/LFM2.5-24B-A2B-MLX-{4bit,6bit,8bit}` (13 / 18 / 24 GB), `Lfm2MoeForCausalLM`: 40 layers, hidden 2048, 30 `conv` layers and 10
`full_attention` layers (GQA 32/8 heads, qk layernorm), layers 0-1 with a dense FFN (`intermediate_size` 11776) and layers 2-39 with a
64-expert top-4 MoE (`moe_intermediate_size` 1536, no shared expert). The smallest is the 4-bit one (snapshot `667408d4`); no bf16 or
smaller (1.2B / 2.6B) LFM2 was found on endor (`find` for lfm2 / liquidai names to depth 6, plus the HF cache listing) and nothing may be downloaded.

### Stage timing

| Stage | Start | End | Wall | Note |
|---|---|---|---|---|
| Read docs, prior drill, checkpoint config and tensor index | 07:38 | 07:43 | 5 min | found MLX-only checkpoint, MoE |
| Advisor check of the premise, offline-reference search, HF router and conv code | 07:43 | 07:46 | 3 min | |
| `intarsia recipe new` on the 4-bit checkpoint (release build, fleetq #6210, 22 s) | 07:44 | 07:44 | under 1 s | **13 TODOs, exit 1**, raw output in `raw/2026-10-27-nab/` |
| Decision: stop, report (spec premise broken) | 07:46 | | | |

Targets: basic test 1-2 h, in-depth 2-4 h. **Neither was measured.** The honest reading is that for this checkpoint the targets are
not reachable by adding one block: see the list below.

### What stands between this checkpoint and `onboard` PASS, ranked by cost

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

### Friction list of part 1

1. **Checkpoint format coverage is the real gate, not block count.** Four of the six families listed in the 10-26 drill, and all three
   LFM2.5 copies here, stop at the quantisation format. A load-time MLX-affine dequant (item 1) would unblock the cached MLX models
   (LFM2.5, Nemotron-H lightning, Kolibri, Qwen3.8 MLX) in one change.
2. **No bf16 MoE experts**: the MoE path is NVFP4-only, so a bf16 or dequantised MoE checkpoint cannot load (item 2).
3. **MoE router variants are hard-wired** to softmax + shared expert (item 3).
4. **`recipe new` finds problems but proposes few fixes** (item 7); every rename is a manual look at the tensor index.
5. **Nothing tells you before downloading or picking a model that a family needs a new block plus a new format.** A `recipe new --list-gaps`
   over all HF cache entries (22 s release build, under 1 s per model) would have shown this on 10-27 07:00.

### Next step (decision at the time)

Either (a) allow a download of a dense `LiquidAI/LFM2.5-1.2B` bf16 (or LFM2-2.6B): that is the intended "one new block" drill (conv block,
GQA, dense FFN, no MoE, no quantisation) and would take the planned 2-3 h of engineering; or (b) widen the lane to items 1-5 on the
24B MLX checkpoint (dequant at load, bf16 MoE experts, sigmoid router, stacked experts, dense-then-MoE layers) before the block, which
is a multi-day change and not a block drill.

### Commands run

* `cargo build --release --features metal -p intarsia-cli` via `fleetq run --host endor --expect 12 --load cpu` (#6210): pass, 22 s.
* `./target/release/intarsia recipe new --model <LFM2.5-24B-A2B-MLX-4bit snapshot> --out raw/2026-10-27-nab/lfm2-scaffold.toml`: exit 1, 13 TODOs (expected; this is the finding).

### After the reader (07:50 re-run of `recipe new` on the same 4-bit LFM2.5)

The 13 TODOs became 9: the four format items (10-13) are gone, `quant: bf16 (MLX affine 4-bit dequantised at load)`; what is left is gaps 2-7 above
(`raw/2026-10-27-nab/recipe-new-lfm2-24b-mlx4-after.txt`). Next for the block drill proper: a bf16 MoE-experts path or an NVFP4 requantisation, the sigmoid +
`expert_bias` router, the dense-then-MoE layer mix, then the short-conv block (`gdn_conv` with the activation off, state in the `Recurrent` slot).
