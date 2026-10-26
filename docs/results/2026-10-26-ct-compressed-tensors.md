# compressed-tensors NVFP4A16 checkpoints (lane ct, 2026-10-26)

Friction item 1 of the onboarding drill (`docs/results/2026-10-26-od-onboarding-drill.md`, section 5): checkpoints with
`quant_method = "compressed-tensors"` stopped `intarsia recipe new` and the loader. This lane makes the weight-only NVFP4 preset
(`NVFP4A16`, format `nvfp4-pack-quantized`) load and run. Branch `lane-ct/compressed-tensors`, built on `origin/main` after
PR #167. Backends, kernels and defaults are unchanged.

## What was wrong, what it is

compressed-tensors NVFP4A16 is the W4A16 NVFP4 maths the engine already has, under other tensor names and with the reciprocal tensor
scale. Read from the `compressed-tensors` 0.17.1 source (`compressors/nvfp4/{base,helpers}.py`,
`quantization/lifecycle/forward_helpers.py`):

| compressed-tensors | dtype / shape | the loader reads | relation |
|---|---|---|---|
| `weight_packed` | u8 `[N, K/2]` | `weight` | same bytes; two e2m1 codes per byte, element 2i in the LOW nibble (`pack_fp4_to_uint8`: `i0 \| i1 << 4`) |
| `weight_scale` | e4m3 `[N, K/16]` | `weight_scale` | same bytes; group size 16 along K |
| `weight_global_scale` | f32 `[1]` | `weight_scale_2` | `weight_scale_2 = 1 / weight_global_scale` |

Dequantisation is `w = e2m1(code) * (weight_scale / weight_global_scale)` (`_dequantize`: `scale = scale / global_scale`).
ModelOpt is `e2m1 * weight_scale * weight_scale_2`. The drill's reciprocal convention is therefore verified: the global scale is
`448 * 6 / amax` (real tensors: 8384.0 for a ThinkingCap `q_proj`, 52736.0 for a MiniCPM expert `down_proj`), the ModelOpt one is
`amax / (6 * 448)`. The fixture slices dequantise to std 1.0e-2 to 1.4e-2 (max 0.03 to 0.07) with the reciprocal, plausible weights.

The engine computes `1/g` once in f32 at open and multiplies; compressed-tensors divides. They differ by rounding only.

## What was built (core/quant, models loader, recipe)

* `crates/intarsia-core/src/quant/compressed_tensors.rs`: `CtNvfp4<S: WeightSource>`, a view over the checkpoint that shows each
  module with `weight_packed` + `weight_global_scale` under the ModelOpt names (`weight`, `weight_scale`, `weight_scale_2` as an f32
  `[1]` computed at open) and hides the compressed-tensors names. Format detection, the dense loader, tensor-parallel slicing and the
  per-expert MoE loader (`quant/moe.rs`, which reads `experts.N.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2}`) run
  unchanged; no backend, kernel or `nvfp4.rs` line changed. Malformed modules (scale not e4m3 or wrong shape, global scale not a finite
  positive f32, both `weight` and `weight_packed`) are refused with the module name. INT4 `pack-quantized` has `weight_packed` but no
  global scale and is not touched (it still fails loudly).
* `quant/scheme.rs`: `QuantScheme::parse_compressed_tensors` (new `Algo::CtNvfp4W4A16`, `COMPRESSED_TENSORS_NVFP4A16`): accepts only
  `nvfp4-pack-quantized`, 4-bit float, group 16, `tensor_group`, symmetric, e4m3 scales, no input/output activation quantisation,
  status `compressed`. Refused with the reason: W4A4 (quantised input activations), a non-empty `transform_config` (rotations the engine would not apply) or `sparsity_config`, a status other than
  `compressed`. A quantised `kv_cache_scheme` becomes a `kv_note`. Layers are NVFP4 or plain float; the `ignore` list (exact names and
  `re:` patterns; the core crate has no regex) is not interpreted, because the tensors found say which layers are quantised.
* `crates/intarsia-models/src/load.rs`: `read_quant_scheme` reads the scheme from `config.json` (also under `text_config`);
  `SchemedSource::open` wraps the safetensors in `CtNvfp4` only when that scheme is the compressed-tensors one. Every other checkpoint
  takes the unchanged path; that includes other compressed-tensors formats (per-tensor FP8 `float-quantized`, INT4, MXFP4), which keep their old behaviour at load (no declared scheme, formats found by tensor names; a test covers the FP8 case) and are still flagged by `recipe new`.
* `crates/intarsia-cli/src/cmd/recipe.rs`: `quant_method = "compressed-tensors"` is accepted and parsed; virtual `.weight` /
  `.weight_scale_2` inventory views (like the EXL3 ones) make `classify` see NVFP4 and the recipe templates resolve; `weight_global_scale`
  is a known parameter name.
* `tools/golden_ct_nvfp4.py`: an independent reference golden (HF Qwen3.5 / Qwen3.5-MoE decoder layers, one at a time, weights
  dequantised by the `compressed-tensors` package itself, 16-bit activations; per-expert tensors stacked into HF's fused layout). CPU
  only: 123 s for the 27B dense model, 208 s for the 35B MoE (8 short prompts).
* Docs: `docs/adding-a-model.md` (the format is no longer in the "stops at recipe" list).

No environment variable was added.

## Fixture test: the convention against the compressed-tensors reference

`scripts/ct/extract_slice.py` (stdlib only) cuts a small block out of one module of a checkpoint; `scripts/ct/make_reference.py`
(`uv run --with compressed-tensors==0.17.1 --with torch --with psutil`) adds the reference dequantisation computed by
`unpack_fp4_from_uint8` + `dequantize` in float32 (`NVFP4PackedCompressor.decompress` itself rounds through bf16; the script checks that
result agrees within bf16 rounding) and writes the two synthetic fixtures. Test:
`crates/intarsia-models/tests/compressed_tensors.rs` loads each through `CtNvfp4` + `load_linear` on the CPU backend and compares
`Nvfp4.dequant_ref` with the reference. Committed fixtures (`crates/intarsia-models/tests/fixtures/ct/`, synthetic only: the repository is
public and the checkpoints' licences are not ours to redistribute, so no real tensor bytes are committed):

| Fixture | Shape | Bit-equal | Worst |
|---|---|---|---|
| synthetic: every packed byte value, e4m3 scales from subnormal to 448 | 16 x 256 | 4096 / 4096 | 0 |
| synthetic: random weights through compressed-tensors compress (`quantize` + `pack_fp4_to_uint8`) | 16 x 128 | 1840 / 2048 | 1 ulp |

Real layers are an opt-in check: `CT_REAL_SLICES_DIR=<dir with extract_slice.py + make_reference.py output> cargo test -p intarsia-models
--test compressed_tensors real_slices` (skipped when unset). Run on slices of ThinkingCap-Qwen3.8-27B-NVFP4 layer 3 `self_attn.q_proj`
(8 x 128: 890 / 1024 bit-equal, worst 1 ulp) and MiniCPM-V-4.7-35B-A3B-NVFP4A16 layer 3 expert 5 `down_proj` (8 x 64: 469 / 512
bit-equal, worst 1 ulp); the slices are kept out of the repository.

The test requires every element within 2 f32 ulp (the difference is `s / g` against `s * (1 / g)`). Further tests: the view's names,
shapes and bytes; ModelOpt and INT4 sources left alone; malformed modules refused; a checkpoint directory (`config.json` +
safetensors) opened through `SchemedSource`, loaded, checked against its scheme, and a W4A4 declaration refused at open; the scheme
parser (accepted, refused with reason); `recipe new` covered for a CT-stored MoE and refused for W4A4 / INT4.

## Real models: `recipe new` and `intarsia onboard` quick tier on CUDA (spark2, GB10)

Both checkpoints were already on spark2 (`~/models`, disk 99 % full, so nothing was copied there; the binary is a 0.7 GB tree
built by `scripts/deploy.sh spark2 --tree ct`, a few minutes). Every box job went through fleetq (`--load cpu` for the build and the
CPU goldens, `--quiet gpu,cpu` for onboard). The reference goldens are the HF + compressed-tensors ones above; they are not committed.

| | ThinkingCap-Qwen3.8-27B-NVFP4 (dense, 20 GB) | MiniCPM-V-4.7-35B-A3B-NVFP4A16 (MoE, 23 GB) |
|---|---|---|
| `recipe new` | **covered** by `builtin:qwen3_5_vl` (45/45 templates, 0 TODOs) | not covered at the time of the run (now covered by the checked-in `recipes/minicpmv4_7.toml`); scaffold with 1 TODO, `norm_offset` (answered 1.0 from the weights: `input_layernorm` mean 0.033, `post_attention_layernorm` min -1.0, `q_norm` mean 0.33; the drill's finding); 61/61 templates of `qwen3_5_moe`, all blocks known, per-expert tensors (no fused layout) |
| smoke (load + 16 tokens) | PASS 48 s (cold weights): "Paris. The capital of Germany is Berlin..." | PASS 75 s: "Paris. The capital of France is Paris..." |
| quick tier | 8 m 43 s | 6 m 18 s |
| **onboard until quick numbers published** | **9 m 32 s** | **7 m 33 s** |
| decode 128 / 1k / 4k (tok/s, % of byte floor) | 12.9 / 12.9 / 12.7 (95 / 95 / 94 %) | 49.9 / 49.7 / 48.9 (98 / 97 / 96 %) |
| prefill 128 / 1k / 4k (tok/s) | 450 / 742 / 765 | 1219 / 2411 / 2489 |
| quick gate | PASS, perplexity 1.36 | PASS, perplexity 1.28 |
| `kernels` | 77 kernels, 1514 cases PASS | 77 kernels, 1494 cases PASS |
| **acc-short vs HF + compressed-tensors golden**, prefill (441 positions) | top-1 99.55 %, mean KL 4.54e-4 | top-1 98.41 %, mean KL 2.34e-3 |
| decode (128 positions) | top-1 100.00 %, mean KL 4.53e-4 | top-1 98.44 %, mean KL 2.51e-3 |

Context for the KL: the dense number is at the bf16-reference noise level (`docs/accuracy.md`: HF bf16 against HF fp32 is 5.6e-4 on
Qwen3-8B). The MoE number is higher, of the size of a reference's own spread on a 35B MoE (vLLM against itself on the Qwen3.6-35B
short golden: 3.16e-3, `docs/accuracy.md`, "Reference noise"): near-tie router choices flip between two bf16 implementations. This was
not separated further (no float32 MoE golden was made). A wrong global-scale convention or a mis-stacked expert would not give 98 to
100 % top-1; it would give garbage.

The onboard verdict of both runs is `FAIL` only because of `MISSING-BASELINE` on `acc-short` (no committed backend baseline for a
new model) and `acc-oracle` (no CPU-oracle self-golden); every executed stage is PASS or SKIP, with no floor violated. That is the
drill's friction item 3 (a new model has no committed baseline), not a loader finding. Decode runs the same NVFP4 GEMV kernels as the
ModelOpt checkpoints, at 95 to 98 % of the byte floor.

## What the accuracy evidence does and does not show

* The independent evidence for the **convention** (names, nibble order, reciprocal scale) is the fixture test against the
  compressed-tensors reference, on real layers, and the end-to-end agreement with a golden whose weights were dequantised by the
  compressed-tensors package.
* `acc-oracle` (engine against its own f64 CPU oracle) was not run: it uses the engine's dequantisation, so it checks kernels, not
  the convention (the oracle self-golden costs 22 min on a CPU for the 27B). It needs a committed self-golden and a stamped run on
  clean main, as the drill's item 3 describes.
* The in-depth tier (long and 4k goldens, envelope, speed sweep) was not run; only the quick tier was asked for.
* Only 8 short prompts per model.

## Not done / open

* MXFP4 compressed-tensors (`Qwen3.6-35B-A3B-MXFP4`, `mxfp4-pack-quantized`): not attempted. The checkpoint is on tatooine's radiance
  directory (HIP, radiance has to go down), and the scheme parser and view would need the MXFP4 layout checked against the
  compressed-tensors MXFP4 compressor first. The adapter shape is the same (rename only, no global scale), so it should be small.
* INT4 `pack-quantized` (AWQ, MiniCPM-V AWQ) stays refused, with the reason in the TODO list; it needs a kernel.
* The `ignore` list is not interpreted. A checkpoint that quantises a layer the recipe expects as plain float (or the reverse) fails
  at load with the engine's usual shape/format error, not with a scheme message.
* Fused `experts.gate_up_proj` layouts (the BF16 MiniCPM-V checkpoint) are a separate item (drill item 2), not touched.
* The MiniCPM-V recipe is checked in as `recipes/minicpmv4_7.toml` (builtin `minicpmv4_7`, text decoder only, the vision tower is
  ignored); `recipe new` now reports the checkpoint as covered.
