# 2026-10-26 lane od: onboarding drill on a model the engine has not seen

Lane `od`, branch `lane-od/onboarding-drill`. Question: when a new model's weights are on the box, does `intarsia onboard` get a
basic report in 1 to 2 hours and the in-depth report in 2 to 4 hours, in one back-to-back pass, with the quick numbers published while
the in-depth tier runs? Method: take the least-covered real checkpoint on the fleet's disks, run `onboard` end to end with wall
clock stamps, fix the cheap friction, rank the rest.

Raw material: `docs/results/2026-10-26-od-onboarding-drill/raw/` (driver logs, `quick.md`, `state.json`, `recipe new` outputs for every
candidate). All times are UTC on 2026-10-26 (EDT is 4 h earlier).

## 1. Result in five lines

* **Nothing that is really new gets past the `recipe` stage today.** Of the 14 rows of the candidate table (section 2): 9 unseen
  rows stop at `recipe new` (7 on a quantisation format, tensor layout or policy the loader does not accept: `compressed-tensors` in
  4 rows, fused experts, MLX, a refused W4A4 NVFP4; 2 on a new mechanism or architecture), 1 was pushed through, 3 are families
  earlier drills already onboarded and 1 is far over the size cap. `compressed-tensors` is the most common stop.
* The one checkpoint that could be pushed through, `Qwen3.8-27B-MXFP4-mtpfp8` (tatooine's radiance weights: the same main weights as
  the M1 drill's amd/Quark checkpoint, an FP8 MTP head, a different packaging), needed one code fix to get past `recipe`
  (Quark `layer_quant_config` that names only `mtp.*`). After it: `recipe` 0 s, `smoke` 47 s on CUDA (61 s on the CPU backend, 7 s on HIP).
* **HIP (tatooine R9700, the box that runs this model): quick numbers public 4 min 54 s after the start, whole pass 6 min 52 s.** Two
  caveats: `acc-short` passed only because a HIP baseline and noise envelope were already committed for these byte-identical
  weights (a genuinely new checkpoint is gated on KL only, and the same golden FAILs decode on CUDA, where no baseline exists); and 4 of the
  6 in-depth stages SKIPPED for want of goldens, so 6 m 52 s is not a measurement of a real in-depth tier. The requirement "quick
  numbers public while the in-depth tier keeps running" was checked on HIP: `quick.md` written `02:38:38`, in-depth ran `02:38:38` to
  `02:40:36`.
* **CUDA (spark1, GB10): targets not met.** The quick tier ended in ERROR at T0 + 51 min 53 s with no speed numbers and the in-depth tier
  was not started. The CUDA MXFP4 W4A4 path runs prefill at 9 tok/s and decode at 6.7 tok/s, so the 4k speed sample (3 reps x 7.5
  min) hit the 30 min stage limit. That is an engine gap, not a pipeline fault; the pipeline's job (stop, publish, say why) worked.
* Top friction is not the speed of the pipeline but what feeds it: formats (section 5, items 1 and 2), the reference and its gates
  (items 3 and 4), and moving the weights to the box (item 6).

## 2. Candidates

Searched: spark1 `~/dl ~/models ~/models-m2`, spark2 `~/models`, tatooine `/data/models/*`, endor `~/models`. `recipe new` was run on
each (headers or index only, no GPU): `raw/recipe-new/`.

| Checkpoint (host, size) | Architecture / quant | `recipe new` | Verdict |
|---|---|---|---|
| MiniCPM-V-4.7-35B-A3B NVFP4A16, `-attn`, `-attn-lmhead` (spark2, 23 / 21 / 20 GB) | `MiniCPMV4_7ForConditionalGeneration`: GatedDeltaNet + gated GQA + MoE with shared expert, the `qwen3_5_moe` blocks, text weights under `model.language_model.*`; compressed-tensors `nvfp4-pack-quantized` (`weight_packed` / `weight_scale` / `weight_global_scale`) | not covered, 3 TODOs | **First choice by the brief (new architecture, known blocks); blocked by the quant format** |
| MiniCPM-V-4.7-35B-A3B AWQ (spark2, 22 GB) | compressed-tensors `pack-quantized` int4, group 128 | not run | blocked, same reason |
| MiniCPM-V-4.7-35B-A3B BF16 (spark2 and endor, 66 GB) | experts stored fused (`experts.gate_up_proj [E, 2I, H]`, transformers 5 layout) | not covered, 4 TODOs (2 duplicates) | blocked: loader needs per-expert tensors; also 35B bf16 is over the size cap |
| MiniCPM-V-4.7-35B-A3B MLX-4bit (endor, spark2) | MLX affine 4-bit (`.scales` / `.biases`) | not covered | blocked: no MLX format |
| ThinkingCap-Qwen3.8-27B-NVFP4 (spark1 `~/dl`, spark2; 20 GB) | `Qwen3_5ForConditionalGeneration`, compressed-tensors NVFP4 | not covered, 2 TODOs | blocked: compressed-tensors |
| Qwen3.8-27B-NVFP4 `52d1adc5f38a` (spark2, 21 GB) | same family, ModelOpt `MIXED_PRECISION` with NVFP4 declared W4A4 | not covered, 5 TODOs | blocked: W4A4 is refused until the recipe says `[defaults] w4a4_as_w4a16 = true` (a decision, not a fix) |
| Qwen3.6-35B-A3B-MXFP4 (tatooine radiance, 22 GB) | `qwen3_5_moe`, compressed-tensors | not covered, 1 TODO | blocked: compressed-tensors |
| **Qwen3.8-27B-MXFP4-mtpfp8 (tatooine radiance, 19 GB)** | `qwen3_5_vl` family, Quark MXFP4 W4A4, MTP head in FP8 (`layer_quant_config` names `mtp.*` only) | 1 TODO before the fix, **covered after** | **chosen** |
| Nemotron-H "lightning" (endor, 17 GB) | `NemotronHForCausalLM`: Mamba2 + MoE + attention, MLX affine 4-bit | not covered, 6+ TODOs (Mamba mixer, no `rms_norm_eps`, non-uniform layers) | new mechanism (Mamba block) and MLX format |
| Kolibri-1 MLX-4bit (endor, 41 GB) | `Kolibri1ForCausalLM`: sliding window, MoE, MLX affine | not covered | unknown architecture and MLX format |
| qwen3-8b-exl3-4.0bpw (spark1, 4.9 GB) | Qwen3 + EXL3 | not run | covered, onboarded by lane m2 |
| Gemma-4-26B-A4B-NVFP4 (spark2, 18 GB) | gemma4 recipe | `covered` | covered, onboarded by lanes g4 / g4f |
| dsv41-exl3 (spark2, 372 GB) | DeepSeek V4.1 | not run | far over the size cap |
| Llama-3.1-8B, Qwen2.5-7B, Qwen3-8B (endor) | covered families | not run | covered |

Why `Qwen3.8-27B-MXFP4-mtpfp8` counts as "not seen": the checkpoint is a different file (single `model.safetensors`, 19.37 GB, FP8
MTP head, 1703 tensors) from the M1 drill's `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (19.80 GB). Six tensors checked by SHA-256
(`embed_tokens`, `lm_head`, an MXFP4 weight and scale of three different layers, a GatedDeltaNet projection) are byte-identical to
the amd checkpoint, which is why the committed W4A4 golden (`qwen3.8-27b-mxfp4-hfquark-top20`, provenance hash `1b32357e...`)
applies. It is the honest ceiling of what the fleet's disks offer; the brief's first choice (MiniCPM-V 4.7) is the stop in
section 5, item 1.

### MiniCPM-V 4.7: what `recipe new` and `onboard` do (stub of config.json + index, no GPU)

`intarsia onboard --model <MiniCPM-V NVFP4A16> --backend cpu` stops in 0 s with exit 3 and a numbered list. After the two fixes
below the list is three honest items instead of five:

1. `model.norm_offset` (decision): answered from the weights, not the config. `input_layernorm.weight` mean 0.033 (min -0.084),
   `post_attention_layernorm` min -1.0, `q_norm` mean 0.33, GatedDeltaNet gated norm mean 0.88. Zero-centred RMSNorm
   (`x * (1 + w)`), so `norm_offset = 1.0` as in `qwen3_5_moe`.
2. `model.merger.*` (6 tensors): the vision projector, now ignored by `recipe new` like `model.vision_tower.*`.
3. `quant: compressed-tensors is not implemented`: a loader task, not a recipe edit.

With 1 and 2 settled, the text blocks of MiniCPM-V 4.7 are exactly `recipes/qwen3_5_moe.toml` (49 of 61 tensor templates resolve in
the bf16 checkpoint; the 12 misses are the per-expert names, see the fused-experts item). No new block would be needed; no recipe
was checked in because none can be loaded until the format is read.

## 3. The run: `Qwen3.8-27B-MXFP4-mtpfp8`, CUDA on spark1

Setup (not counted in the targets, listed because a real user pays for it): the weights lived on tatooine, the model server for
this checkpoint; spark1 has no route to tatooine (host key), so the copy went through the Mac: **11 min 02 s for 19.4 GB (29 MB/s)**,
`00:41:34` to `00:52:36`. The build ran in parallel (deploy.sh, stamped, `cargo build --release --locked`: 46.6 s cargo time). Goldens: the three
committed `qwen3.8-27b-mxfp4-hfquark-*` files from tatooine's `/data/models/intarsia/golden` (245 kB each).

T0 below = weights on the box (`00:52:36`). The `onboard` command started 7 min later (a 5 min CPU pre-pass of mine plus 2 min of
fleetq queue behind lane dc).

| Stage | Start | End | Duration | Verdict | Note |
|---|---|---|---|---|---|
| (CPU pre-pass) `recipe` | 00:53:09 | 00:53:09 | 0 s | PASS | `builtin:qwen3_5_vl covers the model (45/45 templates, quant mxfp4)` |
| (CPU pre-pass) `smoke` | 00:53:09 | 00:54:10 | 61 s | PASS | "Paris. The capital of Germany is Berlin ..." at 0.3 tok/s decode |
| (CPU pre-pass) `quick`: doctor, recipe, plan 43.7 s, kernels skip, gate | 00:54:10 | 00:58:10 | stopped at 4 min | killed | stopped on purpose: the CPU gate of a 27B is slow and the job held `cpu` quiet for lane dc |
| `recipe` | 00:59:37 | 00:59:37 | 0 s | PASS | |
| `smoke` | 00:59:37 | 01:00:24 | 47 s | PASS | |
| `quick` (qualify) | 01:00:24 | 01:44:29 | 44 m 04 s | **ERROR** | stages below |
| .. `doctor` / `recipe` | | | 1 s / 0 s | PASS | |
| .. `plan` | 01:00:25 | 01:01:38 | 73 s | PASS | 19 ops resolved, load 73.3 s |
| .. `kernels` | 01:01:39 | 01:04:06 | 2 m 27 s | PASS | 77 kernels, 1514 cases |
| .. `gate` | 01:04:06 | 01:07:33 | 3 m 27 s | PASS | perplexity 1.58, 5/5 sanity prompts |
| .. `acc-format` | | | 0 s | SKIP | no format-reference golden (needs the format's own reference implementation) |
| .. `acc-short` (W4A4 golden) | 01:07:33 | 01:13:30 | 5 m 56 s | **FAIL** | prefill top-1 91.19 %, KL21 1.97e-1 (pass); decode KL21 2.97e-1 > ref noise 1.99e-1 (1.49x) FAIL; known state of this model, section 5 item 6 |
| .. `acc-oracle` | | | 0 s | MISSING-BASELINE | no f64 CPU-oracle golden for this name |
| .. `bench` (128 / 1024 / 4096) | 01:13:30 | 01:44:29 | 30 m 58 s | **ERROR** | stage limit 1800 s; 128: ttft 13.8 s, 1024: ttft 105 s, prefill 9 tok/s, decode 6.7 tok/s |
| `publish-quick` | 01:44:29 | 01:44:29 | 0 s | PASS | `quick.md` written |
| `in-depth` | | | not run | | quick ended in ERROR: the chain stops |

**Quick numbers were published at T0 + 51 min 53 s (01:44:29), but the quick tier ended in ERROR (no speed numbers) with a FAIL in it: the target
("basic test done") was not met on CUDA.** The in-depth tier was not started.

Engine finding behind the ERROR (cause not yet split, see section 5 item 5): `kernels/cuda/mxfp4.cu` has one GEMV kernel (146 lines) and no
MXFP4 GEMM, and the dynamic W4A4 activation quantiser runs per row; prefill is 9.4 tok/s = 1.4 x decode. HIP has the GEMM/fold paths
(`docs/results/2026-10-25-mx-hip-27b-prefill.md`). On the sizes the pipeline needs this extrapolates linearly: the quick bench alone would have been 128 (14 s x 3) + 1024 (105 s x 3) +
4096 (about 7.5 min x 3) = about 29 min, so the quick tier would have finished near T0 + 50 min and the in-depth speed sweep (1k / 4k /
16k, 16k alone about 29 min per repetition) would have been 1.5 to 2.5 h on top.

### 3.1 Same run resumed with the missing pieces supplied (CUDA)

To see the in-depth tier at all, the same `--out` was run again (`onboard` resume, `-- --skip bench --long-sizes 1024 --oracle-golden
<the 27B CPU oracle I made, section 5 item 3>`); finished stages were skipped as documented ("recipe: already done (PASS), not again").
`02:14:18` to `02:26:47`, **12 min 29 s**:

| Stage | Duration | Verdict | Note |
|---|---|---|---|
| `quick` (rerun part) | 3 m 12 s | FAIL | `acc-oracle` now runs: prefill KL21 1.945e-1 against a limit of 2e-2 |
| `in-depth` | 9 m 16 s | ERROR | |
| .. `acc-long`, `acc-4k` | 0 s | SKIP | no long / 4k golden (see item 3) |
| .. `envelope` | 35 s | **ERROR** | "this build carries no code stamp": `deploy.sh` baked a stamp the tool does not read (fixed, section 6) |
| .. `bench-long` | 8 m 40 s | PASS | 1024 only: decode 6.7 tok/s (47 % of the byte floor) |
| .. `ab`, `tp` | 0 s | SKIP | no `--reference`, no `--tp` |

The `acc-oracle` FAIL is the W4A4 activation quantiser, not a wrong kernel: with the policy golden and `--mxfp4-act none` (the W4A16
mode the decision log of 2026-10-25 12:45 makes the 27B correctness gate) the same CUDA build, `intarsia accuracy --mode both`,
**passes by a factor of 50**: prefill top-1 99.44 %, KL21 5.6e-4; decode top-1 99.38 %, KL21 1.08e-3 (limit 3e-2 for both), 94 s for the
whole command (`raw/cuda-w4a16/driver.log`). So the checkpoint is onboarded correctly on CUDA; what fails is the two W4A4 gates, which
the default `onboard` flow picks because the declared mode of the checkpoint is W4A4.

TBD-GEN

## 4. The run: same checkpoint, HIP on tatooine (one R9700, radiance down for the window)

The weights were already on tatooine. One `fleetq` window (`scripts/onboard/hip-window.sh`, radiance down `22:33:44` EDT to `22:40:51`
= **7 min 07 s**, radiance answered 20 s after the start), the stamped HIP binary built by `scripts/onboard/build-tatooine.sh` (cargo 1 m 11 s).
The job waited 21 min in the fleetq line (the box was busy with lanes l5, l7, tp2, lb). T0 = `02:33:44` UTC.

| Stage | Start | End | Duration | Verdict | Note |
|---|---|---|---|---|---|
| `recipe` | 02:33:44 | 02:33:44 | 0 s | PASS | |
| `smoke` | 02:33:44 | 02:33:50 | 7 s | PASS | |
| `quick` | 02:33:50 | 02:38:38 | 4 m 47 s | FAIL | = `acc-oracle` MISSING-BASELINE only |
| .. `doctor` / `recipe` | | | 1 s / 0 s | PASS | |
| .. `plan` | | | 33 s | PASS | |
| .. `kernels` | 02:34:25 | 02:37:00 | 2 m 35 s | PASS | 76 kernels, 1465 cases (10 kernels had no cases: "untested, not failed") |
| .. `gate` | | | 12 s | PASS | perplexity 1.59 |
| .. `acc-short` | 02:37:13 | 02:37:26 | 13 s | PASS | prefill top-1 91.95 %, decode 88.12 %; gated against the recorded HIP baseline + envelope (`golden/baselines/qwen3.8-27b-mxfp4-hfquark-top20.hip.*`) |
| .. `acc-oracle` | | | 0 s | MISSING-BASELINE | |
| .. `bench` | 02:37:26 | 02:38:39 | 73 s | PASS | decode 33.1 / 32.9 / 32.3 tok/s at 128 / 1k / 4k (82 / 81 / 80 % of the 40.5 tok/s floor); prefill 1055 / 1804 / 1854 tok/s |
| `publish-quick` | 02:38:38 | | 0 s | PASS | **quick numbers public at T0 + 4 m 54 s** |
| `in-depth` | 02:38:38 | 02:40:36 | 1 m 57 s | FAIL | the same MISSING-BASELINE |
| .. `acc-long`, `acc-4k`, `ab`, `tp` | | | 0 s | SKIP | no long / 4k golden, no `--reference`, no `--tp` |
| .. `envelope` | 02:38:38 | 02:39:19 | 41 s | PASS | baseline + envelope written and self-checked |
| .. `bench-long` | 02:39:19 | 02:40:36 | 77 s | PASS | 16.6k prompt: TTFT 9.8 s, prefill 1693 tok/s, decode 30.8 tok/s |
| `publish-in-depth` | 02:40:36 | | 0 s | PASS | **whole pass T0 + 6 m 52 s** |

Targets: quick numbers 1 to 2 h: **met at 4 m 54 s**. In-depth 2 to 4 h: **6 m 52 s, but this is not a measurement of a real in-depth
tier**: four of its six stages SKIPPED for want of a long golden, a 4k golden, a reference server and `--tp`, so what ran was the
noise envelope and the long-context speed sweep (2 min). With the long golden the comparison would be a few minutes more
(section 4.1); the 4k golden (`acc-4k`, where prefix reuse is gated) needs the Quark emulation tool, which needs the amd-quark wheel
(the M1 drill had it on endor; `find` on endor, spark1, spark2 and tatooine finds no `quark/` site-packages now), so it was not made.
Exit code 1 (`FAIL`) because `acc-oracle` has no baseline.

`acc-short` PASS needs its caveat next to the headline: the stage compared against a HIP baseline and noise envelope that were already committed
for this golden (recorded by lane ac on the byte-identical amd checkpoint). A checkpoint without them is gated on KL only, "top-1 not gated"; and on
CUDA, where this golden has no baseline, the same stage FAILs decode (section 3).

TBD-HIP2


## 5. Friction, ranked

## 5. Friction, ranked by what it costs the 1 to 2 h / 2 to 4 h target

Hour estimates are for one engineer who knows the code; none was built in this lane unless section 6 says so.

| # | Friction | Evidence | Fix and estimate |
|---|---|---|---|
| 1 | **`compressed-tensors` checkpoints stop at `recipe`.** 6 directories on the fleet (MiniCPM-V 4.7 NVFP4A16 x3 and AWQ, ThinkingCap-Qwen3.8-27B-NVFP4, Qwen3.6-35B-A3B-MXFP4) and 3 model families; the checkpoints came out of the fleet's quantisation work as far as the directory names and dates say (not checked). | `raw/recipe-new/`; the TODO says "quant_method compressed-tensors is not implemented" | NVFP4A16 (`nvfp4-pack-quantized`: `weight_packed`, `weight_scale`, `weight_global_scale`) is the W4A16 NVFP4 maths the engine has, under other tensor names and a reciprocal scale convention (`weight_scale_2 = 1 / weight_global_scale`; **verify on a one-layer fixture against compressed-tensors' own dequant before trusting it**). The backends do not name tensors (grep: only `crates/intarsia-core/src/quant/nvfp4.rs`, `quant/moe.rs:447-458`, `model.rs` and `recipe new`), so it is a loader adapter, no kernel: **4 to 6 h**. MXFP4 in the same container: +2 h. INT4 `pack-quantized` AWQ is a different scheme (a kernel): 1 to 2 days, not an adapter. |
| 2 | **Routed experts stored fused** (`experts.gate_up_proj [E, 2I, H]`, the transformers 5 layout: every bf16 MoE checkpoint saved by a current transformers). The main MoE block loads per-expert tensors only. | MiniCPM-V BF16; TODO text "needs converting or the block extending" | `model/mtp.rs:204` already reads exactly this layout for the MTP head and quantises to NVFP4 at load; reuse it for the main block: **6 to 8 h** (a 35B bf16 checkpoint is 66 GB, so the load path also has to stream). |
| 3 | **A genuinely new model has no reference, and the pipeline does not say so loudly.** `acc-short`, `acc-long`, `acc-4k`, `ab` all need a golden or a server; none can be made by `qualify` for a MoE / hybrid / quantised model. Here a committed golden existed only because the weights are byte-identical to an earlier checkpoint (checked by hand with SHA-256 of six tensors). `acc-oracle` also needs a CPU-oracle self-golden: **22 min 12 s on spark1's CPU for the 27B** (`raw/oracle1/`), then a commit and a stamped run on clean main. On HIP 4 of the 6 in-depth stages SKIPPED (`acc-long`, `acc-4k`, `ab`, `tp`) and the report still ends in a normal "in-depth done". | `raw/hip/hip1/report/in-depth.md`, `raw/oracle1/` | (a) `onboard` states in the verdict line "in-depth tier: N of M stages skipped for want of a reference" (1 h); (b) `onboard --make-reference` that runs the format's tool, records provenance and does the weights-identity check (6 h); (c) the Quark emulation tool needs the amd-quark wheel, which is on no box now: vendor the `even_round` function the tool already extracts with `ast` (2 h). |
| 4 | **Default W4A4 gating reports FAIL for a correct engine.** Declared mode W4A4: `acc-short` decode KL 2.97e-1 = 1.49 x reference noise on CUDA (HIP: baseline-gated, PASS), `acc-oracle` KL 0.19 against 2e-2. With `--mxfp4-act none` and the W4A16 golden the same CUDA build passes with a 50x margin. The policy is in the decision log (2026-10-25 12:45, draft #150 pending) but not in `onboard`. | section 3.1 | `onboard` runs a second accuracy pass in W4A16 for Quark MXFP4 W4A4 and gates on it (it needs `qualify --mxfp4-act`): **2 to 3 h**. Until then the doc should say "expect FAIL on `acc-short` decode and `acc-oracle` for W4A4". |
| 5 | **CUDA MXFP4 W4A4 is slow**: prefill 9 tok/s and decode 6.7 tok/s on GB10; `acc-short` takes 357 s against 13 s on HIP. The 4k sample of the quick bench exceeds the 30 min stage limit, the quick tier ends in ERROR with no speed numbers, and the chain stops. Two observations that point at different code: `kernels/cuda/mxfp4.cu` is a single GEMV kernel (no MXFP4 GEMM for prefill), and the same CUDA build ran the whole W4A16 accuracy command (`--mxfp4-act none`, 1441 + 160 positions, including a 73 s load) in 94 s against 357 s for the W4A4 `acc-short`, i.e. the dynamic activation quantiser may cost at least 4x by itself. TBD-GEN2 | `raw/cuda1/quick.md`, `raw/cuda-w4a16/driver.log` | Engine: find which of the two it is first (a `generate` or `bench` run in both modes, **1 h**); an MXFP4 GEMM for CUDA prefill (the HIP fold path is the model) is **1 to 2 days**, a fused activation quantiser less. Pipeline, **3 h**: (a) bench preflight: time the 128 sample and, if the projected 4k time passes the stage limit, run the smaller sizes and report "4096 skipped: projected N min"; (b) write the bench JSON after each size so a timeout keeps the finished rows; (c) a bench ERROR in the quick tier should not stop the in-depth accuracy stages (the brief: quick numbers public while the in-depth tier keeps running). |
| 6 | **Moving the weights to the box.** spark1 cannot ssh tatooine (host key) and the repo has no box-to-box copy: 19.4 GB took **11 min 02 s** through the Mac (29 MB/s) plus a checksum check by hand. | `raw/` copy timestamps in section 3 | `scripts/onboard/copy-weights.sh <from> <to> <dir>` as a fleetq `net,disk` job with a header + sampled-tensor hash check: **1 h**. |
| 7 | **Build stamp.** `deploy.sh` baked `head= tree= backend=`, which `qualify`'s `envelope` stage does not read: ERROR "carries no code stamp" on every first run of a deploy-built binary. Three build paths (`deploy.sh`, `build-tatooine.sh`, `phase1b-acceptance.sh`), two stamp forms. | `raw/cuda1-resume/in-depth.md` | `deploy.sh` stamp fixed (section 6). Making the three paths one script: **2 h**. The oracle self-golden I made on spark1 was written by that binary and is refused as a "before"; rebuild and redo before committing one. |
| 8 | **The line.** Boxes are shared: the CUDA job waited 2 min behind lane dc, the HIP window 21 min behind five lanes (plus about 30 min for the build job), and a model-sized job holds `cpu` or `gpu` quiet for others (I stopped my CPU pre-pass at 5 min because it blocked lane dc). Not a pipeline fault; it is the part of "2 hours" nobody controls. | fleetq logs | none in the pipeline. |
| 9 | **Small things.** Disk: spark2 100 % full (27 GB free; nothing can be built or copied there), spark1 7 % free (`doctor` warns). `recipe new` without `--out` writes into the checkout's `recipes/<family>_<n>.toml` (I left two stray files in the worktree before noticing); `publish-quick: PASS` printed after a failed quick tier (the report's verdict line is right); `docs/adding-a-model.md` did not say how to decide `norm_offset`. | | doc and 3 small fixes in section 6; the `recipe new` default `--out` and the log line: 1 h. |


## 6. What was fixed

## 6. What was fixed (all bitwise-neutral: no kernel, no default, no output of an existing model changes)

Branch `lane-od/onboarding-drill`, commits on top of `origin/main` `fecf2d24e`. Checks: `cargo test --release -p intarsia-cli --no-default-features --lib`
(174 passed, 1 ignored), `cargo test --release -p intarsia-core --lib` (126 passed, 1 ignored).

* **Quark `layer_quant_config` that names only `mtp.*` is accepted** (`crates/intarsia-core/src/quant/scheme.rs`, `parse_quark`). Before, any
  non-empty override map was an error, which stopped `Qwen3.8-27B-MXFP4-mtpfp8` (FP8 MTP head beside MXFP4 decoder layers) at `recipe` and
  at load. The engine reads `mtp.*` only for `--spec mtp`, through a loader that reads bf16 heads and rejects anything else; any override
  that names a decoder layer is still an error, now with the first name in the message. Test added (mixed, decoder-only, look-alike prefix).
  Only checkpoints that errored before are affected.
* **`recipe new`** (`crates/intarsia-cli/src/cmd/recipe.rs`): `model.merger.*` (the vision projector outside the vision tower) counts as a non-text
  group, so MiniCPM-V no longer raises a TODO for its 6 tensors; a finding raised by two layer types (full and linear attention both
  have the MoE block) is listed once; the quantisation note names a declared but unread `quant_method` instead of printing a bare
  `bf16` when only an index exists. Test added.
* **`deploy.sh` bakes `head= code-tree= dirty=no` (+ the old `tree=` / `backend=`)**, the form `CodeStamp::parse` reads, so `envelope` no
  longer ERRORs on a deploy-built binary. Test added in `provenance.rs`; `docs/reproducing-results.md` updated.
* **`onboard` timeline** shows `-` for a stage that never ran (`--skip`) instead of 1970-01-01. Test added.
* **`scripts/onboard/build-tatooine.sh` and `hip-window.sh` take `LANE=`** (own tree, cargo target volume and `/models` mount, so a second lane does not
  overwrite the first lane's binary), and `hip-window.sh` checks the binary and the model before it stops radiance.
* **`docs/adding-a-model.md`**: how to decide the `norm_offset` TODO from one tensor, which checkpoint formats stop at the recipe stage, the
  `mtp.*` override rule, and the measured row for this checkpoint in the timings table.
* **`docs/decisions/LOG.md`**: one row, status `active`.

No recipe was checked in: the chosen checkpoint is covered by `recipes/qwen3_5_vl.toml`, and the MiniCPM-V recipe cannot be loaded yet
(section 5, items 1 and 2).


## 7. Reports

## 7. Reports and raw material

| What | Path |
|---|---|
| HIP, quick report (T0 + 4 m 54 s) | `docs/results/2026-10-26-od-onboarding-drill/raw/hip/hip1/report/quick.md` (+ `.json`, `.prom`) |
| HIP, in-depth report (T0 + 6 m 52 s) | `.../raw/hip/hip1/report/in-depth.md` (+ `.json`, `.prom`), `qualify/state.json`, `window.log`, `downtime.log` |
| CUDA, quick report (ERROR, T0 + 51 m 53 s) | `.../raw/cuda1/quick.md`, `driver.log`, `report/state.json`, `bench-bench.err` |
| CUDA, resumed quick + in-depth report | `.../raw/cuda1-resume/in-depth.md`, `driver.log` |
| CUDA, W4A16 policy gate | `.../raw/cuda-w4a16/driver.log` |
| CPU pre-pass (recipe, smoke, plan) | `.../raw/cpu1/` |
| CPU f64 oracle self-golden, 22 min 12 s | `.../raw/oracle1/` (the golden itself, 756 kB, is not committed: unstamped build) |
| `recipe new` of every candidate | `.../raw/recipe-new/` (`before-fix-*` and `post-fix-*`) |

The report files on spark1 are `~/intarsiaml-od/runs/{cuda1,cpu1,oracle1,w4a16}`, on tatooine `~/intarsiaml-od/onboard-runs/hip1`.

## 8. Open items and doubts

* The HIP in-depth number is not comparable to the 2 to 4 h target until a long, a 4k and a reference-server comparison exist; the honest
  statement is "quick tier 4 m 54 s, in-depth tier cannot be measured without goldens". The MiniCPM-V and ThinkingCap-class checkpoints
  would additionally have no committed baseline, so even the quick numbers would be KL-only.
* Time between "weights on the box" and the first public number for this checkpoint on HIP, if the weights arrive on tatooine: build
  1 m 11 s + window about 5 min; on a box that does not hold them add the 11 min copy.
* The CUDA numbers are for the W4A4 dynamic path only. The W4A16 mode was run through `accuracy` (94 s) but not benchmarked through `qualify`.
* MiniCPM-V 4.7, the brief's first choice, was never loaded: nothing here says it would run correctly once the format and layout
  items land; its blocks are the ones `qwen3_5_moe` already has, which is the best case for "no new block" but unproven.
* **A better next subject than a repackaged checkpoint:** `Qwen3.8-27B-NVFP4-52d1adc5f38a` (spark2, 21 GB) is a different quantisation of a covered
  family (45/45 tensor templates resolve) blocked only by the `w4a4_as_w4a16` decision. A scratch recipe passed with `--recipe` would not
  touch any existing model. Not run here (it needs that decision first, and spark2's disk is 100 % full).
* Radiance was taken down by this lane TBD-RAD; each time the script restored it and `fleetq status` showed neither rank DOWN afterwards.
* Left behind for the coordinator (the worktree is not removed by this lane): the 19.4 GB model copy and the goldens at
  `spark1:~/intarsiaml-od/` (spark1 has 7 % disk free; `models/` is the large part), the run directories there, `tatooine:~/intarsiaml-od`
  (tree, `od-intarsia` binary, `onboard-runs`, `extra/` oracle copy), the docker volume `intarsia-cargo-target-od` on tatooine, and the
  unstamped CPU oracle self-golden (section 5 item 7: rebuild before using it).
