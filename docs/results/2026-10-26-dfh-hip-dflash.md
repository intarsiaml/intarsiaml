# DFlash2 speculation on HIP for Qwen3.8-27B (tp=1, tp=2), lane dfh, 2026-10-26

Branch `lane-dfh/hip-dflash` (off origin/main `f0181b2df`). Lane approved by the user 2026-10-26 08:09 EDT. Status: **phase 0
(feasibility) done, no engine code; go/no-go is the coordinator's call (section 6).**

## 0. Sources and licences

| source | what we take | licence |
|---|---|---|
| `z-lab/Qwen3.8-27B-DFlash2` (HF, revision `50307d4c`), copy on tatooine at `/data/models/tensorfold/Qwen3.8-27B-DFlash2` (and `/data/models/radiance/Qwen3.8-27B-DFlash2-bf16`) | drafter weights, bf16, 3,848,817,896 B | **Apache-2.0** (model card front matter) |
| `z-lab/dflash` `dflash/model_mlx.py`, vendored unmodified in `/opt/code/TensorFold/src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` | DFlash2 semantics (conv, sliding window, selector, row layout) and the reference for the parity fixture | **MIT** (c) 2026 Z Lab |
| TensorFold `src/tensorfold/families/qwen3_5/cuda/dflash2.py` (user's fork, `jkuepker/TensorFold`) | read for semantics only (tp=2 vocab-parallel top-16 merge) | MIT |
| radiance (`stilldeadcode/radiance` 1.12.0 container) | only its launcher, systemd unit, journal, `/metrics` and the `.rad` header keys were read; no code | n/a (nothing taken) |

Not used: the FP8 variants (`Qwen3.8-27B-DFlash2-FP8`, `tcclaviger/Qwen3.8-27B-DFlash2-FP8`).

Provenance check (no 3.8 GB hashing): `git hash-object` of `config.json` and `README.md` on both tatooine copies is
`79279cc5...` and `1ee0eb2b...`, the blob ids of the z-lab snapshot `50307d4c` on this Mac (whose weight blobs were never
downloaded; we do not download them).

## 1. The target and the drafter (Q1)

**Target.** "The 27B" is `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (`5233554c`, W4A4 MXFP4, recipe `qwen3_5`), on tatooine at
`/data/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/...` (mounted as `/models` by the tp2 scripts).
Hybrid: 64 layers, GatedDeltaNet in 48, gated full attention in 16 (layers 3, 7, ..., 63, `full_attention_interval 4`);
hidden 5120, FFN 17408, attention 24 q / 4 kv heads at head_dim 256, vocab 248320, untied bf16 `lm_head` (excluded from
quantisation), plus a one-layer bf16 MTP head.

**Radiance's drafter.** The `.rad` file bundles its own drafter (`--draft-model: not implemented (the container's own
drafter)`). Its header carries `draft.*` keys with the DFlash2 shape: `dflash_config.{block_size, conv_group_size,
conv_kernel_size, mask_token_id 248070, selector_rank, selector_top_k, target_layer_ids 5 19 33 47 61}`, `sliding_window 2048`.
The journal at start: "'dflash2' proposes 7 token(s) in ONE block pass over 8 row(s) (mask id 248070, window 2048)". So radiance
runs DFlash2 for this target; whether its weights are the bf16 or an FP8 conversion is not visible.

**A drafter exists for exactly this target, with a usable licence:** `z-lab/Qwen3.8-27B-DFlash2`, Apache-2.0, already on
tatooine. (Trained against the bf16 Qwen3.8-27B; our target is W4A4, so acceptance may be lower than the card's.)

**It is DFlash2, not the DFlash that PR #158 built.** Config (`config.json`) and tensors (safetensors header):

| item | DFlash2 27B drafter | DFlash v1 (#158, Qwen3-8B) |
|---|---|---|
| layers | 5, hidden 5120, FFN 17408, 32 q / 8 kv heads, head_dim 128 (target: 24 / 4 at 256) | 5, same geometry as the target |
| block | 8 rows: `[anchor, mask x 7]`, `logits_start=1`: anchor row dropped, rows 1..7 give 7 drafts, 8 verify rows | 15 rows, anchor row included (`sample_from_anchor`) |
| taps | `target_layer_ids [5, 19, 33, 47, 61]` of 64: 5, 33, 61 are GDN layers, 19, 47 gated attention; `fc` [5120, 25600] | 5 dense layers |
| attention | non-causal over block, **sliding window 2048** over context (keeps the last 2047 context rows) | non-causal, full context |
| per layer extra | two `GroupedDynamicCausalConv` (`attention_conv`, `mlp_conv`): `base_kernel` [2, 2, 5120], `kernel_projection` [1280, 5120]; 320 groups of 16 channels, 2-tap causal along block rows, kernel = base + projection of the normed input | none |
| embed / lm_head | **none in the checkpoint**: binds to the target's `embed_tokens` and `lm_head` | own copies |
| selector | `candidate_selector`: top-16 of logits per row, `hidden_projection` [256, 5120], `predecessor_codebook` / `successor_codebook` [248320, 256]; greedy walk from the anchor: `score = logit + sum(pred[prev] * proj(h) * succ[cand])` | (DCOCO corrector, separate, optional) |

From `model_mlx.py` (`DFlash2DecoderLayer`):
`x, k = attn_conv.prepare(ln1(x)); x = res + attn_conv.finish(attn(x, ctx), k); x, k = mlp_conv.prepare(ln2(x)); x = res + mlp_conv.finish(mlp(x), k)`,
where `conv(h, dyn, base)[t] = (base[0] + dyn[t,0]) * h[t] + (base[1] + dyn[t,1]) * h[t-1]` per group (zero before row 0 of the
block; it never reaches into context).

The selector only chooses which tokens are proposed; verify takes the target's greedy argmax, so the committed ids do not
depend on it (same argument as MTP and DCOCO). Host f32 is fine for it.

## 2. What the drafter needs from the target, and rollback (Q2)

- **Taps**: post-layer residual (pending add included) of layers 5, 19, 33, 47, 61 on every committed position (prefill
  chunks, decode, verify rows), as #158 does for dense layers. Two open points, to pin with the parity fixture (section 5):
  whether the ids mean "output of layer i" (as in v1) or `hidden_states[i]`, and whether the tap in `Model::run_layers`
  (`model.rs:2517-2523`) sees the right residual after a GDN block (its pending-add assumption was only exercised on dense
  layers). Only the last 2047 positions need draft K/V (sliding window), which bounds the prefill ingest cost.
- **The 27B supports it**: 64 layers = the draft's `num_target_layers 64`, hidden 5120 = draft hidden, vocab equal.
- **Rollback**: recurrent state cannot be truncated. `--spec mtp` on the hybrid 35B handles it with `SpecRows`: the verify
  branch of `gdn_block` saves the GDN state after each verify row, `SequenceState::spec_rollback(keep)` (`state.rs:353`)
  copies the kept row's state back (`docs/results/2026-10-22-mtp-spec-decode.md` 1.4). DFlash can reuse it unchanged at
  tp=1. Cost on the 27B: the GDN state is about 149 MiB per sequence, so 7 saved rows are about 1 GB written per pass
  (~2-4 ms at tp=1, half per rank at tp=2); the MTP doc's idea B (out-of-place state, zero-copy accept) would remove it later.
- **On HIP today**: `--spec mtp` loads on HIP at tp=1 (`load.rs:455`, "verify rows run the row-exact kernels"; HIP has
  `matmul_mxfp4_gemv_rows` / `matmul_mxfp4_pair_gemv_rows` rows-exact for m 1..16, `logits_bf16_gemv_rows`, per-row verify
  attention by default, GDN verify one row at a time with solo kernels). It has **never been run on HIP** (MTP doc, open
  items: "A Metal / HIP run"). Under tp>1 every speculative mode is refused (section 3).

## 3. What is missing on HIP (Q3)

No DFlash v1 piece needs a new HIP kernel: non-causal attention is per-row `AttnDecode` with `pos0` set
(`model.rs:3027`, backend-agnostic), the tap is an `Op::Add`, the draft's dense ops are bf16 ops HIP has, and HIP's
`verify_attn_per_row` is already true. Missing:

| piece | tp=1 | tp=2 |
|---|---|---|
| loader / model gates | `load.rs:379` (cpu, cuda only), `dflash.rs:106` ("dense Qwen3 only": gated attention refused), `dflash.rs:108-117` (draft geometry must equal target), `engine.rs:713` (`has_recurrent() && !mtp`) | plus `load.rs:448`, `model.rs` `load_tp` refusals, `enable_verify` (world == 1), `spec_rollback` (`peers` / `remote`) |
| DFlash2 drafter | conv op (cpu-ref + HIP kernel, quick + admission cases), sliding-window ctx, shared embed / head, 8-row layout, selector (host) | same, replicated on both ranks |
| top-16 per row | HIP has no `topk_rows`: host top-k over 7 x 248320 logits first (~7 MB download), kernel later if it shows | vocab-parallel head: each rank's half top-16, merged (TensorFold does the same with an all-gather) |
| multi-row verify | HIP rows kernels exist (above); 8-row bitwise vs solo to be proved at G1 | verify at tp=2 is new: per-peer `SpecRows` save and rollback on every rank, under the process-wide graph lock |

## 4. Upper bound (Q4)

**Radiance acceptance** (`/metrics` since the 04:55 restart, which is essentially the health smoke test, under its temp 0.7
sampling, so weak evidence): `draft_accepted_total 1955` / `draft_tokens_total 4284` = 0.456; 2603 decode tokens in 604 decode
steps = **4.31 tokens per step**; per-position acceptance 0.83, 0.81, 0.79, 0.78, 0.76, 0.73, 0.71. Every step is speculative
(`decode_rows_unspeculated_total 0`) and `decode_step_seconds_best` is **18.5 ms for an 8-row verify plus draft**. The journal
`done` lines carry no acceptance; `tools/radiance_ctx_stats.py` gives context lengths only. Per-request tok/s over 13,731
requests since 10-19 (gen >= 128): median 130 (prompt < 2k, mixed concurrency), 173 (2k-8k), 151 (8k-32k).
Other references: the DFlash2 card (H200, sampling) tau 4.1-5.5; TensorFold on one R9700 (ROCm, its own kernels) 44 -> 106
tok/s at a 4k prompt (2.4x) with tree drafts.

**Our verify cost** (no window spent in phase 0; measured numbers from `2026-10-25-mr-hip-27b-batch-profile.md`, 27B at
tp=1, graph-replayed `forward_batch`, the closest proxy for a verify pass): 1 row 30.3 ms, 4 rows 42.1 (1.39x), 6 rows 50.4
(1.66x), 8 rows **59.5 ms (1.96x)**. The MXFP4 rows GEMV is ALU-bound at 8 rows; the bf16 `lm_head` rows kernel is flat to
M = 8 (4.5 ms) and nearly doubles at 12, so we keep exactly 8 verify rows. At tp=2 no multi-row step was measured in isolation
(localeval 4 streams aggregate 122.4 tok/s, i.e. <= 32.7 ms per 4-row step including prefill interference).

**Projection, tp=2** (plain step 18.8 ms = 53.3 tok/s): verify8 ~36-38 ms (assuming the tp=1 ratio) + replicated bf16 drafter
~6-7 ms + GDN row saves ~2 ms + host top-k / selector ~1 ms = **~45-48 ms per round**.

| tau (tokens per round) | k = 7 | k = 3 (4 rows, ~32-36 ms per round) |
|---|---|---|
| 4.2 (radiance smoke) | ~90 tok/s (1.7x) | ~3.0 tokens: ~85 tok/s (1.6x) |
| 3.0 | ~65 (1.2x) | ~2.6: ~75 (1.4x) |
| 2.0 (prose-like) | ~43 (0.8x) | ~1.8: ~52 (1.0x) |

So the honest ceiling is **about 1.4-1.8x (75-95 tok/s) on high-acceptance text, not 145**: radiance's 8-row verify costs
about one of our plain steps. Not like-for-like either way: radiance samples at temp 0.7, uses a lossy all-reduce wire
(`--tp-wire wht6`) and fp8 KV. The remaining gap after this lane is the multi-row MXFP4 GEMV (8 rows at ~2x one row), a
separate kernel lane. Prose will likely fall below 0.99x at fixed k, so DFlash2 would stay opt-in (as on CUDA, DC section 12).

## 5. Phase 1 plan and estimate (if go)

| step | agent h |
|---|---|
| 1a DFlash2 drafter in core + cpu-ref: conv op, sliding window, shared embed / head, 8-row layout, host selector, geometry check relaxed to the draft's own heads | 8 |
| 1b hybrid target: taps on GDN / gated layers, engine guard, `DflashState` cut together with `SpecRows` rollback; cpu-ref tiny hybrid + random DFlash2 draft vs an f64 python reference | 6 |
| 1c draft-parity fixture: intarsia's tapped features fed to z-lab `model_mlx.py` on this Mac (target `~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`, draft from tatooine), per-slot drafts compared, one context > 2k (the tap mutant is caught here; ids gates cannot catch it) | 4 |
| 1d HIP tp=1: backend gate, conv kernel with quick + admission cases, host top-k; first window: G1 probe (8-row verify bitwise vs solo, verify ms vs plain, `--spec mtp` on the 27B as a second data point), ids gate at tp=1 | 6 |
| 1e tp=2: per-peer `SpecRows` and rollback, verify under the graph lock, drafter replicated on both ranks, vocab-half top-16 merge, refusals lifted, tp=2 mutants | 10-12 |
| 1f gates: G0 (8B, 35B; HIP, CUDA, Metal; short + 4k), ids 18 prompts x tp=1 / tp=2, mutant script with hashes, localeval x3 on `categories-rcp` and held-out + radiance reference, doc / LOG / tuning rows | 8-10 |
| **total** | **~42-46**, +8 contingency if 8-row verify is not bitwise at tp=2 or the tp=2 rollback needs ordering work |

Fleet: about **6-8 tatooine windows with radiance down** (G1 + tp=1 ids: 1; tp=2 bring-up: 2-3; gates: 1-2; speed: 1 of
about 90 min), each `--takes-down` radiance's role as the TP2 lane's scripts did, radiance brought back and checked before release.

Kill criteria for phase 1: G1 shows 8-row verify at tp=1 > 2.2x a plain step or not bitwise (stop and report, as DC's G1);
the parity fixture shows draft acceptance on our W4A4 target below tau 2.5 on the code / recall prompts.

## 6. Verdict

**Go, re-scoped**: the drafter exists for exactly this target (Apache-2.0, on tatooine), the reference code is MIT, and the
rollback machinery exists at tp=1. But this is a DFlash2 port (new drafter pieces), on a hybrid target (new tap / rollback
interplay), and the first speculative mode at tp>1 (a new subsystem, most of the risk), for an expected 1.4-1.8x at 1 stream,
not radiance's 145 tok/s. Decision needed from the coordinator: go on that basis, or stop at tp=1.

## 7. Window 1 (fleetq #5703, 2026-10-26 08:34:28 to 08:36:04 radiance down; `raw/w1/`)

Coordinator go 08:25 with the stop conditions: the 8-row verify at tp=1 must be ids-exact against plain steps and cost at most 2.2x
a plain step. Tree `a8046be` (origin/main `ec9e36b85` + this lane's probe and scripts, no engine change). Script `scripts/dfh/stages-w1.sh`.

**Exactness: PASS.** `tests/verify_rows_real.rs` (`Model::verify_check`, the lane dc method extended to recurrent states), 27B at tp=1,
card 0: every row count q = 2..8 at contexts 17, 130, 1024, 4000 and 16000 with random ids, plus 16 windows of the model's own greedy
continuation: **303 rows, 0 windows with a difference, worst |logit diff| 0, 0 argmax flips** (`raw/w1/mtest-probe-27b-tp1.log`).

**Cost: FAILS as measured, but the measurement compares an eager pass with a captured one.** Same test, length 1024, median of 12:

| pass | ms | vs decode1 |
|---|---|---|
| decode1 (`Model::forward`, one row, graph-replayed) | 30.95 | 1.00x |
| verify4 (`forward_verify`, eager) | 60.85 | 1.97x |
| verify8 (`forward_verify`, eager) | 85.39 | **2.76x** |

On HIP an eager pass pays the launch overhead the captured decode graph does not (the 27B's eager one-row step is 101 ms against 30.3
ms captured, `2026-10-25-mr-hip-27b-batch-profile.md`), so this is not the cost of a verify pass as the engine would run it. The engine
captures `--spec mtp`'s speculative step on HIP (`Model::spec_step`, MTP only; `gen-27b-mtp6-4k.err`: 66 graph replays, 8 captures),
but `--spec dflash` has no captured verify on any backend (on CUDA eager launches are cheap). The captured 8-row batch step of the same
model is 59.5 ms = 1.96x (`forward_batch`, MR doc); a captured verify8 is expected near that plus the GatedDeltaNet row saves (~2-4 ms),
i.e. about 2.0-2.1x. Not measured.

**First HIP run of `--spec mtp` (27B, tp=1, card 1, 256 greedy ids, chat template, `--ignore-eos`).** ids identical to `--spec off` in
4 of 4 runs:

| prompt | arm | decode tok/s | vs off | passes | accepted / proposed | tokens per pass | ms per pass |
|---|---|---|---|---|---|---|---|
| short | off | 33.23 | 1.00x | | | | 30.0 per step |
| short | mtp k=2 | 45.41 | 1.37x | 126 | 129 / 251 (51.4%) | 2.02 | 44.6 |
| short | mtp k=6 | 34.64 | 1.04x | 104 | 151 / 613 (24.6%) | 2.45 | 70.8 |
| 4k (`scripts/hip-decode/prompt4k.txt`) | off | 32.46 | 1.00x | | | | 30.7 per step |
| 4k | mtp k=2 | 56.04 | 1.73x | 98 | 157 / 196 (80.1%) | 2.60 | 46.4 |
| 4k | mtp k=6 | 50.79 | 1.56x | 66 | 189 / 392 (48.2%) | 3.86 | 76.1 |

A captured MTP pass with 7 verify rows plus 6 one-row head passes costs 76 ms (2.5x a plain step), which bounds a captured verify7 from
above. One prompt of each kind, so these are data points, not a benchmark.

**Status after window 1: stopped at the stop condition, decision asked of the coordinator.** Superseded by section 8: the w1 verify timings are profiled passes.

## 8. Captured verify pass, windows 2 and 3: the bar passes (captured verify8 = 1.98x)

Coordinator 08:41: option A (build a captured verify pass, generic for `--spec mtp` and `--spec dflash`; the 2.2x bar applies to the
captured number; exactness must hold with capture on; pause, not stop, if above 2.2x; MXFP4 kernels belong to lane mxr).

**What was built.** `Model::forward_verify` replays a captured verify-only graph where it can (`model/spec_graph.rs`:
`verify_graph_ok`, `verify_replay`, `capture_verify`; one recording per (sequence state, rows), ids from `ws.ids`, row positions from
`ws.pos`, the same launches as the eager pass). HIP only for now (CUDA and Metal keep the eager pass until measured), one rank, graphs
on; `INTARSIA_VERIFY_GRAPH=0` opts out. `--spec mtp`'s own captured step is unchanged; its eager fallback now takes the capture.

**Window 2 (#5709, 08:49:09-08:50:47) measured nothing new.** Its captured arm ran no capture (`spec_captures: 0`) and both arms
gave the w1 numbers (verify8 85.8 ms): the probe loads with `LoadOptions::default()`, whose `profile_every` is 1, so every verify pass
was profiled (eager, synchronised per op), and a profiled pass is never captured. The w1 2.76x is therefore a profiled pass too, not an
eager one. The engine runs with `--profile-every 0`. Fixed in the probe (`profile_every: 0`). w2 also re-ran the `--spec mtp` ids
check on the changed tree: identical to off in 4 of 4 (`raw/w2/steps.log`).

**Window 3 (#5711, 08:55:27-08:56:19 radiance down; `raw/w3/`), probe unprofiled, 27B tp=1:**

| rows q | captured verify ms (card 0) | vs decode1 | eager verify ms (card 1, `INTARSIA_VERIFY_GRAPH=0`) | vs decode1 |
|---|---|---|---|---|
| 1 (decode, graph) | 31.29 | 1.00x | 31.12 | 1.00x |
| 2 | 36.69 | 1.17x | 37.41 | 1.20x |
| 3 | 39.72 | 1.27x | 40.39 | 1.30x |
| 4 | 43.12 | 1.38x | 43.58 | 1.40x |
| 5 | 47.38 | 1.51x | 47.81 | 1.54x |
| 6 | 52.08 | 1.66x | 52.34 | 1.68x |
| 7 | 56.71 | 1.81x | 56.86 | 1.83x |
| 8 | **61.99** | **1.98x** | 61.85 | 1.99x |

- **Bar: PASS** (1.98x <= 2.2x). About 4.4 ms per extra row, the same slope as the captured batch step (MR doc: 59.5 ms at 8 rows).
- **Exactness with capture on: PASS**: 303 rows (q = 2..8 at contexts 17, 130, 1024, 4000, 16000, plus 16 greedy windows), 0 differences,
  worst |logit diff| 0, with 135 verify replays from 43 recordings. The eager arm: 163 rows, 0 differences.
- **Honest reading:** unprofiled, the eager verify pass on HIP is almost as fast as the captured one (0.2-2%, larger at small q). The
  capture is kept (bitwise, opt-out, small) but it is not what moved the number; removing the profiling did.

## 9. DFlash2 in the engine (tp=1), cpu gates and mutants

Coordinator 09:05: the captured verify pass is default-off (`INTARSIA_VERIFY_GRAPH=1`), every window's probe configuration is
dry-run in a cpu slot first. Pre-existing on origin/main (`ec9e36b85`, checked in a clean worktree): `cargo test -p
intarsia-backend-hip --test quick_coverage` fails on macOS with `moe_block_bf16_grouped` having no quick case (lane l7's kernel); not
this lane's, reported to the coordinator.

**Code.**
- `Op::DynConv` (ops.rs contract, cpu reference and fast kernels, admission sweep with the 27B draft's shape H 5120, G 320, T up to
  16) and two HIP kernels (`kernels/hip/dflash.hip` `dyn_conv_bf16`; `attn_decode_bf16_window`, `.ext()`, priority -1, so a call
  with `window == 0` never selects it; written by a sub-agent, reviewed here, compiled on tatooine, GPU admission in window 4).
- `model/dflash2.rs`: the draft with its own geometry (32 / 8 heads of 128 against the target's 24 / 4 of 256) and its own scratch,
  standard RMSNorm (the checkpoint's norm weights have means 0.43-2.65: `x * w`, not the target recipe's `x * (1 + w)`), its own rope
  theta (1e7, full rotary), both conv branches around attention and MLP, per-row sliding window (`window - 1 - r + block`, from the
  reference's rotating cache of `window - 1` keys and its mask), drafts from block rows 1.. . The taps, the context ingest
  scheduling, `DflashState` and its rollback are #158's. The candidate selector is not loaded yet (argmax drafts; it can change
  only which tokens are proposed).
- Loader: `DFlash2DraftModel` detected from `architectures`; DFlash2 allowed on cpu and HIP (DFlash v1 stays cpu and CUDA); engine:
  `--spec dflash` allowed on recurrent targets (the verify pass keeps `SpecRows`, `spec_cut` rolls back through them).
- Tap convention pinned from `model_mlx.py` `_LayerHook`: id `l` is the output of layer `l` (0-based), as in #158.

**cpu gates** (`crates/intarsia-models/tests/dflash2_cpu.rs`, tiny hybrid target `tests/fixtures/tiny_qwen3_5`, f32):

| test | result |
|---|---|
| taps vs HF's own `hidden_states` (layer 1 GatedDeltaNet, layer 3 gated attention) | rel L2 9.4e-7 / 1.5e-6; the neighbouring layers 0.46-0.82 |
| draft vs an f64 host implementation, context 3 and 13 (window 6: shorter and longer) | rel L2 <= 8.0e-7 every row |
| draft vs z-lab's `model_mlx.py` itself (MLX on CPU), through `open_dflash` (`tests/fixtures/tiny_dflash2`, `tools/dfh/dflash2_mlx_fixture.py`) | rel L2 <= 1.2e-6 every row (MLX on the Apple GPU differs from MLX on CPU by 2e-3 in f32, so the fixture is generated on CPU) |
| greedy ids `--spec dflash` vs off on the hybrid, k 1 and 3, random draft | identical (0 accepted) |
| scripted drafts (every acceptance pattern), odd `max_tokens`, follow-ups | identical; k 1: 143 of 338 accepted, k 3: 234 of 716 |

**Mutants** (`scripts/dfh/mutants.sh`, `raw/mutants/cpu-mutants.log`; the script refuses a mutant whose anchor does not match exactly
once or whose test binary hashes equal to the clean one; clean test binary sha256 `1b21dd79c6733189`):

| mutant | binary sha256 | caught by |
|---|---|---|
| `tap-off-by-one` (the draft reads layer l + 1) | `c3e03ca9157e62a1` | taps vs HF |
| `rollback-next-row` (recurrent rollback restores one row too late) | `62f61bd27b666918` | both ids gates |
| `window-off-by-one` | `7ab399d098f566a7` | host reference, MLX parity |
| `draft-norm-offset` (target's `1 + w` norms) | `d13ea3f63adbc6e0` | host reference, MLX parity |
| `selector-pred` (the candidate selector reads the wrong predecessor) and `tp-peer-no-rollback` (rank 1 keeps its recurrent state after a rejection) | rerun at the lane head with the rest: `raw/mutants/cpu-mutants-head.log` | see section 11 |

## 10. Windows 4 and 5: DFlash2 at tp=1 on HIP (exact; faster on code, slower on prose)

**Window 4** (#5726, 09:37:24-09:40:36 radiance down; `raw/w4/`), tree `26b2dcafc` (argmax drafts, no selector yet):
- 27B tp=1, 256 greedy ids, `--spec dflash --spec-k 7` vs off: **identical**, short and 4k (`gen-27b-*.ids`).
- HIP G0 (8B, 35B; short, 4k; plan show JSON and text): **SAME** for all 12 comparisons against origin/main `ec9e36b85`.
- GPU admission (`every_hip_kernel_matches_the_reference`) fails on **both** trees with the same set: 20 lines of `NO CASES (untested)` for
  10 wmma kernels (`matmul_bf16_wmma`, `matmul_fp8_wmma`, their `_rows`, the pair / fused-norm variants), present on main. The lane's two
  new kernels are admitted on the card (`dyn_conv_bf16`, `attn_decode_bf16_window`); the dev-vs-main diff of (kernel, verdict) is exactly
  those two added lines. Pre-existing, not this lane's (reported to the coordinator); rerun at the head in window 6.

**Window 5** (#5737, 09:55:49-10:23:00 radiance down; `raw/w5/`), tree `a4330ed0c` (selector on), 27B **tp=1** card 0, `serve
--max-batch 1`, localeval `quality --judge none`, 512 forced tokens (ignore_eos + min_tokens), one round, 21 runs, 0 problems (every row
512 tokens, same rows in every arm). Decode tok/s pooled per category, x vs off, tokens per verify pass from the server's counters:

| category | off | `--spec dflash` k7 | `--spec mtp` k2 |
|---|---|---|---|
| ho-chat | 33.0 | 37.9, 1.15x, 3.07 t/p | 51.5, 1.56x, 2.30 t/p |
| ho-code | 33.0 | 68.7, 2.08x, 5.58 t/p | 62.8, 1.90x, 2.80 t/p |
| ho-prose | 33.0 | 30.2, **0.92x**, 2.47 t/p | 46.4, 1.41x, 2.08 t/p |
| ho-recall | 33.0 | 43.0, 1.30x, 3.51 t/p | 52.1, 1.58x, 2.34 t/p |
| rcp-code | 33.0 | 62.7, 1.90x, 5.11 t/p | 61.0, 1.85x, 2.72 t/p |
| rcp-prose | 33.0 | 30.9, **0.94x**, 2.53 t/p | 46.4, 1.41x, 2.09 t/p |
| rcp-recall | 33.0 | 48.5, 1.47x, 3.94 t/p | 54.1, 1.64x, 2.43 t/p |

The w5 probe (`raw/w5/mtest-probe27.log`, 3000-token context): draft 23.5 ms for 7 drafts (0.77 of a plain step, 30.7 ms), of which
`matmul` 10.4 ms (56 calls), the draft's logits 7.8 ms (one `[8, V]` bf16 head), `attn_decode` 4.4 ms (40 per-row calls); the row-exact
verify8 65.1 ms (2.1x). A pass costs about 88.5 ms, so DFlash2 needs about 2.9 tokens per pass to break even: prose (2.5) loses.
**Reading:** DFlash2 is opt-in and stays so (prose < 0.99x). `--spec mtp` k2 already wins in every category at tp=1 (its one-layer draft
is cheap, 3 verify rows). The script ended with a bash syntax error after the last arm because `cat-window.sh` was edited while it ran
(bash reads a script as it goes); every run had finished and radiance came back through the EXIT trap (`raw/w5/guard.log`).

Coordinator 10:30 (decisions): (a) finish DFlash2 opt-in with its gates at tp=1 and tp=2, no draft-kernel tuning (follow-ups in
section 13); one DFlash speed round at tp=2 is enough. (b) `--spec mtp` under tensor parallelism is in scope and comes first.

## 11. `--spec mtp` under tensor parallelism (in-process ranks)

**Design.** Rank 0 drafts and verifies, every rank runs every pass on its shard, in lockstep (their all-reduces pair up):
- **The head is sharded like the model's layers** (`model/mtp.rs` `load_head`, `tp_world`): q / k / v / gate / up by rows, o / down by
  columns, `fc` and the norms replicated; each rank's MTP KV cache holds its own KV heads (`ldims`). The MoE head (35B) is refused under
  tp (its experts are not sharded).
- **Queue and catch-up on every rank**: a peer's plain step (graph replay or chunks, `run_peer_inner`) and its verify pass
  (`verify_peer`) queue their hidden rows into the peer's own `MtpState` exactly where rank 0 does (`forward_inner`,
  `forward_verify_tp`), so the catch-up passes (with their all-reduces) happen at the same points on all ranks. Peer states get their
  `MtpState` and `SpecRows` at `new_state`; the engine switches drafting on every rank (`m.active`).
- **Drafting** (`Model::mtp_draft_tp`): each rank runs `mtp_draft_rank` on a thread; the catch-up and every chained pass run on all
  ranks; rank 0 takes each draft's argmax and sends the id to the peers over a channel (they embed it for the next chained pass);
  `None` or a dropped channel ends the chain, an error aborts the communicator. Every rank's head cache returns to the committed length.
- **Draft head on rank 0 only**: the reduced head (`--mtp-draft-vocab`, 65536 for the 27B) is a row prefix of rank 0's own share of a
  vocab-parallel head when it fits (V / world = 124160 rows for the 27B); otherwise rank 0 keeps a whole copy of the head (as the DFlash
  drafter does). No collective in a draft's head.
- **Verify and rollback**: section 9's tensor-parallel verify (every rank saves its own recurrent rows; `spec_rollback` rolls back every
  rank, its MTP state included). The captured speculative step (`spec_step`) and verify graph stay single-rank (`tp.world == 1`), so
  under tp the draft and the verify pass run eagerly (tp=1 measured eager within 0.2-2% of captured, section 8).
- Loader (`crates/intarsia-models/src/load.rs`): `--spec mtp` with a rank in another process stays refused (`Group::Remote`); in-process
  groups load. `tests/tp_mtp_refused.rs` became `tests/tp_mtp.rs`.

**cpu gates** (`crates/intarsia-models/tests/tp_mtp.rs`, dense tiny hybrid `tests/fixtures/tiny_qwen3_5` + a random bf16 MTP head, f32):

| test | result |
|---|---|
| drafts at tp=2 vs tp=1 (3 prompts, after prefill and after two plain steps, k 3), replicated head and vocab-parallel head, draft vocab 0 (whole copy on rank 0) and 40 (inside rank 0's 48-row share) | identical in 12 of 12 |
| greedy ids `--spec mtp` vs off at tp=2, k 1 and 3 (head's drafts), k 2 and 3 scripted (all / some / none accepted), max_tokens 90, 1, 4, 7, both head layouts | identical; scripted k 3: 185 of 308 accepted |

Regression: `mtp`, `dflash2_cpu`, `tp`, `tp_hybrid`, `tp_vocab`, `tp_batch`, `spec`, `dflash_cpu`, `hybrid`, `engine` all pass.

**Mutants** (`TEST=tp_mtp scripts/dfh/mutants.sh ...`, `raw/mutants/mtp-tp-mutants.log`, clean test binary sha256 `aaf836b0997e058b`):

| mutant | binary sha256 | caught by |
|---|---|---|
| `tp-peer-no-rollback` (rank 1 keeps its state after a rejection) | `a827042ce24205c3` | `mtp_tp2_greedy_ids_equal_spec_off` |
| `mtp-peer-stale-chain` (a peer does not keep its chained row) | `46a6143c301cf033` | `mtp_tp2_drafts_equal_tp1_drafts` |
| `mtp-draft-head-slice` (rank 0 drafts over its vocab share instead of a whole head) | `7e234f401ae3902a` | `mtp_tp2_drafts_equal_tp1_drafts` |
| `mtp-peer-no-verify-queue` (peers never queue the verify pass's rows) | `2c330c2cad42feca` | `mtp_tp2_greedy_ids_equal_spec_off` |

## 12. Window 6: G0, admission and the 27B ids gate at the lane head

Window 6 (#5743, 10:45:44-11:00:53 radiance down; `raw/w6/`), tree `e745e9ea1`, `scripts/dfh/stages-w6.sh` (dry-run first, `#5742`).

- **HIP G0**, dev vs origin/main `ec9e36b85`, flags off: 8B and 35B, 256 greedy ids short and 4k, `plan show` JSON and text: **12 of
  12 SAME**. **CUDA G0** (spark1, `raw/cuda-g0/g0-head-bd51d5f.log`) and **Metal G0** (endor, `raw/metal/g0-head-bd51d5f.log`) at
  `bd51d5ffa`: 8 of 8 identical each. (Commits after `bd51d5ffa` touch only the spec-on loader gate and tests.)
- **HIP admission**: dev and main fail `every_hip_kernel_matches_the_reference` with the same set (10 wmma kernels, `NO CASES`); the
  (kernel, verdict) diff is the lane's two kernels, `ADMITTED` (`raw/w6/admission-dev-vs-main.diff`).
- **27B ids**, 18 prompts (`scripts/dfh/ids-prompts`, `scripts/dfh/mkprompts.py`: 6 short of 34-69 tokens, 6 of 2610-3381 tokens, 6 of
  10413-12552 tokens), 256 greedy ids, byte-identical `.ids` files against `--spec off` at the same tp:

| arm | tp=1 | tp=2 |
|---|---|---|
| `--spec dflash` k7 (DFlash2 draft) | **18 / 18** | **18 / 18** |
| `--spec mtp` k2 | **18 / 18** | not run: the engine loader's backend gate still refused it (`raw/w6/gen-tp2-short-0-mtp2.err`); the stage skipped the arm after the first failure |

  The refusal was a second check in `crates/intarsia-models/src/load.rs` (the spec / backend gate) that the cpu gates, which load
  through `Model::load_tp`, never reach; fixed in `3503c1a53` with a cpu test through `load_engine_tp`
  (`mtp_tp2_loads_through_the_engine_loader_and_matches_spec_off`).

**Acceptance per position** (`INTARSIA_SPEC_TRACE=1`, `tools/dfh/accept_table.py`, `raw/w6/acceptance.txt`; P(a pass accepts at
least i drafts)):

| tp | arm | prompts | passes | tokens / pass | i = 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dflash k7 | short | 427 | 3.57 | 0.77 | 0.55 | 0.40 | 0.28 | 0.22 | 0.19 | 0.16 |
| 1 | dflash k7 | ~3k | 416 | 3.68 | 0.80 | 0.60 | 0.44 | 0.33 | 0.24 | 0.16 | 0.11 |
| 1 | dflash k7 | ~11k | 456 | 3.35 | 0.79 | 0.54 | 0.34 | 0.26 | 0.19 | 0.14 | 0.09 |
| 1 | mtp k2 | short | 652 | 2.34 | 0.77 | 0.56 | | | | | |
| 1 | mtp k2 | ~3k | 613 | 2.49 | 0.85 | 0.64 | | | | | |
| 1 | mtp k2 | ~11k | 632 | 2.42 | 0.81 | 0.61 | | | | | |
| 2 | dflash k7 | short | 409 | 3.72 | 0.74 | 0.53 | 0.42 | 0.33 | 0.28 | 0.23 | 0.20 |
| 2 | dflash k7 | ~3k | 433 | 3.53 | 0.82 | 0.59 | 0.39 | 0.29 | 0.19 | 0.15 | 0.10 |
| 2 | dflash k7 | ~11k | 475 | 3.22 | 0.75 | 0.50 | 0.34 | 0.24 | 0.17 | 0.13 | 0.10 |

DFlash2's first draft is accepted about as often as MTP's (0.74-0.82 against 0.77-0.85) and as radiance's own counter on its
traffic (0.84 at position 0, `radiance:draft_position_acceptance_ratio`; radiance reports the ratio per position given the previous one,
which chains to about 0.84, 0.67, 0.54, 0.42, 0.32, 0.23, 0.16, i.e. about 4.2 tokens per pass, on other prompts, sampled at 0.7).
No sign of a defect in the draft path at position 1. The gap is deeper in the block. The radiance run on the same prompts (section 14)
tells whether this is the target or the drafter.

## 13. Window 7: `--spec mtp` at tp=2 on HIP (exact, run-to-run identical, faster in every category)

Window 7 (#5750, 11:37:09-12:04:50 radiance down; `raw/w7/`), tree `baaade4b9` (the loader gate fix included), `scripts/dfh/cat-window.sh`
`TP=2 PRE=ids IDARMS=mtp2 TPS=tp2 SOAK=40` (dry-run first, #5749).

- **ids**: `--spec mtp --spec-k 2` vs `--spec off` at tp=2, the 18 prompts of section 12, 256 greedy ids: **18 / 18 identical**
  (`raw/w7/ids/`).
- **Acceptance at tp=2 equals tp=1's** (`raw/w7/ids/acceptance.txt`), so the peers' graph-replayed steps hand the head the right
  hidden rows: P(first draft accepted) 0.79 / 0.80 / 0.84, tokens per pass 2.38 / 2.38 / 2.44 (short / ~3k / ~11k; tp=1: 0.77-0.85,
  2.34-2.49). 256-id generate runs: off 53.2 / 52.6 / 50.8 tok/s, mtp2 76.0 / 74.6 / 72.5 (1.43x / 1.42x / 1.43x); a pass costs
  31-34 ms (draft about 5 ms, verify + rollback about 27 ms) against a 19 ms plain step.
- **Soak** (`raw/w7/soak.txt`): 40 greedy chat requests to the mtp2 server (two prompts alternating, 256 forced tokens): one answer per
  prompt (sha256 `84ab57f73a70d547`, `cf7458e2cfd7a032`), the same as the off server's and the dfl7 server's.
- **Speed, one round** (early signal only; the bar is the interleaved x3 run of window 8), localeval `--judge none`, 512 forced tokens,
  21 runs, 0 problems (`raw/w7/cat-table.txt`):

| category | off | `--spec dflash` k7 | `--spec mtp` k2 |
|---|---|---|---|
| ho-chat | 53.3 | 51.7, 0.97x, 3.22 t/p | 74.5, 1.40x, 2.32 t/p |
| ho-code | 53.3 | 85.7, 1.61x, 5.32 t/p | 89.4, 1.68x, 2.78 t/p |
| ho-prose | 53.3 | 40.2, **0.75x**, 2.50 t/p | 66.1, 1.24x, 2.07 t/p |
| ho-recall | 53.2 | 57.7, 1.08x, 3.60 t/p | 77.7, 1.46x, 2.43 t/p |
| rcp-code | 53.3 | 83.7, 1.57x, 5.20 t/p | 87.0, 1.63x, 2.71 t/p |
| rcp-prose | 53.3 | 40.8, **0.77x**, 2.54 t/p | 67.3, 1.26x, 2.10 t/p |
| rcp-recall | 53.3 | 62.6, 1.18x, 3.89 t/p | 78.7, 1.48x, 2.46 t/p |

At tp=2 the DFlash2 draft (whole, on rank 0) costs about 24 ms, 1.3 plain steps, and its 8-row verify plus rollback about 43.6 ms
(2.3x the 19.1 ms step; tp=1 63.1 ms, 2.05x), so DFlash2 at tp=2 wins only on code. It stays opt-in. MTP k2 is the tp=2 speculator.
