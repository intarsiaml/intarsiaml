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
