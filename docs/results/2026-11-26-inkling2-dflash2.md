# Inkling-2-Mini + DFlash 2 drafter on one R9700 (lane ink-dflash2, 2026-11-26)

Status: **k7: OPT-IN** by the pre-registered rule (faster on short / 1k / prose / chat, slower at 4k and 16k). **k4: GO** by the rule (update below, ABBA 4 + 4, 4k 0.991 [0.987-0.995]); k5: OPT-IN. Branch `ink/dflash2` from
origin/main `178d87ece` (main `a4c8f3b19` merged in after the measurements). Opt-in only; no engine default changes.

## Update 2026-11-26 21:00-21:55 EST: ABBA at k4 and k5 vs DSpark k5 (binary `dfl6` = the PR head)

**Verdict: k4 meets GO by the pre-registered rule; k5 and k7 are OPT-IN. No default changed (the coordinator asks the user).**

DFlash2 / DSpark k5 ratio per cell, geometric mean over (prompt, rep), 95 % CI = t over the rep pairs (`raw/speed.py`; k4 n = 4 pairs
in 2 windows, k5 n = 2 pairs in 1 window, k7 n = 4 from the earlier `dfl5` run, same card and prompts). One process at a time, card 0.

| DFlash2 / DSpark k5 | short | 1k | prose | chat | code (extra) | 4k | 16k | mean of the 6 cells (geo / arith) | prefill 4k / 16k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **k4** (n = 4) | **1.189** [1.184-1.194] | **1.142** [1.138-1.146] | **1.048** [1.046-1.050] | **1.216** [1.212-1.221] | 1.303 [1.299-1.306] | **0.991** [0.987-0.995] | **1.101** [1.098-1.104] | **1.112 / 1.115** | 0.996 [0.991-1.002] / 0.996 [0.990-1.001] |
| k5 (n = 2) | 1.206 [1.191-1.221] | 1.121 [1.116-1.127] | 1.053 [1.047-1.058] | 1.211 [1.198-1.224] | 1.432 [1.414-1.451] | **0.970** [0.967-0.973] | 1.039 [1.031-1.046] | 1.066 / 1.067 | 0.996 [0.991-1.002] / 0.998 [0.989-1.006] |
| k7 (n = 4, `dfl5`) | 1.190 [1.182-1.199] | 1.047 [1.041-1.052] | 1.045 [1.042-1.048] | 1.154 [1.146-1.161] | 1.584 [1.575-1.594] | **0.905** [0.901-0.909] | **0.950** [0.946-0.954] | 1.044 / 1.049 | 0.998 / 0.998 |

tok/s (median over the processes of the cell's geometric mean):

| | short | 1k | prose | chat | code | 4k | 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| DSpark k5 (k4 windows, 4 processes) | 109.35 | 101.83 | 120.85 | 100.18 | 127.16 | 116.84 | 90.79 |
| DFlash2 k4 | 130.19 | 116.39 | 126.68 | 121.96 | 165.80 | 115.93 | 100.04 |
| DSpark k5 (k5 window, 2 processes) | 109.41 | 101.77 | 120.84 | 100.21 | 127.21 | 116.85 | 90.79 |
| DFlash2 k5 | 131.94 | 114.12 | 127.20 | 121.37 | 182.21 | 113.33 | 94.30 |
| DFlash2 k7 (`dfl5`) | 130.13 | 106.54 | 126.34 | 115.56 | 201.41 | 105.85 | 86.29 |

Tokens per verify pass / ms per pass (decode ms / passes, pooled; `raw/abk4-analysis.txt`, `raw/abk5-analysis.txt`):

| | short | 1k | prose | chat | code | 4k | 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| tokens a pass: DSpark k5 / k4 / k5 / k7 | 2.66 / 3.32 / 3.57 / 3.92 | 2.68 / 3.19 / 3.37 / 3.58 | 3.06 / 3.33 / 3.61 / 4.04 | 2.40 / 3.16 / 3.36 / 3.61 | 3.19 / 4.36 / 5.19 / 6.46 | 3.22 / 3.23 / 3.43 / 3.64 | 2.47 / 3.00 / 3.07 / 3.23 |
| ms a pass: DSpark k5 / k4 / k5 / k7 | 27.08 / 25.85 / 27.81 / 31.43 | 28.44 / 27.21 / 29.38 / 33.41 | 29.36 / 26.51 / 28.76 / 32.47 | 26.29 / 25.91 / 27.87 / 31.47 | 25.69 / 26.32 / 28.61 / 32.07 | 29.46 / 27.84 / 30.41 / 34.39 | 29.23 / 29.99 / 32.58 / 37.38 |

**Rule at each k** (GO = mean over the six cells >= 1.03x, no cell more than 1 % slower where a 95 % CI upper bound < 0.99 counts as
slower, every md5 equal to `--spec off`):
- **k4: GO.** Mean 1.112 (geo) / 1.115 (arith). The weakest cell is 4k at 0.991 [0.987-0.995]: a 0.9 % loss by the point estimate, its CI upper
  bound 0.995 is not below 0.99, so it passes by the rule's own definition, but narrowly. Faster on the other five cells (+4.8 % to +21.6 %), 16k
  now +10.1 % (k7: -5.0 %). Ids equal across arms and reps (19 prompts x 8 processes) and md5 equal to `--spec off` (below).
- **k5: OPT-IN.** Mean 1.066 but 4k 0.970 [0.967-0.973] (upper bound < 0.99: more than 1 % slower). n = 2.
- **k7: OPT-IN** (earlier result; 4k 0.905 and 16k 0.950).
- The cost of lower k is code: k4 gives up 18 % of k7's code speed (165.8 vs 201.4 tok/s) while still +30 % over DSpark. The drafter's
  pass is cheaper at fewer rows, the context-dependent verify cost shrinks, and the 4k / 16k cells flip from loss to parity / gain.

**Greedy ids vs `--spec off` on `dfl6` (fleetq #12366, `raw/w4/`, `raw/window9-12366.log`):** `md5.json` (golden `code` 101 ids + the 4108-id
prompt), `--ignore-eos`: 256 tokens off = k4 = k5 = `380af8a5e897bdbed231818477038f37` (k4 66.7 % accepted, 3.64 tokens a pass; k5 61.0 %, 4.04);
64 tokens off = k4 = k5 = `f2e334dd8f1657b3ece2c7f8aa125deb` (90.7 % / 90.3 % accepted). The md5s are the same as the k7 ones on `dfl2` / `dfl4`.

**Default.** `--spec dflash` stays at k7 (block - 1) for this drafter. k4 is not an obviously better default on its own: it is better
on every cell of the rule but loses 18 % on code against k7, and a cell-dependent k (an adaptive k) would beat both. Whether to make k4
the opt-in default, or to propose DFlash2 k4 as the Inkling-2 HIP default over DSpark k5, is the user's call. `--spec-k 4` turns it on.

**Radiance down (this update):** window 6 #12344 21:03:22-21:07:05 (3 min 43 s, k4 r1 r2, A B B A), window 7 #12352 21:16:34-21:20:17 (3 min 43 s, k4
r3 r4, B A A B), window 8 #12358 21:39:34-21:43:17 (3 min 43 s, k5 r1 r2), window 9 #12366 21:52:04-21:53:19 (1 min 15 s, md5 gate k4 / k5). Total 12 min 24 s of
the 14 min 47 s (2 min 47 s left + the new 12 min); the lane total is 29 min 37 s of 32. Radiance answered after every window
(`/v1/models` checked, 15 s). The planned k5 pair r3 (`PAIR=1` in `raw/stage_w4.sh`) was not run: k5 had already failed 4k with a CI far from 0.99.

**Not verified:** k5 has n = 2 pairs; k4's 4k verdict rests on 4 pairs (CI half-width 0.4 %: process-to-process variance is tiny, but all
reps ran in one session on one card; a thermal / clock state shared by all would not show up); GO needs a decision on the 4k = 0.991
point estimate (a tighter margin than the 3 % mean). No md5 at 16k or with sampling. The ABBA binary `dfl6` is the PR head (main
`aee06201c` merged in); k7 numbers are from `dfl5` and were not re-run on `dfl6` (DSpark k5 numbers match across the two binaries within
0.1 %). 1k / prose / 16k at k4 have the combined ms a pass only, no draft / verify split. The verify-cost cause of the long-context loss
is still inferred, not profiled.

Logs: `raw/abk4/`, `raw/abk5/`, `raw/abk4-analysis.txt`, `raw/abk5-analysis.txt`, `raw/prefill.py`, `raw/stage_ab_k.sh`, `raw/stage_w4.sh`,
`raw/window6-12344.log` .. `raw/window9-12366.log`, `raw/w4/`; tatooine `~/intarsiaml-ink-dflash2/logs/{abk4,abk5,w4}`. fleetq: #12344, #12352,
#12358, #12366 (windows), #12340 (push).

## Brief (coordinator, 2026-11-26 ~18:55 EST; user approved "Yes, start the probe", incl. the 990 MB download)

Goal: find out whether the published DFlash 2 drafter for Inkling-2-Mini, `incoai/Inkling-2-Mini-DFlash2`, makes
single-stream greedy decode faster on one R9700 (tatooine, HIP) than today's default speculation (DSpark k5), with
greedy output ids identical to `--spec off`. Build it as an opt-in path, measure it, give a pre-registered GO / NO-GO,
open a PR, do not merge. No default changes.

Rules: `.worktrees/hip-swarm/COMMON-1126.md` (tatooine rules, window pattern, no `--mem`, radiance, shared files
add-only, push via fleetq on endor, report format). Exceptions: radiance budget **20 min total, <= 5 min per window**.
GPU windows `--host tatooine-r9700`, `HIP_VISIBLE_DEVICES=0`; lane ink-hip-bpf shares tatooine windows (fleetq
serializes). `env VAR=x cmd` in sh scripts, not `VAR=x func`. Window scripts trap and kill their own process tree.

Known (coordinator): drafter = `DFlash2DraftModel`, qwen3-style, 5 layers, hidden 2304, 32/8 heads, head_dim 128,
intermediate 6912, all sliding window 2048, `is_causal: false`, vocab 201024, untied; `dflash_config`: block_size 8,
conv_kernel_size 2, conv_group_size 16, selector_rank 256, selector_top_k 16, selector_softcap_mode "after_correction",
final_logit_softcapping 30.0, mask_token_id 200056, target_layer_ids [5, 23, 41, 48]. Card (SGLang H200, T=1.0,
7 drafts/pass) acceptance length DSpark vs DFlash2: GSM8K 3.52 / 4.96, HumanEval 3.66 / 4.80, MBPP 3.49 / 4.88,
MT-Bench 2.95 / 3.90; speedup over AR at c=1: DSpark 1.56-2.17x, DFlash2 2.03-2.77x. RAD (R9700, Q4, self-reported):
DSpark K6 148, DFlash2 K8 302 tok/s on code, prose 110.9. Our 27B precedent (LOG 2026-11-06 20:30): DFlash2 beat MTP
only on code/recall, lost on prose (0.84x) and chat (0.95x) -> include prose and chat cells.
Reusable: `model/dflash2.rs`, `model/dflash.rs` (27B), HIP DynConv / window attn / split-KV window / device top-k.
Inkling DSpark taps visits 5, 23, 41 (partial entering the visit) and 48 (final-norm output) in `model/inkling2.rs`.
Loader: Inkling `--spec dflash --spec-draft <dir>` -> `open_dspark` in `crates/intarsia-models/src/load.rs`.
Missing: `final_logit_softcapping` / `mask_token_id` handling in dflash*.rs. Verify rows <= 8 bitwise on HIP since #536.
Current speed (main 178d87ece, lane ink-hip-combined PR #539): spec default DSpark k5 short 109.57, 1k 102.01,
4k 117.12, 16k 91.01 tok/s; plain short 83.69, 4k 80.35, 16k 75.31.

Steps: 0 prior work; 1 download (one fleetq net,disk job, pin HF revision, LFS sha256; read
`sglang_inkling_dflash2.py` as reference only, never run); 2 build (route by config: `dflash_config` -> DFlash2,
`mtp_config` -> DSpark; feed taps; softcap + mask token; share target embed / lm_head if the reference does;
drafts per pass up to 7; unit tests vs CPU reference); 3 GPU probe (admission for new ops; greedy md5 == `--spec off`
at short and 4k, 64 and 256 tokens; speed DFlash2 vs DSpark k5, ABBA 4+4, 95 % CI of the ratio; cells short (12
golden), 1k, 4k, 16k, one prose, one chat; per cell tokens/pass, draft ms, verify ms, tok/s; if time, k 4..7 sweep).
Scope: Inkling on HIP only; Metal / CUDA / 35B / batched serve unchanged (refuse the drafter under `--max-batch > 1`).
Don't edit ink-hip-bpf's files (batched prefill, serve scheduler) without telling the coordinator.
Work folders: Mac `.worktrees/ink-dflash2`; tatooine `~/intarsiaml-ink-dflash2`, volume
`intarsia-cargo-target-ink-dflash2`. Don't delete folders.

## Pre-registered rule (written before any measurement)

- **GO** (for a later default-on proposal): DFlash2 at its best single k is >= 1.03x DSpark k5 on the mean over the
  cells, no cell more than 1 % slower (95 % CI upper bound of the ratio < 0.99 counts as slower), and every md5 equal
  to `--spec off`.
- **OPT-IN**: faster on some cells, slower on others.
- **NO-GO**: not faster anywhere. Keep the code if it is clean and opt-in, and say so.
- Sanity: if greedy acceptance on code comes out under ~3 tokens per pass, suspect a tap or mask bug before
  concluding anything.

## Measurement plan (pre-registered with the rule, before any GPU run)

- **Prompts** (`raw/speed.json`, md5 `7763e4744bb4abcfef7a19552708b9cf`; rebuilt because the canonical file lived in
  `~/intarsiaml-ink-hip-verify`, removed by #12258; the ink-sampled copy is a 6-prompt cut): 0 warm-up (excluded), 1-12 the 12
  greedy golden prompts (`golden/inkling2-mini-vllm-greedy.json`, checked equal), 13 / 14 / 15 the 1021 / 4108 / 16455-id prompts
  (ink-spec-k `longp.json`), 16-17 `long_prose_300` / `long_prose_700` (ink-spec-k `prose.json`), 18 `code_tests` (ink-spec-k
  `code.json`). 256 tokens, greedy, EOS honoured.
- **Cells** (`speed.py` classes): short = 1-12, 1k = 13, 4k = 14, 16k = 15, prose = 16-17, chat = 5-12 (the 8 `chat_*` golden
  prompts). The rule's cells are these six. Code = 3 + 18 is reported as an extra cell (not in the rule).
- **Arms, same binary and card 0:** DSpark = no `--spec` flags (the recipe default, DSpark k5, cut 0.5); DFlash2 = `--spec dflash
  --spec-draft /models/Inkling-2-Mini-DFlash2 --spec-k K` with K chosen as below. ABBA 4 + 4 (two windows of A B B A / B A A B),
  ratio DFlash2 / DSpark per (prompt, rep), geometric mean per cell, 95 % CI = t-interval over the 4 rep pairs (lane
  ink-hip-verify's `speed.py`). Prefill 4k / 16k from the same processes.
- **k choice (deviation from "best single k", the window budget does not hold a k 4..7 sweep next to ABBA 4 + 4):** window 1 traces
  DFlash2 k7 (`INTARSIA_SPEC_TRACE=1`: accepted per pass, draft ms, verify ms) and the default; the predicted best k is the one
  maximising mean(min(accepted, k) + 1) / (draft ms + verify ms(k + 1 rows)) over the traced passes, verify ms per row count from the
  DSpark trace (2..6 rows) and the DFlash2 trace (8 rows), 7 rows interpolated. ABBA runs that k only. A k sweep runs only if a
  window is left.
- **Gates:** greedy ids (`raw/md5.json`: golden `code`, 101 ids, and the 4108-id prompt; `--ignore-eos`, 64 and 256 tokens)
  `--spec off` = DFlash2 k7; every ABBA process's ids equal across arms and reps; HIP tiny suite `tests/inkling2_dflash2.rs`;
  `kernels test --op dyn_conv` at H 2304 (no op added or changed).
- **Low acceptance:** under ~3 tokens per pass on code means a tap / embedding / mask bug first; window 1 also runs
  `INTARSIA_DFLASH2_EMBED_NORM=1` (Inkling's `embed_norm` on the block rows) and `INTARSIA_DFLASH2_HEAD=mxfp4` (a quantised draft
  head) on the variant set for tokens per pass and tok/s.

## Download (fleetq #12261)

`/data/models/intarsia/Inkling-2-Mini-DFlash2` on tatooine (mounted as `/models/Inkling-2-Mini-DFlash2`), HF revision
`4f9088ea66c21f2b99ed657d1941c306b03e84d2` (in `REVISION`), `model.safetensors` 989,975,616 bytes, sha256
`217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee` = the HF LFS sha256. Files read-only. `raw/download.sh`.
`sglang_inkling_dflash2.py` read as text only (never run). What it says about Inkling:
- taps: `set_dflash_layers_to_capture` is an alias of `set_dspark_layers_to_capture` ("both drafts use native post-residual visit
  features and final norm"), ids sorted, unique, ending at `num_layer_visits` (48). So the DFlash2 taps are the DSpark taps.
- vocabulary: candidate logits over the real `vocab_size` (200058) rows of the target head; 201024 is padded storage.
- softcap: `selector_softcap_mode = "after_correction"`: no softcap on the unary logits, `cap * tanh(score / cap)` on the selector's
  path scores (`_score_edges` output). Monotonic, so greedy picks equal the uncapped ones except f32 ties.
- the draft has no head and no embedding of its own: it uses the target's (the worker points `lm_head` at the target's).
  The file does not show the embedding call; window 1 settled it (below).

## Window 1 (fleetq #12276, 19:24:25-19:27:58 EST, radiance down 3 min 33 s), binary `dfl2`

`raw/stage_w1.sh`, `raw/window1-12276.log`, `raw/w1/`.
- **Raw `embed[id]` block rows (the generic DFlash path): 0 of 616 drafts accepted** (k7, trace set; 1.00 tokens a pass, 70-75
  tok/s). With `INTARSIA_DFLASH2_EMBED_NORM=1` (block rows `rms(embed[id], embed_norm)`, the target's embedding output): **51.4 %,
  4.57 tokens a pass, 139.1 tok/s** on the variant set (code 6.07 tokens a pass 184.3 tok/s, code_tests 7.08 / 214.2, prose_300
  3.34 / 101.3, warm 3.08 / 95.0). So the reference embeds with the target's embedding module, which includes `embed_norm`.
  Fixed on the branch: embed_norm is the default for Inkling DFlash2 drafts (`INTARSIA_DFLASH2_EMBED_NORM=0` = raw rows).
- md5 (off = dfl k7, 64 and 256 tokens) was equal, **but does not count**: with nothing accepted every token was a verify row's own
  argmax. Redone in window 2.
- The k-choice trace is void (zero acceptance). Its timing still holds: DFlash2 draft 4.9-5.5 ms a pass (flush 0.4, block 2.36,
  bf16 head 1.80, download + select 0.25); verify + rollback by rows 2 / 3 / 4 / 5 / 6 / 8 = 15.29 / 17.71 / 19.35 / 22.41 /
  24.85 / 28.49 ms (`raw/trace.py`); DSpark draft 2.9-3.7 ms.
- `INTARSIA_DFLASH2_HEAD=mxfp4` ran with the raw rows (0 accepted): uninformative, redone in window 2.
- Admission: `kernels test --backend hip --op dyn_conv --model /models/Inkling-2-Mini --quick`: dyn_conv_bf16 2 cases ADMITTED
  (max_abs 0). No op is added or changed by this lane.
- HIP tiny suite: `timeout 30` killed it after one test (it rebuilt test deps first). Redone in window 2 from the built executable.
- Script bug: `stage_w1.sh` named a function `tr`, which the admission / tiny lines' `tr '\n' ' '` called: two extra DSpark trace
  runs (about 45 s of the window). Renamed in later scripts.

## Window 2 (fleetq #12284, 19:44:04-19:47:08 EST, radiance down 3 min 4 s), binary `dfl4` (embed_norm on by default)

`raw/stage_w2.sh`, `raw/window2-12284.log`, `raw/w2/`.
- **Greedy ids gate, now with drafts accepted:** `raw/md5.json` (golden `code` 101 ids + the 4108-id prompt), `--ignore-eos`:
  64 tokens off `f2e334dd8f1657b3ece2c7f8aa125deb` = DFlash2 k7 `f2e334dd...` (88.4 % accepted, 6.63 tokens a pass); 256 tokens off
  `380af8a5e897bdbed231818477038f37` = DFlash2 k7 `380af8a5...` (52.4 %, 4.64 tokens a pass). Equal; the off md5s also equal
  window 1's (binary `dfl2`). 8-row verify passes are exercised (accepted 7 of 7 in many passes).
- **k7 trace** (`INTARSIA_SPEC_TRACE=1`, `raw/trace.json`): 294 passes, 47.0 %, 4.27 tokens a pass. Per prompt tokens a pass:
  warm 3.08, golden code 6.07, chat_system 3.56, 4k 3.75, prose_300 3.34, code_tests 7.08 (DSpark k5 trace, window 1: 2.11 / 3.22 /
  2.24 / 3.22 / 2.62 / 3.16). Draft 4.9-5.4 ms a pass vs DSpark 2.9-3.7. Verify + rollback by rows 2..8: 15.33 / 17.75 / 19.35 /
  22.41 / 24.85 / 28.67 / 28.41 ms.
- **k choice by the pre-registered predictor** (`raw/kchoice.py`, pooled over the 294 passes): k1..k7 = 87.4 / 105.3 / 119.0 /
  120.4 / 122.4 / 118.1 / **127.8** tok/s (geomean over prompts 88.4 / 107.2 / 121.4 / 122.4 / 123.8 / 118.6 / 127.3). **k7.** (The
  predictor reproduces the traced k7 rates: code 182.3 predicted vs 184.1 traced.) Per prompt the best k differs: k3 for warm /
  chat / 4k / prose (by 1-10 %), k7 for code.
- **Draft head:** `INTARSIA_DFLASH2_HEAD=mxfp4` 142.24 tok/s, 4.55 tokens a pass vs `bf16` 138.91, 4.57 on `raw/var.json` (one
  process each). Chosen: mxfp4, now the default on HIP for this draft (`bf16` = the target's head, as the reference).
- HIP tiny suite (built test executable, `INTARSIA_INKLING2_BACKEND=hip`): `greedy_ids_equal_spec_off` ok (k 1 / 3 / 7, 4 prompts,
  1-40 tokens, ids = spec off on HIP); `runs_with_admitted_kernels_only` did not finish inside the 85 s timeout (it admits every
  kernel with `no_cache`; it passes on cpu). Not rerun (radiance budget).

## Pre-registered amendment (written after window 2, before any ABBA process)

- ABBA DFlash2 arm: binary `dfl5` (branch head: embed_norm on, mxfp4 draft head on HIP), `--spec dflash --spec-draft
  /models/Inkling-2-Mini-DFlash2 --spec-k 7`, no env. DSpark arm: the same binary, no `--spec` flags. Window 3 = A B B A (reps
  r1 r2), window 4 = B A A B (reps r3 r4). The rule and the cells are unchanged. No k sweep (radiance budget: 6 min 37 s used of 20).

## Speed: DFlash2 k7 vs DSpark k5, ABBA 4 + 4 (windows 3 and 4)

Windows: 3 = fleetq #12297 (19:58:39-20:02:24 EST, radiance down 3 min 45 s, A B B A, reps r1 r2); 4 = #12305 (20:10:15-20:14:00,
3 min 45 s, B A A B, reps r3 r4). Card 0, one process at a time, binary `dfl5` (md5 `c00616e706713da142eade4c41142e57`, the branch
before the main merge). Arms: `dspark` = no `--spec` flags (main's default, DSpark k5 cut 0.5); `dfl` = `--spec dflash --spec-draft
/models/Inkling-2-Mini-DFlash2 --spec-k 7` (embed_norm, mxfp4 draft head: the branch defaults). `raw/speed.json`, 256 tokens,
greedy, EOS honoured. Ratio per (prompt, rep), geometric mean per cell, 95 % CI = t over the 4 rep pairs (`raw/speed.py`, output
`raw/speed-analysis.txt`, logs `raw/ab/`).

| | DSpark k5 (main's default), tok/s | DFlash2 k7, tok/s | change | DFlash2 / DSpark, 95 % CI |
|---|---:|---:|---:|---:|
| prefill 4k (4108 ids) | 3420 | 3418 | -0.2 % | 0.998 [0.992-1.004] |
| prefill 16k (16455 ids) | 3232 | 3226 | -0.3 % | 0.998 [0.991-1.004] |
| spec decode, short (12 golden) | 109.40 | 130.13 | **+19.0 %** | **1.190** [1.182-1.199] |
| spec decode, 1k | 101.84 | 106.54 | **+4.7 %** | **1.047** [1.041-1.052] |
| spec decode, 4k | 116.95 | 105.85 | **-9.5 %** | **0.905** [0.901-0.909] |
| spec decode, 16k | 90.87 | 86.29 | **-5.0 %** | **0.950** [0.946-0.954] |
| spec decode, prose (long_prose_300 / 700) | 120.94 | 126.34 | **+4.5 %** | **1.045** [1.042-1.048] |
| spec decode, chat (8 chat golden) | 100.22 | 115.56 | **+15.4 %** | **1.154** [1.146-1.161] |
| extra (not in the rule): code (golden code + code_tests) | 127.19 | 201.41 | **+58.4 %** | **1.584** [1.575-1.594] |

tok/s = median over the 4 processes of the cell's geometric mean. The DSpark arm reproduces main's numbers (lane ink-hip-combined
on `178d87ece`: short 109.57, 1k 102.01, 4k 117.12, 16k 91.01). Prefill: same processes, ratio per rep.

| per cell | short | 1k | 4k | 16k | prose | chat | code |
|---|---:|---:|---:|---:|---:|---:|---:|
| tokens per pass, DSpark k5 | 2.66 | 2.68 | 3.22 | 2.47 | 3.06 | 2.40 | 3.19 |
| tokens per pass, DFlash2 k7 | 3.92 | 3.58 | 3.64 | 3.23 | 4.04 | 3.61 | 6.46 |
| decode ms per verify pass, DSpark k5 | 27.07 | 28.44 | 29.44 | 29.22 | 29.35 | 26.29 | 25.69 |
| decode ms per verify pass, DFlash2 k7 | 31.43 | 33.41 | 34.39 | 37.38 | 32.47 | 31.47 | 32.07 |

(ms per pass = decode ms / verify passes, pooled over the 4 processes, `raw/perpass.py`.) The draft / verify split per pass is from
the traces (window 2 DFlash2 k7, window 1 DSpark k5; traced runs, a few % slower): DFlash2 draft 4.9-5.4 ms (block 2.4, head 1.8 bf16
/ about 0.5 less with mxfp4), verify + rollback at 8 rows 28.4 ms; DSpark draft 2.9-3.7 ms, verify at its 2-6 rows 15.3-24.9 ms.

Every process's ids are identical across arms and reps (19 prompts x 8 processes), and prompts 1-15 equal main `178d87ece`'s
spec-default ids (lane ink-hip-combined `raw/c1/speed/new@c0a.err`; prompt 0, the warm-up, is a different prompt there).

**Why long contexts lose (measured: the per-pass gap; inferred, not profiled: its cause):** DFlash2 accepts more at every length
(3.64 vs 3.22 at 4k, 3.23 vs 2.47 at 16k), but its passes cost 4-8 ms more, and the gap grows with context (+4.4 ms short, +5.0 ms
4k, +8.2 ms 16k). On HIP the verify rows run the row-exact kernels with one attention decode call per row (the load line says so;
bitwise the single-row decode), so an 8-row verify most likely reads the whole context 8 times against DSpark's 3-4;
the draft's own attention is windowed (2048) and does not grow. A batched, still-bitwise multi-row verify attention is the lever
that would decide 4k / 16k; it belongs to the HIP verify lanes (ink-hip-spec / ink-hip-vrow ownership), not this one.

## Verdict against the pre-registered rule

Radiance down time: window 1 3 min 33 s, 2 3 min 4 s, 3 3 min 45 s, 4 3 min 45 s, 5 3 min 6 s; total 17 min 13 s of the 20 min
budget, every window under 5 min.

- GO needs: best single k >= 1.03x on the mean over the cells (k7: geomean of the six cells 1.044, arithmetic 1.049: met), no cell
  more than 1 % slower (4k CI upper 0.909, 16k CI upper 0.954: **failed**), every md5 equal (met).
- **OPT-IN**: faster on short (+19.0 %), 1k (+4.7 %), prose (+4.5 %) and chat (+15.4 %), slower on 4k (-9.5 %) and 16k (-5.0 %).
- k: k7 by the pre-registered predictor (window 2). Per prompt the predictor puts k3 ahead on warm / chat / 4k / prose; it never
  puts any k above DSpark at 4k (k3 predicted 113.0 vs DSpark 117.6 traced), so no single k turns 4k around.
- Card comparison (tokens per pass; the card is SGLang, H200, T = 1.0, 7 drafts; ours greedy): card DFlash2 GSM8K 4.96, HumanEval
  4.80, MBPP 4.88, MT-Bench 3.90; ours code 6.46, chat 3.61, prose 4.04, short 3.92. Card DSpark 3.52 / 3.66 / 3.49 / 2.95; ours
  (k5, cut 0.5) 2.40-3.22. The card's 1.3-1.4x acceptance gain over DSpark holds here (1.31-2.02x); its speedup does not carry to
  long contexts on this card. Like RAD's R9700 report and our 27B (LOG 2026-11-06 20:30), code gains most; unlike both, prose and
  chat gain here too (prose +4.5 %, chat +15.4 %).

## Window 5: k sweep on short + 4k (fleetq #12312, 20:20:41-20:23:47 EST, radiance down 3 min 6 s), informational

`raw/stage_k.sh`, `raw/ksweep.json` (warm + 12 golden + 4108 ids), one process per arm, binary `dfl5`, `raw/k/`,
`raw/ksweep-analysis.txt`. Not part of the rule (the ABBA's k7 was fixed before it ran); n = 1 per arm, no CI. The second DSpark
process was skipped at the time limit.

| DFlash2 / DSpark k5 (same window) | short | 4k | chat (8 of short) | golden code |
|---|---:|---:|---:|---:|
| k7 | 1.187 | 0.903 | 1.151 | 1.473 |
| k6 | **1.208** | 0.932 | 1.190 | 1.417 |
| k5 | 1.201 | 0.965 | 1.207 | 1.338 |
| k4 | 1.183 | **0.987** | **1.211** | 1.230 |
| tokens per pass k7 / k6 / k5 / k4 | 3.92 / 3.80 / 3.57 / 3.32 | 3.64 / 3.54 / 3.43 / 3.23 | 3.61 / 3.54 / 3.36 / 3.16 | 6.07 / 5.54 / 4.90 / 4.18 |

The k7 row reproduces the ABBA (short 1.187 vs 1.190, 4k 0.903 vs 0.905). Lower k trades code speed for long-context parity: at
4k, k4 gives back most of the loss (0.987, still 1.3 % slower by its point estimate), and short / chat stay +18-21 %. The
pre-registered predictor missed this. It priced verify rows from a table pooled over all contexts, while at 4k an extra row costs
more. Not measured: 16k, 1k and prose at k4 / k5. So whether k4 or k5 would pass the GO rule is open. The radiance budget left
(2 min 47 s) does not hold the ABBA 4 + 4 that would decide it. The next window for this drafter would run ABBA k4 (and k5) vs
DSpark k5 on all six cells.

## Checks the brief asked for

- **Selector used when drafting:** yes. Every traced pass logs `selproj` and `select` (`raw/w2/trace-dfl.err`), and
  `draft_pass_matches_host_reference` checks the device drafts against a host selector walk (>= 4 picks) on the tiny fixture.
  The header comment that said "not loaded yet" is fixed.
- **Tap quantity:** the reference aliases the DFlash capture to the DSpark capture ("native post-residual visit features and final
  norm"). `taps_equal_the_dspark_drafts_taps` checks the DFlash2 taps bitwise against DSpark's. `taps_are_the_visit_partials_and_the_final_norm`
  checks them bitwise against `ink_dump`'s `vNN.partial_out` / `final_hidden`.
- **8-row verify bitwise under the DFlash path:** greedy md5 off = DFlash2 k7 (window 2) across many passes that accepted all 7
  drafts (an 8-row verify).
- **Which binary each gate covered:**
  - The formal off-vs-DFlash2 md5 gate ran on `dfl4` (bf16 draft head).
  - `dfl5`, the ABBA binary with the mxfp4 draft head, is covered by the ABBA ids equality. DFlash2 = DSpark on all 19 prompts x 8 processes, and both equal main `178d87ece`'s spec default on prompts 1-15.
  - The PR head (main merged in) has CPU tests and a HIP compile check, no GPU run.
- **Draft vs verify split:** from the traces only (warm, code, chat_system, 4k, prose_300, code_tests). 1k and 16k have the combined
  ms per pass only.
- **Scope:**
  - Inkling DFlash2 is refused on Metal and CUDA (`model.rs`).
  - `--max-batch` > 1 refuses any `--spec dflash` draft, unchanged (`load.rs`).
  - DSpark is unchanged: same tap buffer, same default.
  - No HIP kernel or op added or changed, and no file owned by ink-hip-bpf touched.

## fleetq ids

- Download: #12261.
- Builds: #12268 (seed + dfl1), #12273 (dfl2), #12279 (dfl3), #12282 (dfl4), #12292 (dfl5), #12324 (dfl6 = PR head, compile check).
- GPU windows: #12276, #12284, #12297, #12305, #12312.
- CPU tests on endor: #12269, #12286, #12306, #12330.
- Push: #12318 and later.

## Later messages from the coordinator

- **20:37 EST, PARK NOW:** the user restarts Claude Desktop in about 10 min. Let #12324 finish only if under 3 min are left. Queue
  nothing new. Bring radiance back if it is down. Commit locally (no push needed). Add a "Parked 2026-11-26 20:40" section. Reply in
  10 lines or fewer.

- **~19:10 EST, file ownership:** lane ink-hip-bpf (prefill overlap) now owns `kernels/hip/ink_mix.hip`, `kernels/hip/ink_attn.hip` and
  `kernels/hip/ink_route_fused.hip`. Do not edit them; tell the coordinator first if the DFlash2 path seems to need a change there.
  The batched-prefill / serve-scheduler code also stays with ink-hip-bpf. Everything else in the brief is unchanged.

## Parked 2026-11-26 20:40

- **Head:** `ink/dflash2` at the commit that adds this section. Its parent `d1c01cb34` is code-identical: main `aee06201c` merged in, then
  docs-only commits. Local commits after the 20:16 push of PR #541 are not pushed: the window-5 doc, the LOG wording, the main merge
  and the checks / fleetq-id sections.
- **Radiance:** up (rank0 and rank1, `fleetq status` 20:38). No job of this lane is running or queued.
- **Passed:**
  - **Windows:** #12276 (window 1: admission dyn_conv ADMITTED; embed_norm finding), #12284 (window 2: md5 off = DFlash2 k7 at
    64 / 256 tokens, `f2e334dd...` / `380af8a5...`, with 88 % / 52 % accepted; HIP tiny `greedy_ids_equal_spec_off` ok),
    #12297 + #12305 (ABBA 4 + 4: short 1.190, 1k 1.047, 4k 0.905, 16k 0.950, prose 1.045, chat 1.154, code 1.584, prefill
    0.998, ids equal across all 8 processes and to main), #12312 (k sweep, n = 1).
  - **CPU tests:** round 4 #12330 (after the aee06201c merge), all pass: core lib 275, models lib 80, inkling2_dflash2 8,
    dflash2_cpu 13, dflash_cpu 7, inkling2_dspark 4, inkling2_dspark_cut 1, recipes_doc 3, docs_env_vars 2. Earlier rounds:
    #12269, #12286, #12306.
  - **HIP build of the merged head:** #12324, `dfl6` md5 `689cde21f795f1e38ba968c25d06daea`, cli + test build exit 0.
- **Not verified:**
  - No GPU run of the merged head: the ABBA binary is `dfl5`, the branch before the merges.
  - `runs_with_admitted_kernels_only` on HIP timed out at 85 s (passes on cpu).
  - No ABBA at k4 / k5.
  - 1k, 16k and prose untested below k7.
- **Logs:**
  - Mac: `docs/results/2026-11-26-inkling2-dflash2/raw/` (window1-12276.log .. window5-12312.log, w1/, w2/, ab/, k/, build-dfl*.log,
    cpu-tests-1..4.log, speed-analysis.txt, ksweep-analysis.txt).
  - tatooine: `~/intarsiaml-ink-dflash2/logs/` (window-1126-*.log, w1, w2, ab, k, build logs); binaries `.scratch/bin/intarsia-dfl1..6`
    (folder 2.4 GB in all); volume `intarsia-cargo-target-ink-dflash2` (1.58 GB); drafter `/data/models/intarsia/Inkling-2-Mini-DFlash2`
    (945 MB). No containers left. Radiance-down total 17 min 13 s of 20.
- **Next step:**
  1. Push `ink/dflash2`: `fleetq run --host endor --load cpu --expect 6 -- git -C /opt/code/intarsiaml/.worktrees/ink-dflash2 push`.
  2. `gh pr edit 541 --body-file` with the k-sweep table. The draft body was in the session scratchpad and is lost at the restart.
  3. The speed windows of this brief are done (ABBA 4 + 4 at k7, verdict OPT-IN). The open speed work is an ABBA 4 + 4 at k4 and k5
     vs DSpark k5 on all six cells, about 2 windows of 3.75 min each per k. That needs a new radiance budget, since 2 min 47 s are
     left of this lane's 20.

- **~21:00 EST, RESUME (coordinator brief to the new agent):**
  1. Push `ink/dflash2` from endor via fleetq. 2. Rewrite the PR #541 body (what it adds, speed table + k sweep, bitwise results,
  embed_norm finding, how to turn it on, not verified; ends with the Claude Code line). 3. New radiance budget **12 min on top of
  the 2 min 47 s left (about 14 min 45 s)**: ABBA 4 + 4 at k4 vs DSpark k5 on all six cells with `dfl6`, 2 windows of about 3.75 min;
  then k5 with what is left (ABBA 2 + 2 is OK, state n). Same rule. If a k meets GO, change no default, report it (the coordinator
  asks the user). If the best k is an obvious better default for `--spec dflash` itself (opt-in), it may be set, with a test.
  4. Results in this doc (speed table first), LOG.md row, unanimis (leave pending edit `801edc93`; new capture for the k4 / k5
  result; no `project` / `scope`, "intarsiaml" in source). 5. Push again, update the PR body, do not merge. Windows: `--host
  tatooine-r9700 --takes-down radiance --quiet gpu,cpu`, each <= 5 min, none started after 06:00 EST. Never `fleetq release` a
  running window job; abort = kill the driver's process tree and confirm `curl -s http://tatooine:8080/v1/models` answers; confirm
  radiance after every window. Merge origin/main, never rebase. Waits <= 4 min per call.
