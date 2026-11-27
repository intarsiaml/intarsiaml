# Inkling-2-Mini + DFlash 2 drafter on one R9700 (lane ink-dflash2, 2026-11-26)

Status: in progress. Branch `ink/dflash2` from origin/main `178d87ece`. Opt-in only; no default changes.

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

## Later messages from the coordinator

- **~19:10 EST, file ownership:** lane ink-hip-bpf (prefill overlap) now owns `kernels/hip/ink_mix.hip`, `kernels/hip/ink_attn.hip` and
  `kernels/hip/ink_route_fused.hip`. Do not edit them; tell the coordinator first if the DFlash2 path seems to need a change there.
  The batched-prefill / serve-scheduler code also stays with ink-hip-bpf. Everything else in the brief is unchanged.
