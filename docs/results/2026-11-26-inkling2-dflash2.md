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

## Later messages from the coordinator

(none yet)
