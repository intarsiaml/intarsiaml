# 27B DFlash2 draft: split-KV windowed attention (lane dw, 2026-11-06)

Lane dw, branch `lane-dw/draft-win-split` from main `3f46da4f7` (includes #323, lane dh). Follow-up to lane dh's item 1
(`docs/results/2026-11-06-27b-dflash-hip.md`): with `INTARSIA_DFLASH_TP=1` the draft missed its 4k rule (0.717x against <= 0.70x) because the
context-dependent part of the draft block, 3.20 ms (4k minus short-0), did not shrink when the heads were halved. Cause: past its 2048-key window
the draft's attention runs `attn_decode_bf16_window`, one block per head, a serial 2048-key scan, 5 layers x 8 rows = 40 launches a block pass.

Tatooine: lane dh's tree `~/intarsiaml-dh` and volume `intarsia-cargo-dh` only. Scripts: `scripts/dw/` (copies of `scripts/dh/`, `.dw/` scratch is git-ignored).

## Decisions and go-aheads received by message

* User decision 2026-11-06, "keep bitwise": the target model's verify and plain decode stay bit-identical; only the DFlash2 draft's attention may change bits.
  `attn_decode_bf16_window` is also the Gemma sliding-layer kernel (`model.rs:1308`), so it is left exactly as it is; the draft gets a separate kernel
  behind a separate attribute.
* Ship as opt-in behind `INTARSIA_DFLASH_WIN_SPLIT=1`; a default flip is the user's call (numbers only).
* Coordinator, mid-task: lane dm opens a small PR changing `dflash_dry_run` (`crates/intarsia-core/src/model/dflash.rs`) to record context
  ingest at every M from 1 to 8 (a Metal serve fix). This lane does not edit that function (its changes: `ops.rs`, `model/dflash2.rs`,
  `admission.rs`, the HIP backend, one new kernel file, `docs/tuning.md`), so the hunks stay separate. Merge `origin/main` into the branch once that
  PR lands, and note it here.
* Standing approval to take radiance down via fleetq `--takes-down radiance`, windows <= 35 min.

## What changes (design, written before measuring)

* `OpAttrs::win_split` / `ShapeInfo::win_split` (default false). Only the DFlash2 draft's block attention sets it (`d2.win_split`, from
  `INTARSIA_DFLASH_WIN_SPLIT=1`); no other call site does, so every other call's `ShapeInfo` and kernel pick are unchanged.
* New HIP kernel `attn_decode_bf16_window_splitkv` (`kernels/hip/attention_win_splitkv.hip`, new file; no existing kernel source is touched):
  split-KV over the window `[max(0, p + 1 - window), p]` of a T = 1 decode call, stage 1 grid (split, kv head), stage 2 combine, scratch as the
  existing split-KV decode. Registered `.ext()`, `.scratch(..)`, accepted only when `win_split` is set, at priority 5: above the window kernel (-1) for
  windowed calls, below the plain split-KV decode (10) so a draft call with `window == 0` (context shorter than the window) still picks the same
  kernel as today (short-context draft bits unchanged by the knob).
* Not bitwise the old window kernel (other summation order, `exp2f`); it is within the bf16 tolerance of the reference (admission). Greedy ids are
  unaffected by draft numerics (verify decides), and are gated.

## Rules (pre-registered, written before any measurement)

Same binary, tatooine tp2, `generate --spec dflash --spec-k 7`, `INTARSIA_SPEC_TRACE=1`, 256 tokens, prompts short-0 and 4k-1
(`scripts/dfh/ids-prompts`), mean of the passes after the first 4. `INTARSIA_DFLASH_DEVICE_TOPK` is on (default since #323) in every arm.
Arms, runs in the order A B C D A B C D, two runs each:

* A = knobs unset (whole draft), B = `INTARSIA_DFLASH_WIN_SPLIT=1`, C = `INTARSIA_DFLASH_TP=1`, D = C + `INTARSIA_DFLASH_WIN_SPLIT=1`.

**R1, kernel.** The draft block's context-dependent part, block ms at 4k-1 minus block ms at short-0 (the trace does not isolate attention; the
windowed attention is the only context-dependent op in the block, and lane dh's 3.20 ms was this difference), must be <= 0.5x of today's,
i.e. B <= 0.5 x A and D <= 0.5 x C (today: 3.20 ms in both A and C, so <= 1.60 ms). Supporting data, not the rule: the kernel microbench
(`tests/win_splitkv.rs`) us a call, old vs new, at the draft's shapes (32/8 and 16/4 heads, D 128, 2054 keys), and the profiled op table.

**R2, pass.** At 4k-1, pass ms of D <= pass ms of C - 1.5 ms. At short-0, pass ms of D <= pass ms of C + 0.3 ms.

**R3, lane dh's item 1 rule re-evaluated with WIN_SPLIT on.** TP on (D) against TP off (B): draft total ms <= 0.70x and pass ms <= B - 3.0 ms, at
both short-0 and 4k-1. Verdict reported either way.

**Gates.**
* `kernels test --backend hip --quick` passes with the new kernel admitted (cases with `win_split` set, window 5 / 1024 / 2048-class, against the
  reference backend); the new kernel's own test against the old window kernel and a cpu f32 reference.
* `plan show` (json and text) for the 27B tp2 `--spec off`, 27B tp2 `--spec mtp --spec-k 2`, 35B tp2 off, 35B tp2 MTP k2 and 8B tp1 off identical
  to main's binary with the knob unset; DFlash 27B plan with the knob unset identical too (the new kernel is registered but never selected).
* Greedy ids of A, B, C and D equal `--spec off` at short-0 and 4k-1 (tp2).
* The target's kernels are untouched: `git diff 3f46da4f7 --stat` shows no edit to `attention.hip`, `attention_splitkv.hip`, and no change to the
  window kernel's registration.

**Categories (data only).** 512 forced tokens, the 7 category files, a fresh `serve --spec dflash --spec-k 7` per visit, tp2, visits C D C D if the
window has room (C = `INTARSIA_DFLASH_TP=1`, D = C + WIN_SPLIT; the category files are short-context, so no effect is expected); no pass/fail.

## Results

Binary: branch head at build `dw-intarsia` (code tree of `242a1bd44`, the test-only fix after it does not change the CLI), main `3f46da4f7` as
`dw-intarsia-base`. Tatooine tree `~/intarsiaml-dh`, volume `intarsia-cargo-dh`. Jobs: build #8838 (base + dev) and #8840 (test-compile fix), window t1
#8841 (13:55:45-14:01:50, radiance down 6 min, back after 50 s), window c1 #8843 (14:02:30-14:16:25, radiance down 14 min, back after 15 s;
`curl :8080/v1/models` answers). Raw: `raw/t1/` (`steps.log` has every trace line), `raw/c1/`. Radiance was confirmed answering after each window.

### Per-pass table (mean of 2 runs, passes after the first 4, 256 tokens, `generate --spec dflash --spec-k 7`, tp2)

| ms a pass | A short | B short | C short | D short | A 4k | B 4k | C 4k | D 4k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft flush | 1.38 | 1.39 | 1.35 | 1.39 | 1.42 | 1.43 | 1.42 | 1.38 |
| draft block | 6.54 | 6.50 | 4.13 | 4.10 | 9.70 | 6.98 | 7.31 | 4.49 |
| draft head | 3.21 | 3.21 | 1.71 | 1.71 | 3.23 | 3.33 | 1.71 | 1.76 |
| selector proj | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 |
| candidates download | 0.20 | 0.20 | 0.15 | 0.15 | 0.20 | 0.20 | 0.15 | 0.16 |
| host select | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| **draft total** | **11.47** | **11.43** | **7.47** | **7.48** | **14.66** | **12.07** | **10.71** | **7.90** |
| verify + rollback | 25.21 | 25.20 | 23.98 | 23.98 | 27.36 | 27.27 | 25.95 | 26.02 |
| **pass** | **36.69** | **36.64** | **31.47** | **31.48** | **42.03** | **39.34** | **36.67** | **33.94** |
| tokens a pass | 3.56 | 3.56 | 3.61 | 3.61 | 3.81 | 3.75 | 3.75 | 3.75 |
| decode tok/s (`generate`) | 96.4 | 96.5 | 113.7 | 113.7 | 90.4 | 95.0 | 102.1 | 110.3 |

A = knobs unset (whole draft), B = `INTARSIA_DFLASH_WIN_SPLIT=1`, C = `INTARSIA_DFLASH_TP=1`, D = both. `INTARSIA_DFLASH_DEVICE_TOPK` on in every arm (default).
Greedy ids equal `--spec off` in all 16 runs (A, B, C, D x 2 runs x short-0 and 4k-1). Tokens a pass at 4k: 3.81 (A) against 3.75 (B, C, D): the
draft's numerics change a few drafts (lane dh saw the same 0.06 from the tp split; the new kernel's other summation order gives the same shift
on the whole draft); never the output.

Kernel microbench (`WS_BENCH=1 tests/win_splitkv.rs`, 2000 back-to-back launches of one call, GPU 1, includes launch cost): `attn_decode_bf16_window` 93.8 us
(2054-key window at pos0 4102, both head layouts), 91 us (window 2048); `attn_decode_bf16_window_splitkv` **22.7 us** (32/8 heads), **20.7 us** (16/4),
23.2 / 16.4 us (window 2048): 0.22-0.25x of the old kernel, 40 calls a block pass = 2.8 ms saved, the 2.7 ms the trace shows.

### Verdicts

* **R1, kernel: met.** Context-dependent part of the draft block (block ms at 4k-1 minus short-0): A 9.70 - 6.54 = **3.16 ms**, B 6.98 - 6.50 =
  **0.48 ms (0.15x of A)**; C 7.31 - 4.13 = **3.18 ms**, D 4.49 - 4.10 = **0.39 ms (0.12x of C)**. Rule <= 0.5x, both with and without `INTARSIA_DFLASH_TP`.
* **R2, pass: met.** 4k: D 33.94 against C 36.67 = **-2.73 ms** (rule <= -1.5). Short-0: D 31.48 against C 31.47 = **+0.01 ms** (rule <= +0.3).
* **R3, lane dh's item 1 rule with WIN_SPLIT on (D against B): met at both prompts.** Short: draft 7.48 / 11.43 = **0.654x**, pass 31.48 - 36.64 = **-5.16 ms**.
  4k: draft 7.90 / 12.07 = **0.655x**, pass 33.94 - 39.34 = **-5.40 ms** (rules: <= 0.70x, <= -3.0 ms). Without WIN_SPLIT (C against A) the 4k draft is still
  10.71 / 14.66 = 0.73x: lane dh's miss stands for the old kernel; the new kernel is what takes it under the rule. The two knobs together
  (D against A): pass 42.03 -> 33.94 ms at 4k (-8.1 ms), 36.69 -> 31.48 ms at short (-5.2 ms); decode 90.4 -> 110.3 tok/s at 4k.
  Making either knob a default stays the user's call.

### Gates

| gate | result | job |
|---|---|---|
| `kernels test --backend hip --quick` | exit 0, 86 kernels, 506 cases; `attn_decode_bf16_window_splitkv` ADMITTED, 18 cases, 0 failed, max_abs 2.89e-3, rel_l2 1.35e-4 (the old window kernel runs the same 18 `win_split` cases too: 39 cases, ADMITTED) | #8841 |
| `tests/win_splitkv.rs` (GPU) | 3 passed: 430 cases against an f64 host reference and the old window kernel (draft heads 32/8, 16/4, groups of 8, 3 and 2, D 32-256, positions 0-6000, windows 0-2054), device position bitwise the host position, bench | #8841 |
| `plan show` json and text, knob unset, main vs branch: 27B tp2 off, MTP k2, DFlash k7; 35B tp2 off, MTP k2; 8B tp1 off | 12 SAME | #8841 |
| plan with `INTARSIA_DFLASH_WIN_SPLIT=1` at `--max-seq 20480` (27B DFlash k7 tp2, with and without `INTARSIA_DFLASH_TP=1`) | lists `attn_decode_bf16_window_splitkv` (1 each; 0 with the knob unset) | #8841 |
| greedy ids equal `--spec off`, A B C D x 2 x short-0, 4k-1 | 16 of 16 PASS | #8841 |
| target untouched | `git diff 3f46da4f7 --stat -- kernels crates/intarsia-backend-hip/src`: one new kernel file, additions only (5 files, +387 -0); `attention.hip`, `attention_splitkv.hip` and the old window kernel's registration are not touched; the new registration is `win_split`-gated and ranked below the plain split-KV decode | - |
| Mac: `cargo check --workspace --all-targets`, `intarsia-core --lib` (191 passed), `intarsia-backend-hip --lib` (29 passed, incl. the new pick test `window_splitkv_is_taken_by_win_split_calls_only`), `docs_env_vars`, `dflash2_cpu` (11 passed), `kernels test` cpu and metal `--quick --op attn_decode` | pass | - |

### Categories (data only; window c1, #8843)

7 files, 512 forced tokens, `serve --spec dflash --spec-k 7 --max-batch 1` tp2, a fresh server per visit, visits C D C D (two rounds), decode tok/s
pooled over both rounds (`tools/dfh/cat_table.py raw/c1`; its "rows differ from off" lines only mean there was no off arm). C = `INTARSIA_DFLASH_TP=1`
(the "before": lane dh's recommended configuration), D = C + WIN_SPLIT. The prompts are under 250 characters and the answers 512 tokens, so the context never
reaches the 2048-key window: the new kernel is never called in these files, and no change is expected.

| category | C: split draft | D: + WIN_SPLIT | D / C | tokens a pass C / D |
|---|---:|---:|---:|---|
| ho-chat | 98.1 | 98.4 | 1.003 | 3.12 / 3.12 |
| ho-code | 172.5 | 171.0 | 0.991 | 5.44 / 5.44 |
| ho-prose | 78.1 | 78.6 | 1.006 | 2.50 / 2.50 |
| ho-recall | 117.5 | 117.3 | 0.998 | 3.74 / 3.74 |
| rcp-code | 167.9 | 167.7 | 0.999 | 5.31 / 5.31 |
| rcp-prose | 82.2 | 82.3 | 1.001 | 2.62 / 2.62 |
| rcp-recall | 128.6 | 128.8 | 1.002 | 4.10 / 4.10 |

Run-to-run noise (lane dh's table reads 0.3-0.5 % between repeated visits); ho-code's -0.9 % rests on 2 rows.

### Notes

* Attribution: this lane's commits end with `Co-Authored-By: Claude Sonnet 5.5` (the model that wrote them), not the Opus 5.5 line the brief named.
* lane dm's `dflash_dry_run` PR: not on `origin/main` yet at the time of writing; not merged here. The branch does not touch that function.
* Tatooine disk: `/` 85 G used before (228 G, 132 G free), 80 G after (136 G free; other sessions' cleanups moved it); this lane added the tree sync
  (1.7 G for `~/intarsiaml-dh` with both CLIs) and growth inside the existing volume `intarsia-cargo-dh`; no new volumes.
* Not done: `INTARSIA_DFLASH_WIN_SPLIT` on a context in the 1k-2k range (window 0, plain split-KV already; unchanged by design) and a serve suite cell
  at 4k+ (the category files are short; the 4k effect is in the trace table). A row-batched variant (8 rows in one launch) was not needed.

## Default flip (user decision 2026-11-06, "flip both")

`INTARSIA_DFLASH_TP` and `INTARSIA_DFLASH_WIN_SPLIT` are default on in the same PR: unset = on, `=0` = off. TP acts only at tp>1 with in-process ranks and a
DFlash2 draft (left unset, a draft whose kv heads or ffn do not divide by the world stays whole on rank 0; `=1` forces the split and refuses it at load);
WIN_SPLIT only where the backend registers the kernel (HIP; `ShapeInfo::win_split` is read by no other backend's kernel selection). The measurements above
used explicit `=1`. After the flip the arms are: A = both `=0`, B = `INTARSIA_DFLASH_TP=0`, C = `INTARSIA_DFLASH_WIN_SPLIT=0`, D = knobs unset.
Merged with main e45bd5e3c (#324, `dflash_dry_run`) before the flip. Window t2 (below, filled when run) checks the plans and one ids run on the flipped head.

