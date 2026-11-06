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

## Progress log

* (nothing measured yet)
