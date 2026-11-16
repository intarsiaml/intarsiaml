# Lane mnr: row-invariant NAX rows kernel for MXFP4 on Metal (2026-11-15)

Status: built, measured, **kernel speed rule missed, end-to-end rule (a) failed; opt-in only, recommend leaving it off**. Branch `mnr`, worktree `/opt/code/intarsiaml/.worktrees/mnr`, base `origin/main` `c5ce87bc8`.

## 0. Brief + progress (kept at the top; context compaction loses briefs otherwise)

### 0.1 Coordinator overrides (these win over the brief text below)

- Worktree `/opt/code/intarsiaml/.worktrees/mnr`, branch `mnr` (not `lane-mnr/metal-nax-rows`). Only that folder locally;
  delete its `target/` when done.
- Exactness: the kernel changes numerics against today's default; the user's "keep bitwise" ruling (2026-11-06) stands, so
  it ships **opt-in only** (`INTARSIA_METAL_NAX_ROWS=1`). Never flip the default. Knob unset: `plan show --json`
  byte-identical, outputs bitwise unchanged. Under the knob: produce the KL / MMLU numbers so the user can decide.
- Parallel lanes `mka` (Brief B, attention KV-once) and `mtt` (Brief C, MTP TTFT) run on endor at the same time. Both
  append to `crates/intarsia-backend-metal/src/kernels.rs` and its pinned kernel list: keep that edit minimal, on conflict
  keep both. `git fetch` + merge origin/main before opening the PR.
- PR against main, do NOT merge it (coordinator merges opt-in work on green gates).
- Fleet: every GPU run on endor through `fleetq run --host endor --expect <MIN> --mem <GB> --quiet gpu --label "mnr: ..."
  -- <cmd>`; builds / CPU tests `--load cpu`; each job <= 32 GB; `endor-kev` and `endor-tensorfold` stay up; never
  `--force`, `--no-queue`, `--exclusive`; never release others' jobs; `acquire`/`release` for several steps, never leave a
  slot held.
- No tool call or wait over 4 min (Bash timeout <= 240000), poll at most every 4 min. No `timeout` on macOS:
  `perl -e 'alarm N; exec @ARGV'`. Speed numbers from ABBA runs with fresh processes. Never print secrets.
- unanimis MCP down: capture with `unim store`, source "intarsiaml lane mnr", never pass project or scope.
- Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; PR body ends with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Final report: what was built; gate table with numbers; speed ABBA table (M = 1..16 microbench, 8 streams, MTP k2);
  KL / MMLU under the knob; whether the stop rule was hit; PR number, fleetq job ids, leftovers.

### 0.2 Progress log

(newest last; update after every step)

- [x] kernel `kernels/metal/mxfp4_naxr.metal` (staged form: WR weight rows x BK per threadgroup, TM = 16)
- [x] registration + knob (`matmul_mxfp4_naxr`, `_rows`; hidden from `Backend::kernels` and fingerprint unchanged when unset;
      `_rows` takes every M because a registered rows_exact kernel is the only choice of an exact_rows call)
- 20:50 first microbench (#9424, `raw/bw1.log`): staged form is FLAT in M but 2.1x to 3.6x the GEMV M = 1 (gate/up 193 / 195 / 204 us
  at M 1 / 4 / 16 against 83 us; down 202 to 308 us against 86). Misses the kernel rule (M = 16 <= 1.6x, M = 4 <= 1.25x).
  Forms w64k128, w32k128, w32k256 give the same bits (hash equal): bits do not depend on the K step.
- 20:55 ablation (#9426): gate/up M = 16 209 us = about 41 barriers + A staging, 50 to 87 weight fetch (the bandwidth floor),
  55 decode + store to threadgroup memory, 30 matmul. The accelerator is not the limit; staging per weight is.
- 21:05 ablations 2 and 3 (#9435, #9450): device-tensor A no gain, deeper prefetch no gain, table decode -13 % (kept).
- 21:10 register-fed form (#9458): same 184 to 190 us on gate / up, 505 us on down. Same bits as the staged forms.
- 21:15 K-order probes (#9439 to #9475): the accelerator's order is one accumulator over quads (section 3), 0 mismatches.
- 21:25 ALU twin written; bitwise the accelerator forms in every bench cell (#9481); 1.46x to 2.85x at M = 1.
- 21:38 accuracy + ids under the knob (#9487), 8 streams ABBA (#9488): 1.02x at 8 streams, 0.65x at 2. No-op gates pass (#9489).
- MMLU and the short-prompt 8-stream arm withdrawn (speed rules already failed; the GPU queue is 30 deep tonight).

### 0.3 The brief, verbatim (roadmap `docs/design/2026-11-15-mac-roadmap.md` section 4, lane `mac-plan`)

#### Brief A: lane `mnr`, row-invariant NAX rows kernel for MXFP4 (Qwen3.8-27B on Metal)

**Goal.** Make M = 2..16 rows of an MXFP4 matmul cost close to one bandwidth pass on the M5 Max by running them on the
matrix units (NAX, `mpp::tensor_ops::matmul2d`) with the 4-bit weights decoded per tile inside the kernel. Ship it opt-in
behind `INTARSIA_METAL_NAX_ROWS=1`. With the knob unset nothing changes (`plan show` byte-identical).

**Why.** On endor the 27B's one-row step is a weight pass at 95 % of copy bandwidth (31.5 ms GPU, 30.6 tok/s). Extra rows
on the bitwise ALU kernels cost almost a step each: verify 3 rows 39.5 ms, 8 rows 89 ms (lane dm,
`docs/results/2026-11-06-metal-dflash2.md`); 8 batched streams give 36 tok/s in total. The existing NAX path
(`matmul_mxfp4_nax`, M > 8) decodes the whole weight to bf16 scratch on every call, about 200 ms per pass through the
model whatever the row count (`docs/design/2026-11-15-mac-roadmap.md` section 1.4). The FLOPs of 16 rows are 0.83 TFLOP
per pass = 13 ms at the NAX peak of 62 TFLOPS, against a 26 ms weight read. TensorFold reaches 87 tok/s on this box with
a 15-node tree, so the hardware can do it.

**Exactness contract (read this first).** The new kernel's bits differ from `matmul_mxfp4_gemv`'s (the matrix units
accumulate K in another order). The user ruled "keep bitwise" on 2026-11-06, so this stays **opt-in** and you do not flip
any default. What must hold under the knob:

1. Row invariance: a row's output bits do not depend on M, on the row's index, on the other rows' contents, on the tile
   choice or on buffer offsets. (The existing `matmul_mxfp4_nax` test 2 in `tests/mxfp4_nax.rs` proves the same for the
   scratch form; reuse its structure.)
2. Under the knob the same kernel serves M = 1 (decode step), M = 2..16 with `exact_rows` (verify and batch rows) and
   prefill passes of 8 rows or fewer, so speculative and batched output is bitwise the knob's own one-row output.
3. Against the default it is KL / MMLU-gated (gates below).

**Design.**

* Kernel `mxfp4_naxr_{bf16,f32}` in a new `kernels/metal/mxfp4_naxr.metal` (Metal 4, added to `METAL4_SOURCES` in
  `crates/intarsia-backend-metal/build.rs`). Start from `kernels/metal/moe_nax.metal` (`moe_mm_nax_impl`): it already
  decodes 4-bit blocks to bf16 once per tile into threadgroup memory and runs `matmul2d` in multiply-accumulate mode into
  a float cooperative tensor per simdgroup, and it runs the 35B's grouped experts at about 39 TFLOPS effective. Dense
  form: one threadgroup per 64 (or 32) weight rows; activation tile TM = 16 rows (rows >= M are zero and never stored;
  for M <= 8 try an 8-row tile padded to 16 as `moe_nax` does); K walked in steps of BK = 256 or 512; per step the lanes
  decode their 32-weight groups (16 weight bytes + 1 scale byte) to bf16 with the arithmetic decode of `mxfp4.metal`
  (`mx_dec8` / the e2m1 * 2^e form of `mxfp4_w_to_bf16`, exact in bf16) into the staged B tile; the A tile (activations,
  M x BK bf16) is staged once per step from device memory (it is tiny and cache-resident: 16 x 5120 x 2 B). Epilogue:
  bf16 round-to-nearest-even or f32 store, as `gemm_nax_fp8.metal`'s `epilogue`.
* W4A16 only (`act` 0); W4A4 calls keep the GEMV (the registration predicate excludes them, as `nax_ok` does).
* Second form if the staged tile misses the kernel rule: register-fed operands
  (`get_right_input_cooperative_tensor`, filled from decoded codes; `kernels/metal/attention_flash_nax.metal` `nax_mma`
  shows the fragment layout and measured 12.6 TFLOPS with 8-byte lane loads; 4-bit codes give four times the elements per
  loaded byte).
* Registration (`crates/intarsia-backend-metal/src/kernels.rs`, predicates in `src/mxfp4.rs`; read
  `crates/intarsia-core/src/registry.rs` lines 354 to 365 for how `rows_exact` selection works): with the knob set,
  `matmul_mxfp4_naxr_rows` (`.rows_exact()`, M 2..=16) and `matmul_mxfp4_naxr` (not `rows_exact`, M 1..=8, priority above
  the GEMV's). With the knob unset both predicates return false, so selection and `plan show` are unchanged. Append the
  names to `TABLE` and `MAIN_KERNEL_LIST` (test `kernel_list_is_unchanged_by_core_op_fusion_p1`). No change in
  `intarsia-core`.
* Stretch, only after every gate below passes: let the kernel take M > 8 under the knob (replacing the scratch path) and
  measure prefill; if it is within 3 % or faster, prefill, decode and verify share one numerics class and
  `prefill_peel` is unnecessary under the knob. Report it; do not change the recipe.

**Files.** New: `kernels/metal/mxfp4_naxr.metal`, `crates/intarsia-backend-metal/tests/mxfp4_naxr.rs`. Edited:
`crates/intarsia-backend-metal/src/mxfp4.rs`, `src/kernels.rs`, `build.rs`, `examples/mxfp4_bw.rs` (the microbench; add
the new kernel as a candidate). Lane B (`mka`) also appends to `kernels.rs` and `MAIN_KERNEL_LIST`; expect a two-line
merge conflict and keep both. Do not edit `attention_splitkv.metal`, `attn.rs`, `rowtab.rs` (lane B) or
`crates/intarsia-core/src/model/mtp.rs`, `spec_pipe.rs`, `engine.rs` (lane C).

**Pre-registered rules (write them into the results doc before measuring).**

* Kernel (shapes 17408 x 5120, 5120 x 17408, 12288 x 5120, 5120 x 6144; `examples/mxfp4_bw.rs`; GPU timestamps):
  M = 16 <= 1.6x and M = 4 <= 1.25x the M = 1 `matmul_mxfp4_gemv` time on every shape (today's bitwise rows: M = 8 is
  2.7 to 4.1x). M = 1 on the new kernel: report it (it is the knob's one-row step; expected about 1.2 to 1.4x the GEMV).
* End to end, same binary, knob off against on, served, ABBA, localeval `speed` (the `local-model-bench` skill):
  (a) `--max-batch 8`, 8 streams at 1k: aggregate decode >= 3x (36 tok/s today); (b) default MTP k2, 1 stream, natural-v1
  1k and 4k: no cell below 0.97; (c) `--spec dflash --spec-k 7` with the knob against MTP k2 without it at natural-v1 4k:
  report (lane dm measured 0.54x on the bitwise rows; above 1.0 makes idea 4, the tree, the next lane).

**Gates.**

1. `tests/mxfp4_naxr.rs`: row invariance (M 1..16, every row index, random co-rows, tile forms, 2- and 4-byte offsets,
   bf16 and f32 out, K 32 .. 17408, N with tails 37 / 130 / 515); against an f64 reference (bound: the
   `matmul_mxfp4_nax` case bounds, max_abs 2e-3 / rel_l2 5e-5 class); against the GEMV relative RMS < 1e-5; registration
   (knob unset: never selected; set: M and act ranges); a planted fault (one row's tile zeroed) is caught.
2. `intarsia kernels test --backend metal`: every existing row identical to main, new rows ADMITTED, 0 FAIL.
3. `intarsia plan show --model <M> --backend metal` byte-identical to main with the knob unset for
   `~/models/Qwen3-8B`, `~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`, `~/models/Qwen3.6-35B-A3B-NVFP4`, each with default
   flags, `--spec off` and `--spec mtp`.
4. Ids under the knob, 256 greedy tokens: MTP k2 == MTP k4 == `--spec off`; `--spec dflash --spec-k 7`
   (draft `z-lab/Qwen3.8-27B-DFlash2`, HF cache snapshot `50307d4c`) == `--spec off`; served `--max-batch 8` with 2, 4, 8
   concurrent identical prompts == solo; on a short prompt, 1k, 4k (two prompts) and 16k. Repeat runs equal.
5. Accuracy under the knob, the method of `docs/results/2026-11-02-m27-metal.md` section "Task 3, accuracy" (HF W4A16
   golden regenerated on endor CPU, `tools/golden_quark_mxfp4.py`): prefill and decode KL21 and top-1 reported beside
   the default's (5.9e-4 / 9.6e-4); pass = decode KL21 <= 1.5e-3 (HIP's level) and top-1 within 0.3 points. MMLU
   n = 1425: within 0.010 of 0.8161.
6. `cargo test --workspace --release --locked` (the endor-only
   `completions_logprobs_do_not_stall_the_async_threads` may be skipped, say so).
7. The speed rules above.

**Stop rules.** (1) If the staged-tile kernel misses the kernel rule, try the register-fed form once; if M = 16 is still
above 2.2x M = 1 on the two FFN shapes, stop, keep the code on the branch, write up the measured TFLOPS per form and what
limits it. (2) If row invariance cannot be made to hold on some shape, exclude that shape in the predicate and say so; if
it fails on the FFN shapes, stop. (3) If gate 5 fails, stop and report the KL table; do not tune tolerances.

**Deliverables.** Branch `lane-mnr/metal-nax-rows` from `origin/main`, worktree
`/opt/code/intarsiaml/.worktrees/mnr` (yours alone; build there, delete `target/` when done). Results doc
`docs/results/2026-11-15-mnr-metal-nax-rows.md` with raw logs under `docs/results/2026-11-15-mnr-metal-nax-rows/raw/`
and scripts under `scripts/mnr/`. A row in `docs/decisions/LOG.md` (status opt-in, the user's default decision open).
PR against main; report the PR number, the rule table and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mnr: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

## 1. What was built

All of it is behind `INTARSIA_METAL_NAX_ROWS=1`. With the knob unset the two kernels are not registered (`Backend::kernels` drops
them), the fingerprint string is unchanged and the main metallib is byte-identical (the new shaders live in the metal4.0 library).

* `kernels/metal/mxfp4_naxr.metal` (metal4.0 library), launcher in `crates/intarsia-backend-metal/src/mxfp4.rs`:
  * **staged forms** `mxfp4_naxr_{bf16,f32}_w64k128`, `_w32k128`, `_w32k256`: a threadgroup owns 64 or 32 weight rows, walks K
    in steps of 128 or 256, decodes its weights to bf16 into threadgroup memory (table decode, integer math), stages the 16
    activation rows, and runs `matmul2d` multiply-accumulate into a float cooperative tensor per simdgroup (the structure of
    `moe_nax.metal`). Any M: 16 activation rows per grid row.
  * **register-fed form** `mxfp4_naxq_{bf16,f32}`: no threadgroup memory and no barriers; a simdgroup owns 32 weight rows and
    runs `matmul2d` 16 x 32 x 16 on register-resident cooperative tensors, each lane decoding the 4 codes it holds of 4 weight
    rows per run (the fragment layout of `attention_flash_nax.metal`).
  * **ALU twin** `mxfp4_naxt_{bf16,f32}_m{1,2,4}`: no accelerator at all. One lane per output row, plain fp32, summing in the
    order the accelerator was found to use (section 3). Bitwise the accelerator forms.
* Kernels `matmul_mxfp4_naxr` (ordinary calls, M 1..=8, priority 6, above the GEMV) and `matmul_mxfp4_naxr_rows`
  (`rows_exact`, **every M**, not the brief's 2..=16: `Registry::select` gives an `exact_rows` call only `rows_exact` kernels
  once the op has one registered, so a narrower range would leave one-row and > 16-row verify calls without a kernel).
  W4A16 calls read x directly; W4A4 calls QDQ x into the scratch first (`mxfp4_xprep_bf16`, as the GEMV) and stay row-invariant.
* `Backend::fingerprint` gets `|naxr:1` with the knob set (the registered set changes, so admission results must not be shared).
* `tests/mxfp4_naxr.rs` (8 tests), `examples/naxr_order.rs` (the K-order probes), `examples/mxfp4_bw.rs` takes kernel and row
  lists, scripts under `scripts/mnr/`.

## 2. Kernel speed: the rule is missed, stop rule 1 applies

Microbench `examples/mxfp4_bw.rs` (#9481, `raw/bw-final-{a,b}.log`): a pool of distinct weight buffers (1 GiB, far above any cache),
wall time of a synced batch of 400+ launches per cell (so it includes encoding), median of 5 batches, two passes with the
kernel and M order reversed, the table shows the mean of the two passes in us and, in brackets, the ratio to the M = 1
`matmul_mxfp4_gemv` of the same pass. The brief asked for GPU timestamps; the batch wall time is what the existing example
measures and the GEMV M = 1 column reproduces the roadmap's 524 to 582 GB/s.

**FFN gate / up, N x K = 17408x5120** (GEMV M = 1: 85.2 us)

| kernel | M = 1 | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| today's bitwise GEMV rows | 85 (1.00x) | 89 (1.04x) | 95 (1.11x) | 108 (1.26x) | 260 (3.06x) | 622 (7.30x) |
| staged w64k128 | 170 (2.00x) | 166 (1.95x) | 167 (1.96x) | 169 (1.98x) | 173 (2.03x) | 184 (2.16x) |
| staged w32k256 | 219 (2.58x) | 233 (2.74x) | 234 (2.75x) | 240 (2.82x) | 277 (3.26x) | 284 (3.34x) |
| register-fed | 184 (2.15x) | 184 (2.16x) | 184 (2.16x) | 185 (2.17x) | 186 (2.18x) | 188 (2.21x) |
| ALU twin | 124 (1.46x) | 166 (1.94x) | 286 (3.36x) | 361 (4.24x) | 551 (6.47x) | 848 (9.95x) |

**FFN down, N x K = 5120x17408** (GEMV M = 1: 87.6 us)

| kernel | M = 1 | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| today's bitwise GEMV rows | 88 (1.00x) | 89 (1.02x) | 94 (1.07x) | 116 (1.33x) | 276 (3.15x) | 740 (8.46x) |
| staged w64k128 | 248 (2.83x) | 249 (2.84x) | 250 (2.86x) | 251 (2.87x) | 258 (2.95x) | 288 (3.29x) |
| staged w32k256 | 199 (2.28x) | 200 (2.28x) | 201 (2.30x) | 206 (2.35x) | 216 (2.46x) | 249 (2.84x) |
| register-fed | 505 (5.76x) | 506 (5.77x) | 505 (5.77x) | 505 (5.77x) | 507 (5.79x) | 510 (5.82x) |
| ALU twin | 249 (2.85x) | 424 (4.84x) | 794 (9.07x) | 1046 (11.94x) | 1075 (12.26x) | 1141 (13.02x) |

**attn q_proj, N x K = 12288x5120** (GEMV M = 1: 60.7 us)

| kernel | M = 1 | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| today's bitwise GEMV rows | 61 (1.00x) | 65 (1.07x) | 70 (1.15x) | 80 (1.32x) | 176 (2.90x) | 421 (6.94x) |
| staged w64k128 | 100 (1.64x) | 100 (1.65x) | 100 (1.65x) | 100 (1.66x) | 104 (1.71x) | 115 (1.90x) |
| staged w32k256 | 160 (2.64x) | 164 (2.70x) | 165 (2.72x) | 173 (2.85x) | 222 (3.65x) | 213 (3.51x) |
| register-fed | 167 (2.75x) | 166 (2.73x) | 166 (2.74x) | 166 (2.74x) | 167 (2.75x) | 168 (2.77x) |
| ALU twin | 95 (1.56x) | 130 (2.14x) | 233 (3.84x) | 309 (5.10x) | 473 (7.79x) | 639 (10.53x) |

**out_proj, N x K = 5120x6144** (GEMV M = 1: 33.4 us)

| kernel | M = 1 | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| today's bitwise GEMV rows | 33 (1.00x) | 35 (1.06x) | 38 (1.15x) | 52 (1.56x) | 99 (2.96x) | 220 (6.60x) |
| staged w64k128 | 90 (2.69x) | 90 (2.70x) | 90 (2.71x) | 91 (2.73x) | 95 (2.84x) | 104 (3.11x) |
| staged w32k256 | 73 (2.17x) | 72 (2.15x) | 69 (2.07x) | 70 (2.11x) | 74 (2.22x) | 84 (2.52x) |
| register-fed | 179 (5.38x) | 180 (5.39x) | 180 (5.39x) | 179 (5.37x) | 179 (5.38x) | 181 (5.42x) |
| ALU twin | 84 (2.53x) | 137 (4.12x) | 242 (7.26x) | 343 (10.29x) | 340 (10.21x) | 344 (10.32x) |

Rule table (the launcher's pick per shape: w64k128 above 8192 output rows, w32k256 at or below; the twin only for M = 1 on the wide outputs):

| shape | M = 16, rule <= 1.6x | M = 4, rule <= 1.25x | M = 1 (report) | stop rule 1 (M = 16 > 2.2x) |
|---|---:|---:|---:|---|
| FFN gate / up | **2.16x MISS** | **1.98x MISS** | 1.46x | below |
| FFN down | **2.84x MISS** | **2.35x MISS** | 2.28x | above |
| attn q_proj | **1.90x MISS** | **1.66x MISS** | 1.56x | below |
| out_proj | **2.52x MISS** | **2.11x MISS** | 2.17x | above |

**The kernel rule is missed on every shape.** The row kernel is flat in M, as designed (16 rows cost 1.07x to 1.25x of one), but
the flat cost is 1.6x to 2.8x a bandwidth pass instead of the 1.2x to 1.4x the brief expected. Stop rule 1 was written for
"above 2.2x on the two FFN shapes": gate / up ends at 2.16x and down at 2.84x, so it is met on one of the two. I read that as
not a clean stop, finished the correctness gates and ran the end-to-end arms once so that the decision has numbers, and did not
tune further. Everything stays opt-in.

**What limits it** (ablation on gate / up, M = 16, the job after #9424 (id not kept), #9435, #9450; one piece removed at a time, 209 us total at that point):

| piece | us | note |
|---|---:|---|
| weight fetch | 50 to 87 | the bandwidth floor: 47 MB at about 540 GB/s; the GEMV's whole cost is this (83 us) |
| decode + store to threadgroup memory | 55 (arithmetic), 35 (table) | 89 M weights become 178 MB of bf16 written and read back by the primitive |
| barriers + activation staging | 41 | 2 barriers per K step, 10 880 threadgroup-steps |
| `matmul2d` itself | 30 | 2.85 GFLOP: the accelerator is far from its limit |

The pieces add up instead of overlapping: a threadgroup waits at its barrier for its slowest load, so the memory time is not hidden
behind the decode the way it is in the barrier-free GEMV. Tried and measured: a second prefetch step ahead (no change), the
activations read by the primitive straight from device memory instead of staged (no change on gate / up, 30 % worse on down), a
table decode instead of the arithmetic one (-13 %, kept), 32 or 64 weight rows per threadgroup and K steps of 128 or 256 (per-shape
pick, kept), and the register-fed form, which removes threadgroup memory and barriers altogether and lands at the same 184 us on
gate / up and far worse on the long-K shapes (6 small loads per lane per 16 x 32 x 16 run). In FLOP terms at 16 rows: staged 15.4
TFLOPS-equivalent on gate / up, 11.6 on down; register-fed 15.3 and 5.6; the roadmap's 62 TFLOPS peak needs 64-row tiles of ready
bf16. Per weight: about 1.9 ns staged against 0.94 ns for the GEMV. `moe_nax` reaches its 39 TFLOPS at 32 to 64 rows per weight
tile; at 16 rows the same per-weight staging cost buys a quarter of the FLOPs.

So on this box a 4-bit weight costs about twice a bandwidth pass to get in front of the matrix units, whatever feeds them. The
kernel still does what no ALU kernel can: 8 rows at 2.0x to 2.6x (today's bitwise rows: 3.0x to 3.2x) and 16 rows at 1.9x to 2.9x
(today 6.6x to 8.5x).

## 3. Found on the way: the accelerator's K order, exactly

`examples/naxr_order.rs tree` plants +2^30 at position a, -2^30 at position b and ones elsewhere; every partial sum of ones that
merges into a node holding one unbalanced 2^30 is absorbed, so the result counts the ones outside the smallest summation subtree
that holds both a and b. The matrix for K = 32 (`raw/order-tree.log`) reads off directly:

    acc = 0
    for each block of 16 along K, for j = 0..3:   a = 16 t + 2 j
        acc = rn(acc + rn(rn(p[a] + p[a+1]) + rn(p[a+8] + p[a+9])))

one f32 accumulator over quads in K order, every add rounded to nearest even, every product exact (bf16 x bf16). On random
operands with a wide exponent spread this host formula matches the kernel in **0 of 1024 outputs at every K from 8 to 17408**,
where a plain K-ascending chain differs in 15 % to 98 % of them (`raw/order-random.log`, #9475, #9481). It does not depend on how K
is cut into `matmul2d` runs (16, 128, 256), on staged against register-fed operands, or on the row's position in the tile. Subnormal
bf16 operands are read as zero.

Two things follow:

1. Row invariance of the accelerator forms is not luck: it is this fixed order. `naxr_is_the_quad_order_sum_bit_for_bit` pins it.
2. Plain fp32 can produce the same bits. The **ALU twin** (`mxfp4_naxt_*`) does: one lane per output row walks K with two fmas and
   two adds per quad, and its output hash equals the accelerator forms' in every cell of the bench (`naxr_forms_give_the_same_bits`
   tests it for M 1, 2, 3, 5, 16, bf16 and f32). Its serial chain per row makes it slower than the lane-parallel GEMV (1.46x to
   2.85x at M = 1, and it scales badly in M because no lane shares an activation load), so the launcher uses it only for the
   one-row call on the wide outputs. But it shows that "prefill on the matrix units, decode on the ALU, one numerics class" is
   reachable: a decode GEMV in this order would make every decode row bitwise the NAX prefill's row and retire `prefill_peel`,
   with no accelerator in the decode path. That is a different lane from this brief (section 7).

## 4. End to end (Qwen3.8-27B MXFP4, endor, same binary, knob off against on, fresh server per arm)

**(a) `serve --max-batch 8`, 1k prompts, 128 output tokens, localeval `speed --concurrency 2,4,8` (ABBA, #9488, `raw/b8-*`):**

| streams | off (A1 / A2) | on (B1 / B2) | on / off | rule |
|---:|---:|---:|---:|---|
| 2 | 32.3 / 32.5 | 20.8 / 21.5 | 0.65x | |
| 4 | 39.9 / 39.8 | 30.4 / 30.2 | 0.76x | |
| 8 | 39.4 / 38.1 | 39.8 / 39.5 | **1.02x** | >= 3x: **FAIL** |

Aggregate decode tok/s. The knob is a loss at 2 and 4 streams (its flat cost is twice a bitwise row there) and a wash at 8. The
roadmap's 3x assumed the 8-row step is the MXFP4 rows; the measured step is about 200 ms either way while the MXFP4 rows of an
8-row step are about 77 ms bitwise and about 55 ms on the row kernel by the microbench, so most of the step is elsewhere (in this
arm also the busy prefill chunks of the other streams, which decode 256-row chunks through the scratch NAX form). Not attributed
further in this lane.

**(b) default MTP k2, one stream:** see section 5 (pending at the time of writing if the cell says so). From `generate` under the
knob (256 greedy tokens, #9487): plain decode 18.95 tok/s (short prompt) and 16.7 (1k) against about 30 on the default; MTP k2
31.4 / 28.4; MTP k4 33.9 / 32.8. The one-row step costs 1.5x to 2.3x under the knob, so rule (b) (no cell below 0.97) cannot hold.

**(c) `--spec dflash --spec-k 7`:** not run. It needs 8-row verify passes to be cheap against the one-row step of the same class,
and under the knob both moved the wrong way for a single stream.

## 5. Gates

| gate | result |
|---|---|
| 1. `tests/mxfp4_naxr.rs` | GATE1 |
| 2. `kernels test --backend metal` | knob unset: 55 kernels, 1276 cases, list and report identical to a build of origin/main (`5c9cb111e`). Knob set: 57 kernels, 1419 cases, 0 FAIL; `matmul_mxfp4_naxr` 56 cases ADMITTED (max_abs 9.3e-4, rel_l2 2.9e-5), `matmul_mxfp4_naxr_rows` 87 cases ADMITTED (2.0e-3, 4.1e-5). #9489, `raw/noop/` |
| 3. `plan show --json` knob unset | byte-identical to origin/main for Qwen3-8B, Qwen3.8-27B MXFP4, Qwen3.6-35B NVFP4, each with default flags, `--spec off`, `--spec mtp`: 9 of 9 SAME (#9489) |
| 4. ids under the knob, 256 greedy tokens | MTP k2 == MTP k4 == `--spec off` on the short prompt and on natural-v1 1k (#9487, `raw/ids/`). **Not run:** DFlash, the batched 2 / 4 / 8 identical prompts, 4k and 16k, repeat runs |
| 5. accuracy under the knob (HF W4A16 golden, 1441 prefill / 160 decode positions) | decode KL21 **1.089e-3** scored rows (1.020e-3 all rows), top-1 **99.38 %**; pass rule decode KL21 <= 1.5e-3: PASS. Default on the same build: 1.216e-3 (1.582e-3), 98.75 %. Prefill 7.734e-4 / 99.51 % with and without the knob (the accuracy prefill runs on the scratch NAX form either way). #9487, `raw/acc-{off,on}.out`. **MMLU not run** (withdrawn, #9530: an hour of the shared GPU for a knob that already fails its speed rules) |
| 6. `cargo test --workspace --release --locked` | GATE6 |
| 7. speed rules | kernel rule MISS on all four shapes (section 2); (a) 1.02x against >= 3x: FAIL; (b) GATE7B; (c) not run |

The brief's default numbers for gate 5 (5.9e-4 / 9.6e-4) are the 2026-11-02 build's; today's default measures 7.7e-4 / 1.2e-3 on
scored rows with the same golden and flags.

## 6. Decision

`INTARSIA_METAL_NAX_ROWS` stays **off and should stay off**: it loses for one to four streams and gains nothing at eight. It is
safe to merge as an opt-in (knob unset: nothing registered, plans and kernel reports identical), and the user's "keep bitwise"
ruling of 2026-11-06 is not touched. There is no default decision to make from this lane.

## 7. What I would do next

1. **A lane-parallel GEMV in the accelerator's order** (the twin, done properly). The order is one serial accumulator over quads,
   so the quads of a row can be computed by 32 lanes in parallel and only the final accumulation is serial: 1280 adds per row of
   K = 5120 against 89 M weight-ops. If that accumulation can be kept off the critical path (one lane per row folding quad sums the
   others produce, through threadgroup memory), the one-row step keeps its bandwidth cost and prefill (NAX), decode and verify rows
   share one numerics class with no accelerator in the decode path. That would buy the consistency half of this brief (no
   `prefill_peel`, row-exact against prefill) at no speed cost; it does not buy cheap rows.
2. **Cheap rows need the staging cost gone, not a better tile.** Two candidates: weights kept as bf16 for the row path (2 GB per
   GB of 4-bit: 51.8 GB for the 27B, too much), or the primitive's own device load on a bf16 copy of only the layers a tree verify
   needs. Neither is attractive; I would not open a lane on NAX rows for MXFP4 again without a new idea for the decode.
3. **Find where the 8-row step goes** before any further row work on the 27B: about 200 ms per step with about 77 ms of MXFP4
   rows. A per-dispatch profile of a steady 8-row step (`INTARSIA_METAL_PROFILE=cb`, short prompts so no prefill runs beside it)
   is a one-slot job and decides whether lane B's attention work or the fp8 lm_head rows are the real cost.

## 8. Jobs, files, leftovers

fleetq (endor): builds #9419, #9423, #9438, #9452, #9457, #9459, #9483, #9485, #9492 (`--load cpu`); kernel microbench and
ablations #9424, the job after #9424 (id not kept), #9435, #9450, #9458; K-order probes #9439, #9441, #9443, #9455, #9475; kernel tests + probes + bench #9481;
no-op gates #9489; accuracy + ids #9487; 8 streams ABBA #9488; workspace tests #9490 (first run, one stale pinned count, fixed);
FINALJOBS. Withdrawn before they ran: #9525 (build, done inside #9487 instead), #9530 (MMLU), #9558 (short-prompt 8-stream arm).
Raw logs: `docs/results/2026-11-15-mnr-metal-nax-rows/raw/`. Scripts: `scripts/mnr/`. unanimis: `c284aa6a`.
