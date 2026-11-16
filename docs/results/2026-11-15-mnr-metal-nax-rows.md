# Lane mnr: row-invariant NAX rows kernel for MXFP4 on Metal (2026-11-15)

Status: IN PROGRESS. Branch `mnr`, worktree `/opt/code/intarsiaml/.worktrees/mnr`, base `origin/main` `c5ce87bc8`.

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
- next: table decode / device-tensor A (ablation 2, #9435), NAX K-order probe (can an ALU kernel give the same bits for small M?)
- [ ] microbench (kernel rule)
- [ ] gate 1 tests
- [ ] gates 2, 3
- [ ] gate 4 ids
- [ ] gate 5 KL / MMLU
- [ ] gate 6 workspace tests
- [ ] gate 7 speed ABBA
- [ ] LOG row, PR

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

