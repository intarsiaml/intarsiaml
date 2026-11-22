# Inkling-2-Mini Metal decode: host gap (lane ink-hostgap, 2026-11-21)

Status: waiting on endor #10943 (items 2 and 3 timing); code done and CPU-checked. Branch `ink/hostgap` (from ink/metal a88db4620, fast-forwarded to e61f392fd), not pushed.

## Briefs received

- 2026-11-21 (launch): during Metal decode the GPU should never sit idle waiting for the host. Starting point (integration
  lane, a88db4620): 50-52 tok/s on endor, 19.8 ms wall against 15.8 ms GPU busy per token, about 4 ms (20 %) host gap; 417 op
  calls and 613 dispatches per token. Measure first (host encode, commit / wait points, CPU-GPU syncs, per-dispatch host
  cost), then remove the gap (chunked commits, GPU-side sampling with encode-ahead, cheaper encoding, ICB replay). Target
  wall <= GPU busy + 0.5 ms. Other models must not regress; new modes default-on for Inkling only after its gates; logits
  bitwise; stay out of kernels and op contracts (dfuse); single-stream first; rerun G1-G3 before reporting.
- 2026-11-21 (coordinator, rescope): Fable's review (`.worktrees/ink-fable/decode-profile-review.md` section 0.1) finds the
  "4 ms host gap" is the `generate` step profiler: `--profile-every 16` is the Metal default for `generate`
  (`crates/intarsia-cli/src/common.rs:357`; only `for_serving` sets 0), so every 16th step runs as 419 waited command buffers
  (77-82 ms); normal steps run at 15.76 ms with 1 us median GPU idle (about 63.4 tok/s), so the target is already met for
  greedy. New scope, in order:
  1. Measurement A now, one fleetq job (`--quiet gpu,cpu`, `--expect 15`): the prof.sh generate command with
     `--profile-every 0 --max-tokens 256` twice, then once at `--temperature 0.7`; message the three tok/s numbers.
  2. Propose the profiler default (generate defaults to 0 on Metal, or it keeps profiling but reports tok/s from unprofiled
     steps only); message main before changing a CLI default other models share.
  3. argmax: 172 us for a 0.8 MB scan (review item 8); target <= 25 us.
  4. The sampled path (temperature / top-k / top-p): make sure it has no per-step host sync, and size it.
  5. The draft-cycle readback the MTP lane will need (review section 4): device-side accept / verify so the host is not in
     the loop per cycle; interface with ink-mtp through main.
  No encode-ahead or ICBs unless measurement A shows a real gap.
- 2026-11-21 (coordinator): move measurement A up with a reserved start (`--at 22:27`); go ahead with argmax and the
  sampled-path host-timer breakdown meanwhile. (Both reservations, 22:27 and 23:26, were refused by fleetq with 409:
  collisions with our own waiting jobs; reported back.)

- 2026-11-21 (coordinator): go-ahead for option (a) of item 2: Metal `generate` defaults to `--profile-every 0`. Conditions: its
  own commit (revertable alone); `generate` says in its `top ops` line that profiling is off and how to turn it on; the
  `--profile-every` help text and the CLI comment updated; a docs/benchmarking.md note that Metal `generate` tok/s from before
  the change is 12-25 % low, plus a LOG.md row.
- 2026-11-21 (coordinator): build #10829 let ahead of the line (done 22:02, exit 0).

- 2026-11-21 (coordinator, relaying ink-mtp): (1) approved: `INTARSIA_HOST_PROFILE` hooks in the spec loop in `engine.rs`,
  profiling lines only; stay out of `crates/intarsia-core/src/model/dspark.rs` (ink-mtp's, changing fast). (2) Correction to
  the item-5 note: Metal now has `Backend::topk_rows` (ink/mtp c770aaef3: two dispatches, f32 / bf16, k <= 64, sampler-order
  ties; `supports_topk` stays false), so S2 downloads 7 x 64 ids + values (about 3.6 KB), not 2.8 MB, and the host Markov head
  over 64 candidates is pure arithmetic; fix the M2 line with a note. (3) ink-mtp queued the per-cycle split (flush / block /
  head / markov / wall / ingest) as a default-load backfill job; H1 / H2 still wait on those numbers. Go ahead with the
  engine.rs hooks if small: a real check (build + the tiny spec test), its own commit, no new GPU job for them.
- 2026-11-21 (coordinator): H1 / H2 not to be built until S1 + S2 + S3 is shown to be well over 1 ms per cycle; #10839 stays
  queued.
- 2026-11-21 (coordinator): item 5 dropped on ink-mtp's per-cycle split (host 1.2-1.6 ms of a 35 ms cycle, under 5 %; numbers in
  item 5). The lane finishes with #10839 / #10840 / #10888: report those, then stop.

- 2026-11-21 (coordinator, after the final report): three more items.
  1. `argmax_f32_v4t` stays default-on. It is exact (max_abs 0, admitted, greedy md5 identical), only V % 4 != 0 rows take it,
     and `plan show` is identical for 3 models. Note the revert switch `INTARSIA_METAL_ARGMAX_V4T=0` in the doc.
  2. Settle the base-vs-branch confound first, in one quiet ABAB job (`--quiet gpu,cpu`, an honest `--expect`):
     - base vs b1 interleaved, at least 3 reps per arm, `--profile-every 0`, medians reported
     - workloads: the short prompt, 5k, and t0.7 seed 7
     - ink/hostgap merges into ink/metal only if 5k and t0.7 are within 1 % (or faster). If a gap is real, find it before item 3.
  3. Then build option (a), the bitwise host sampler: soft-cap / keys / exp split across threads, a stable chunked radix sort,
     the f64 sum kept sequential.
     - Target: sampler + soft-cap at t1.0 <= 1 ms per token, from 3.47.
     - Gate: sampled md5 identical to today at t0.7 and t1.0 for 2 seeds each, plus the microbench before and after.
     - CPU tests and the microbench correctness run on spark2. Timing stays on endor, folded into the item-2 job if (a) is
       built first, else a second short quiet job.
     - (b), the device sampler, is out: it is not bitwise.
     Also: say which sampling path real `serve` traffic takes, given the checkpoint's generation config.

- 2026-11-21 (coordinator): #10943 cannot jump the line. fleetq has no reorder, and `--force` / `--no-queue` need the user's
  OK, which main won't ask for over one job. It stays at #18; its plan is good. Keep the f64 total sequential: don't trade bits
  for the last 0.26 ms. The per-model default (Inkling on, others off) is approved. Open item: row-parallel sampling for
  batched serve (B > 1, one row per pool worker), from an earlier coordinator message this lane never received (lost in
  compaction). Build it now, gated by batched sampled tokens identical at B = 4 / 16 on the tiny model on spark2, so it can
  ride in #10943 if the build lands before that job starts.

- 2026-11-21 (coordinator, answers + next build):
  1. **Waiting on #10943:** the lane starts its own background waiter that returns when #10943 ends.
  2. **B = 8 instead of 16 in the CPU gate:** accepted.
  3. **The 35B B = 16 gate:** no. It costs a tight-memory endor job and does not check Inkling. The real Inkling check runs in
     ink-batch after both branches merge: batched-sampled == solo with the pool at its default and at `INTARSIA_HOST_THREADS=1`.
  4. **json_schema on the batched sampled path:** write that test now. Tiny model, CPU, pool on vs `INTARSIA_HOST_THREADS=1`,
     bitwise tokens, on spark2 (`--load cpu`).
  5. **The ink/metal merge waits for #10943:** agreed. The criterion is unchanged: 5k and t0.7 within 1 % or faster.
  6. **Next build, after #10943 (the bigger lever):** a device sampler for plain-temperature rows. Serve's default request
     (temperature 1.0, no top-k or top-p) downloads the full 200k row and samples on the host: about 22 ms per token against
     16.9 greedy.
     - Method: Gumbel-max on the device, argmax(softcap(l) / T + g) with g = -log(-log(u)). The device already has the softcap
       and `argmax_f32_v4t`.
     - RNG: a counter-based Philox keyed on (seed, row, position), so a fixed seed gives fixed output.
     - Scope: only rows with no top-k, top-p, min-p, penalties, stop masks or constraints. Every other row keeps today's path.
     - Opt-in, default off, the lane names the switch. It changes which token a given seed produces, so a default flip is a
       separate decision.
     - Tests: a chi-square test of the draw distribution against the exact softmax (small vocab, many draws); a fixed seed
       gives the same tokens across runs; batch == solo bitwise.
     - Measure: sampled decode tok/s at t1.0 with the switch on vs off, and vs greedy, in a quiet endor slot.

- 2026-11-21 (coordinator): the `GumbelArgmax` op contract is OK (as the lane proposed: inputs logits `[R,V]` f32 and keys
  `[R,4]` u32 = (seed lo, seed hi, absolute position, bits of 1/T), output `[R]` i32, attr `scale` = softcap cap; Philox4x32-10
  with key (seed lo, seed hi) and counter (token id, position, 0, 0), no batch row; u = ((x >> 8) + 0.5) 2^-24; ties go to the
  smaller id; cpu reference + Metal only, CUDA and HIP fall back to the host path; `INTARSIA_GUMBEL_SAMPLING=1`, default off
  for every model; work stays on `ink/hostgap-gumbel` until #10943 has run; one endor job queued after #10943), with three
  changes:
  1. Discriminant 96, not an append: 96..=99 are reserved for ink-hostgap (76..=95 are reserved for other lanes, so an
     unnumbered append can collide at merge). The reservation comment says so.
  2. `LastGumbel` also excludes rows with min_p > 0, any repetition / presence / frequency penalty, a logit_bias, or a logprobs /
     top_logprobs request (they need host logits), with a test that each keeps the host path.
  3. The chi-square gates run at T 0.7 and T 1.0, with the softcap on.

- 2026-11-21 (coordinator, revising change 2): no `GenRequest::host_sampling`. The engine applies none of min_p, penalties or
  logit_bias: it drops them on both paths, so excluding those rows from the device sampler changes no output, and the field
  would touch 7 struct literals in files other lanes edit. Instead: (1) keep the test that logprobs / top_logprobs
  (`trace_logits`) keep the host path; (2) a comment at the `LastGumbel` eligibility check says any future feature that changes
  logits before sampling (min_p, penalties, logit_bias) must exclude `LastGumbel`.
- 2026-11-21 (lane, reported to main): the approved u = ((x >> 8) + 0.5) 2^-24 is not usable in f32. At x >> 8 = 2^24 - 1 the sum
  16777215.5 rounds to 2^24, so u = 1 and g = +inf: about 1 token in 84 at V = 200058 would be uniform noise. It also leaves only
  24 bits near u = 1, where the winners come from. The op takes w = (f32(x) + 0.5) 2^-32 (u = 1 - w), E = -ln(1 - w) by its
  series below w = 1/16 and by `ln` above, g = -ln(E): full relative precision at the top, no u = 1.
- 2026-11-21 (coordinator): the w = (f32(x) + 0.5) 2^-32 form is OK (series below 1/16, precise ln above). Addition: f32(x)
  rounds to 2^32 for x >= 2^32 - 128 (w = 1, E = inf, g = -inf: harmless, but) so w is clamped to at most 1 - 2^-24 and g is
  always finite, which keeps the admission and NaN / inf checks clean. Two more tests: the x = 2^32 - 1 and x = 0 edge cases,
  and a property test that g is finite for random x. Keys as an i32 `[R,4]` tensor of u32 bit patterns: OK.
- 2026-11-21 (coordinator, rule for all Inkling lanes): the session and every worktree run on endor, so a local cargo build /
  check / test or shader compile loads endor's CPU and skews every quiet timing job there (a candidate cause of the 5k / t0.7
  "run-order confound"). No local builds or tests on endor outside a fleetq slot. macOS compile checks: bespin
  (`fleetq run --host bespin --load cpu --mem 6 --expect <min>`, a `git archive` copy in `~/intarsiaml-<lane>`, toolchain in
  `~/.cargo/bin`; no GPU timing or real model there). CPU tests: spark2. GPU work: endor fleetq jobs. Reading, git and editing
  locally are fine.

## Measurement (before)

### The integration trace, re-read (no new GPU run)

`.scratch/cbstat.py` over the integration lane's cb trace (`.worktrees/ink-metal/.scratch/prof_cb.jsonl`, `generate`,
19-token prompt, 65 tokens, default `--profile-every 16`): decode runs with device-side token feedback (Inkling's state is
`Ring` / `AttnKv` slots, no `Recurrent` one, so `Model::feedback_ok` holds). Each step is two command buffers: the first 16
dispatches (committed when encoded, queued behind the previous step) and the held rest of 597, committed once the previous
token is read.

| per step (57 steps) | median | mean | min | max |
|---|---:|---:|---:|---:|
| wall (start to next start) | 15.83 ms | 19.99 ms | 15.24 | 98.17 |
| GPU busy | 15.77 ms | 16.65 ms | 15.24 | 32.67 |
| GPU idle | 0.001 ms | 3.34 ms | 0.001 | 65.49 |
| release slack (rest committed before the GPU needed it) | 0.305 ms | 0.295 | 0.124 | 0.334 |
| GPU time of the 16-dispatch buffer | 0.424 ms | | 0.403 | 0.450 |

The mean is the brief's 19.8 / 15.8 ms; the median is the normal step. The difference is the profiled steps: every 16th step
synchronizes around every op (`Model::exec_call_one` with `profile`), 419 one-dispatch buffers, 77-82 ms each. This agrees
with Fable's 0.1.

### Summary: before and after

| per token, Inkling-2-Mini Metal decode, endor M5 Max | before (integration lane, `generate` default `--profile-every 16`) | after (this branch, default `--profile-every 0`) |
|---|---:|---:|
| wall, mean (cb trace, 19-token prompt) | 19.99 ms | 16.86 ms |
| GPU busy, mean | 16.65 ms | 16.67 ms |
| GPU idle, mean | 3.34 ms (16.7 %) | 0.19 ms (1.1 %) |
| GPU idle, median | 0.001 ms | 0.001 ms |
| wall / busy / idle at a 5051-token prompt, median (mean idle) | not traced | 18.08 / 18.08 / 0.001 ms (0.005; 0.03 %) |
| greedy decode tok/s, short prompt (`generate`, no profiler) | 50-52 (base binary at p0: 63.1) | 63.7, 64.2 |
| greedy decode tok/s, 5051-token prompt | 48.3 (base binary at p0: 57.3) | 55.8 |

Most of the 50 -> 64 is the profiler default: the base binary at `--profile-every 0` already runs at 63.1. The code's own greedy
gain is about 1 %, the faster argmax.

Where the 4 ms went: it was the profiler, not the host. With `--profile-every 16`, every 16th step synchronises around each op
(419 waited command buffers, 77-82 ms). Unprofiled steps already ran back to back. Token feedback holds each step's rest until
the previous token is read, and that rest is committed 0.30 ms (median) before the GPU needs it.

Target, wall <= GPU busy + 0.5 ms:
- **Greedy:** met at both prompt lengths.
- **Sampled decode:** not met. See item 4: 2.4-3.6 ms of host work per token while the GPU is idle.

The cb traces of #10840 carry the cb profiler's own cost, so their busy time (16.6 ms) is above the 15.6-15.7 ms per token
that the unprofiled tok/s implies.

### Measurement A and the timing runs (fleetq #10840, endor, quiet gpu + cpu, 22:28-22:31)

Script: `.scratch/measT.sh`; raw output: `.scratch/mT/`.
- **base:** ink/metal e61f392fd.
- **b1:** the branch at 63e0d1895, the binary #10839 checked.
- **b3:** the branch at 5b487b79c, used for the host-profile, cb-trace and Qwen3 runs.

All runs are `generate --profile-every 0` with no profiler in the environment. They are single runs, and the first run of the
job was cold.

| run | base decode tok/s | b1 decode tok/s | md5 (base = b1) |
|---|---:|---:|---|
| greedy, 256 tokens, 19-token prompt, run 1 | 56.85 (cold: prefill 1185 ms) | 63.68 | d64363ea |
| greedy, run 2 | 63.08 | 64.18 | d64363ea |
| temperature 0.7, seed 7, 256 tokens | 54.91 | 51.86 | c10264d0 |
| greedy, 128 tokens, 5051-token prompt | 57.33 | 55.76 | 6376fdf5 |

Measurement A (the coordinator's three numbers):
- **b1:** greedy 63.68 and 64.18 tok/s, temperature 0.7: 51.86 tok/s.
- **base:** greedy 63.08 tok/s (warm run), temperature 0.7: 54.91 tok/s.

The greedy gain of b1 over base, about 1 %, is argmax's 0.16 ms of GPU time per step.

The temperature-0.7 and 5k differences between base and b1 (-5.5 % and -2.7 %) are single runs, and there is a confound:
every base run finished before any b1 run, with no interleave. The code does not explain either difference:
- **Sampled decode** never dispatches argmax. There b1 differs from base only by host-profile spans that are off (one atomic
  load each).
- **At 5k** decode is GPU-bound (idle 0.03 %), and b1's only GPU-side change is the faster argmax.

One quiet ABAB job of about 5 min would settle it. Not run: the lane stopped. These are `generate` numbers, not localeval runs
(see "Not done").

The final code (5b487b79c, b3) with `INTARSIA_HOST_PROFILE=1` gives the same tokens:
- greedy feedback and plain path: md5 d64363ea
- temperature 0.7: c10264d0

The profiler changes no output on the real model either.

Host profile (b3, `INTARSIA_HOST_PROFILE=1`, 255 decode steps, ms per step):

| path | wall | encode | sync (GPU) | softcap | sampler (`accept`) | other host | host gap (GPU idle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy, token feedback (default) | 16.77 | 0.99 (`launch`, overlapped: held behind the running step) | 15.74 | - | - | 0.04 | 0.015 |
| greedy, plain path (`INTARSIA_TOKEN_FEEDBACK=0`) | 16.23 | 0.98 (`step-setup`; the GPU starts after the first 4 dispatches) | 15.23 | - | - | 0.01 | 0.018 |
| temperature 0.7 | 19.59 | 1.21 | 15.99 | 0.68 | 1.67 | 0.05 | 2.40 |
| temperature 1.0 (checkpoint default) | 21.98 | 1.71 | 16.72 | 1.37 | 2.10 | 0.09 | 3.56 |

The cb trace of the temperature-0.7 path (b3) agrees: 21.13 ms wall against 16.51 ms busy (median), 4.55 ms idle, 29 % of
the step.

`--profile-every` default on other models (b3, Qwen3-8B, 128 tokens):
- new default: 31.06 tok/s, `top ops: off (pass --profile-every 16)`
- `--profile-every 16`: 29.56 tok/s
- md5 749132e0 is the same for both

## Correctness (fleetq #10839, endor, 22:26-22:28)

base = ink/metal e61f392fd, b1 = ink/hostgap at 63e0d1895 (argmax v4t, the solo host profile, the generate profile default).
Script `.scratch/measC.sh`, raw `.scratch/mC/`.

| check | result |
|---|---|
| Metal tests `lowlat` / `dense` / `g0b_small` (incl. `argmax_tail_path_matches_host`, `argmax_tail_kernel_is_the_vector_kernel_on_aligned_rows`, `argmax_vector_path_matches_host`, `argmax_ties_nan_and_all_neg_inf`, `argmax_rows`) | 10 + 7 + 7 passed |
| `kernels test --backend metal --op argmax` (sweep now has V = 101) | `argmax_f32`, `argmax_bf16` ADMITTED, 4 cases each, max_abs 0 |
| greedy 256 tokens (lighthouse prompt, `--profile-every 0`), token feedback | md5 d64363ea... base = b1 |
| the same on the plain path (`INTARSIA_TOKEN_FEEDBACK=0`) | md5 d64363ea... base = b1 = feedback path |
| t 0.7 seed 7, 256 tokens | md5 c10264d0... base = b1 |
| G2 short (oracle W4A16 top-20, `--max-mean-kl 4e-3`) | PASS both |
| G2 short self-golden (top-64 rows, prefill + decode modes) base vs b1 | 48 tensors, 0 differ: bitwise identical |
| `plan show` Qwen3-8B / Qwen3.6-35B-A3B-NVFP4 / Inkling-2-Mini, base vs b1 | stdout identical (21 / 59 / 31 lines); stderr differs only in the "loaded in X s" timing line |

## Item 3: argmax (commit 942f72f31)

The trace's `argmax_f32` dispatch takes 170-177 us for one 200058-element f32 row (0.8 MB; at 560 GB/s that is 1.4 us of
bytes). The cause is the shape, not the kernel. `MetalBackend` takes `argmax_f32_v4` (1024 threads, 8 float4 loads in flight per
thread; 12-15 us on Qwen3's 151936 in docs/results/2026-10-14-metal-phase-b.md) only when `V % 4 == 0`. Inkling's head is the
unpadded vocab, V = 200058, and 200058 % 4 = 2. So it fell back to the 256-thread scalar `argmax_t`: one load in flight per
thread, latency-bound.

`argmax_f32_v4t` (kernels/metal/dense_ops.metal) is `argmax_f32_v4` for any V:

- Row r starts at element r * V. Its 0-3 elements before the first 16-byte boundary go to threads 0..head-1 as single loads,
  the aligned body goes as float4 chunks with the same 8-in-flight loop, and the 0-3 elements after the last chunk go to
  threads 0..tail-1.
- Each thread visits its elements in increasing index order (head < body < tail), so strict `>` keeps the lowest index of its
  own maximum.
- The cross-thread resolution is `argmax_f32`'s: the same index for every input (lowest index of the maximum, NaN never wins,
  an empty or all -inf row gives 0).

The launcher keeps `argmax_f32_v4` for V % 4 == 0, so Qwen3 / the 35B run the kernel they ran before. `INTARSIA_METAL_ARGMAX_V4T=0`
restores the scalar kernel. Checks:

- `tests/lowlat.rs::argmax_tail_path_matches_host`: V in {200058, 151937, 1001, 4099, 7, 5, 3, 2, 1}, 1-5 rows (rows starting
  4 / 8 / 12 bytes past a boundary), byte offsets 0 and 16; ties of the maximum at the first and last element, NaN / -inf /
  +-0 mixtures, all -inf / NaN rows.
- `argmax_tail_kernel_is_the_vector_kernel_on_aligned_rows`: v4t by hand on V % 4 == 0 rows gives v4's indices.
- The admission sweep gains V = 101 and 4099.
- `examples/argmax_cost.rs` times the three kernels.

GPU results (#10839, #10840; `examples/argmax_cost.rs`, us per dispatch, median of 15 buffers of 200 dispatches):

| kernel | V | hot (one row) | rotating (48 rows, 38 MB) |
|---|---:|---:|---:|
| `argmax_f32` (scalar, what Inkling ran) | 200058 | 83.4 | 235.8 |
| `argmax_f32_v4t` (new) | 200058 | 13.5 | 13.9 |
| `argmax_f32` (scalar) | 151936 | 63.8 | 175.7 |
| `argmax_f32_v4` (Qwen3's, unchanged) | 151936 | 11.3 | 11.9 |
| `argmax_f32_v4t` on aligned rows (not taken there) | 151936 | 11.1 | 11.8 |

The target was <= 25 us, met: 172 us in the decode trace becomes about 14 us, about 0.16 ms of GPU time per greedy step. The
Metal tests and `kernels test --op argmax` pass, and the greedy md5 and G2-short self-golden are bitwise base = b1 (#10839).

## Item 4: the sampled path (code reading; host-profile numbers pending)

Sampled decode (temperature > 0) does not take token feedback: `Model::feedback_ok` is reached only for greedy
(`HeadMode::LastArgmax`). Every sampled step is therefore the plain path, with the GPU idle from the end of the step until the
next step's first commit. Per token, on the host:

1. `synchronize`, then the 0.8 MB f32 logits download (`head_to_host`).
2. The soft-cap `30 * tanh(x / 30)` over all 200058 logits (`Model::softcap_logits`).
3. `Sampler::sample` (`sample_row_in`). At the checkpoint's default (`generation_config.json`: temperature 1.0, no top-k, no
   top-p; what `serve` applies to a request that sends none) this is 200058 sort keys, a 3-pass radix sort of all of them, and
   200058 f64 `exp` plus divides, then the walk. With top_k = 20 it is a `select_nth` over 200058 instead of the sort.
4. `on_token`, then the encode of the next step's first dispatches.

Host cost, measured: `sampler::tests::sampler_host_cost` on endor's CPU inside #10840 (n = 200058, median of 41).

| step | ms |
|---|---:|
| soft-cap (copy + tanh) | 0.67 |
| host argmax | 0.20 |
| `sample_row_in`, T 1, no top-k / top-p (checkpoint default) | 1.43 |
| `sample_row_in`, T 0.7, no top-k | 1.40 |
| `sample_row_in`, T 1, top-p 0.95 | 1.43 |
| `sample_row_in`, T 0.7, top-k 20, top-p 0.95 | 0.19 |
| `sample_row_in`, T 1, top-k 64 | 0.19 |

In the decode loop the same work costs more than in the microbench. There the host profile shows softcap 0.68-1.37 ms and
sampler 1.67-2.10 ms per token, a host gap of 2.4-3.6 ms; the bench runs with a warm cache.

Which path real `serve` traffic takes. The `generation_config.json` of the checkpoint is
`{"do_sample": true, "temperature": 1.0, "eos_token_id": 200006, "pad_token_id": 199999}`, which sets no `top_k` and no `top_p`.
`SamplingDefaults::effective` (crates/intarsia-server/src/api.rs) fills a field the file lacks with the built-in: temperature
1.0, top_p 1.0, top_k 0 = off. So:
- **A request that sends no sampling fields** takes the full-vocabulary path: T 1, no top-k, the 1.43 ms row above.
- **A request that sends `top_k`** (1..=200057) takes the `select_nth` path, 0.19 ms.
- **A request that sends `top_p` only** still sorts the whole vocabulary.
- **`localeval speed`** pins temperature 0 (`src/localeval/speed.py`), so its decode numbers are the greedy token-feedback
  path and do not see the sampler at all.

This is the remaining host gap of Inkling decode, about 12-16 % of a sampled step. Served requests that send no sampling
parameters get it: `serve` applies the checkpoint's temperature 1.0, no top-k. Not built here; the lane stopped. The bitwise
option is below.

What can and cannot remove it while keeping the token bitwise the same:

- A device sampler cannot be bitwise the host one: the host softmax and walk are in f64, which Metal GPUs do not have. A device
  sampler would draw with the same law but not the same draw for a given seed. That is a numerics change and needs a decision
  (CUDA / HIP made the same trade for batched top-k rows, `INTARSIA_DEVICE_SAMPLING`, with an exactness argument for
  `top_k == k`).
- No partial sort shortcut exists at temperature 1 with no top-k. The walk's total is the f64 sum of every term in sorted
  order. A term drops out of that sum exactly only below half an ulp of the running sum (2^-53 at sum 1), i.e. for logits
  more than 36.7 * T below the maximum, and with soft-capped logits in (-30, 30) almost every token stays in.
- What does keep it bitwise: split the per-element work across threads. The soft-cap, the key build, the exp and the divides
  are all per element. The radix sort is stable, so per-chunk histograms and scatter give the same order. The f64 sum stays
  sequential (about 0.15 ms). Worth building only if the measured host share is large: the profile columns `softcap` and
  `accept` of `hp_t1` / `hp_t07` in #10840.

## Item 3b: the bitwise host sampler, option (a) (commits aed3dc496, 8d8ac6c78, a2f2b87b9)

**What it does:**
- `crate::par` is a persistent host pool. By default it runs min(8, cores) threads, counting the calling thread. Between task
  lists a worker spins for 300 us, then blocks. One caller uses it at a time; a second, concurrent caller runs its tasks
  inline. A nested `run` also runs inline.
- Every task writes only its own outputs, so no result depends on which thread ran what.
- **Default (approved by the coordinator):** a per-model switch set at load, on for Inkling-2-Mini and off for every other model.
  `INTARSIA_HOST_THREADS=N` turns it on for every model with N threads; `1` turns it off.

**Soft-cap:** for a row of at least 32768 entries, `cap * tanh(x / cap)` runs per element on the pool. Each element gets the
same expression as before, so the result is bitwise the same.

**Sampler:** for a row of at least 32768 entries with top-k off (`top_k` 0, Inkling's checkpoint default), `sample_row_par`
does:
- the sort keys and the NaN check, on the pool (a NaN row falls back to the comparator path, as before)
- a stable chunked LSD radix sort on the pool (8-bit digits, 4 passes)
- the softmax exp terms and their divides on the pool
- the f64 total, the top-p cut, the cut's sum and the walk sequentially, in the same order as `cut_in`

It gives the same probabilities as before, so the same token.

**Why each step stays exact:**
- The keys are distinct (`key << 32 | index`), so every correct sort gives the same order.
- The f64 total stays sequential, as the coordinator asked: it is the floor, about 0.10 ms on spark2.

**Two rules came from measurements:**
- A top-k row keeps the one-thread path. On the pool, top-k 20 and 64 went from 0.27 to 0.34 ms, because `select_nth` and its
  few candidates gain nothing from threads.
- The radix sort uses 8-bit digits in 8 parts. The first version used 11-bit digits in 16 parts and took 0.63 ms of the
  1.0 ms: each part's run of a bucket was about 6 keys, less than a cache line, so the parts shared nearly every line they wrote.

**Batched serve** (`Scheduler::accept_rows`, coordinator item):
- With the pool on and B > 1, each row of a decode step is accepted as its own pool task, and the results are handled in row
  order afterwards.
- A sequence's accept touches only that sequence (sampler, seeded RNG, constraint), so its token is the one the sequential
  loop gives.
- `BatchModel::Out` gains a `Send` bound.
- Logits rows recycled on a worker go to a shared spill, which the step's thread takes back; no fresh 0.8 MB rows means no
  GB10 page faults.
- Not done: ink/batch adds the batched soft-cap (`softcap_logits` per row in model/batch.rs). It uses the within-row pool, and a
  row-parallel soft-cap would be a small follow-up in that file.

**Tests:**
- `sampler::threaded_sampler_matches_the_one_thread_path`: every chunk count, ties, +-0, -inf, NaN, top-k / top-p /
  temperature, all-equal rows; the public entry point with top-k on and off; the soft-cap bitwise.
- `par::tests::every_task_runs_once`: concurrent callers, panics.
- `batch_engine::batched_sampled_equals_solo`: every request sampled, B = 4 and 8, against solo. The cpu backend caps batch
  rows at `SERVE_MAX_BATCH` = 8, so `enable_batch(16)` is refused on the tiny model; rows > threads is covered by running at
  `INTARSIA_HOST_THREADS=3`. Run with the pool at 8, 3, 1 and the default.

**spark2 microbench** (Grace, 8 threads, warm, 200058 logits, ms; one thread -> pool):

| step | aed3dc496 | 8d8ac6c78 | a2f2b87b9 |
|---|---:|---:|---:|
| soft-cap | 1.05 -> 0.23 | 1.05 -> 0.25 | 1.05 -> 0.26 |
| sampler, T 1, no top-k | 1.84 -> 1.06 | 1.87 -> 1.01 | 1.86 -> 0.94 |
| sampler, T 0.7, top-k 20, top-p 0.95 | 0.28 -> 0.34 | 0.27 -> 0.27 | 0.28 -> 0.28 |

These are `--load cpu` slots, shared with other lanes' builds, so the numbers carry noise of about 0.1 ms (the a2f2b87b9 soft-cap
read 0.37 ms in #10951 and 0.26 ms in #10958).

Phase split of the threaded sampler on ce292492b (before the radix fix): keys 0.04, radix sort 0.63, exp 0.15, f64 total 0.10,
divide 0.04, walk 0.00 ms (median walk depth 22). After the fix (a2f2b87b9, #10958): keys 0.04, radix sort 0.53, exp 0.17,
f64 total 0.10, divide 0.03 ms. The cache-line explanation was only part of it: the sort fell 0.63 -> 0.53 ms, not to the
0.2 ms expected. It is the largest phase left. Not chased further: spark2's shared CPU cannot settle it, and the coordinator
accepts the remainder.

**spark2 checks:**
- **#10951 (a2f2b87b9):** core lib 237/237; core scheduler / herd / fault_inject 11 + 52; docs_env_vars; the cli build.
  `batched_sampled_equals_solo` failed in setup only (B = 16 refused by the cpu cap).
- **#10958 (e4299fe24):** the same suites pass, plus `batch_engine` 10/10 with the pool at 8, 3, 1 and the default.

**Endor** (#10943, in-loop host profile, against the <= 1 ms target for soft-cap + sampler at t1.0): ENDOR_SAMPLER_PLACEHOLDER

**Gate changes and limits, reported to main:**
- **B = 16 became B = 8.** The coordinator asked for B = 4 / 16 on the tiny model; the cpu backend allows at most 8 batch rows.
  The case of more rows than threads is covered by the run at 3 threads.
- **Not reachable on Inkling yet.** In this branch `enable_batch` refuses Inkling (`Block::Ink(_) => false`), so the default-on
  row-parallel accept first runs on Inkling once ink/batch merges. Before then nothing exercises it on the real model.
- **A possible B = 16 real-model gate:** `batch_metal_real.rs::real_35b_batched_engine_equals_the_local_engine` (BatchedEngine,
  sampled rows, Metal) run with `INTARSIA_HOST_THREADS=8`. Offered to main, not run.
- **json_schema on the batched sampled path (coordinator item 4):** `structured::batch_tests::json_schema_batched_sampled_equals_solo`
  (commit 280d8716c) runs the real llguidance constraint on half of 16 sampled requests at B = 4 and 8 against `LocalEngine`.
  The grammar env is built over the tiny fixture's byte vocabulary. Results on spark2 #10968: bitwise equal with
  `INTARSIA_HOST_THREADS=8` (the constrained rows' mask, apply, sample and commit on pool workers) and with `=1`; server lib 59/59.

## Pending: #10943 (endor, quiet, `--expect 35`)

Results land in `.scratch/mAB/summary.txt`. Script: `.scratch/measAB.sh`. `.scratch/b4.ready` = e4299fe24; the job builds HEAD
at its start as b4 and refuses a HEAD that lacks e4299fe24.

1. ABAB base vs b1, 5 reps per arm, after one warm-up per arm: short greedy, 5k greedy, t0.7 seed 7. Medians and deltas, for the
   merge rule (5k and t0.7 within 1 %).
2. The sampled md5 gate b1 vs b4: t0.7 and t1.0 at seeds 7 and 11, and t0.7 with top-k 20 / top-p 0.95. Also b4 with the pool
   on vs b4 at `INTARSIA_HOST_THREADS=1`.
3. ABAB b1 vs b4, 5 reps per arm: 5k, t0.7, t1.0. b4 is what would merge.
4. The b4 host profile at t1.0 and t0.7 (soft-cap + sampler in the loop, for the <= 1 ms target), the microbench, and the phase
   timing.

Merging ink/metal into this branch is held until #10943 has run, so b4 stays the code the gates measure.
`git merge-tree HEAD ink/metal` is clean today.

## Item 5: the speculative draft cycle without a host round trip (proposal, not built; dropped)

**Dropped 2026-11-21 (coordinator).** ink-mtp's per-cycle split at k3, synced:

| part | ms per cycle | where |
|---|---:|---|
| S1 ingest | 0.94 | host + GPU |
| drafter block | 2.64 | GPU |
| head | 1.34 | GPU |
| Markov + small downloads | 0.70 | host |
| S3 verify, extra wait | 0.04 | host |

Host total is about 1.2-1.6 ms of a 35 ms cycle, under 5 %. So H1 / H2 are not built, and M1-M4 are not requested. The text
below is the proposal as it was sent, kept for reference.

Read from `ink/mtp` 7c0a2a956 (`crates/intarsia-core/src/model/dspark.rs`, `model/dflash.rs`); the DSpark cycle has three host
syncs, each followed by host work the GPU waits through:

| sync | where | host work before the GPU has more |
|---|---|---|
| S1 | `dspark_ingest` (lazy flush of the verified rows) | download 4 taps x rows x H, `mix` (alpha-weighted tap sum + rms norm) per draft layer, upload per layer |
| S2 | `dspark_draft`, after the block pass and the base head | download hw / fw and the device top-64 of each base row (7 x 64 ids + values, about 3.6 KB; corrected, see below), the r-dim Markov recurrence over the 64 candidates and the gate |
| S3 | `forward_verify_run` | the verify rows' argmax (4 B per row), accept, `on_token` per accepted token |

The end state is the token-feedback pattern (`model/feedback.rs`) applied to a cycle: the host reads one small record per cycle
and has already encoded the next cycle behind it.

What ink-hostgap would provide (generic, no op-sequence dependence):

- H1. A per-cycle record `[n_acc, bonus, k_eff, drafts[7]]` (i32, 40 B) read with the existing `Backend::fence` / `read_after`,
  the only host read of a cycle.
- H2. Encode-ahead: cycle c+1 encoded and held (`Backend::hold`, as `feedback_step`) before cycle c's record is read, released
  when it is. A stop or a constrained request discards the held cycle (`feedback_discard`).

What it needs from the drafter and the target (ops and op contracts: ink-mtp, m-attn, dfuse; not this lane):

- M1. The ingest mix on the device: the weighted sum of 4 taps then RmsNorm. All k+1 verify rows are ingested
  unconditionally; rejected rows land past the draft cache's device length and are overwritten, as the block scratch already is.
- M2. The Markov head on the device (the recurrence over the 64 candidates and the gate), writing drafts and k_eff into the
  record. The top-64 is on the device already. Corrected 2026-11-21 (ink-mtp via the coordinator): the first version of this
  note said Metal had no `topk_rows`, so the base rows (2.8 MB at kmax 7) came to the host. ink/mtp c770aaef3 added
  `Backend::topk_rows` on Metal (two dispatches, f32 / bf16, k <= 64, sampler-order ties; `supports_topk` stays false), so S2 now
  downloads about 3.6 KB, and the host Markov head over 64 candidates is pure arithmetic.
- M3. A verify of kmax + 1 rows always (rows past k_eff computed and ignored), so the verify can be encoded before k_eff is
  known, and an accept op: drafts vs the verify argmax, giving n_acc and the bonus token.
- M4. Every position-dependent op of the next cycle takes its position from a device buffer, because `st.len` after a cycle
  depends on n_acc. The generic Metal attention / KV append / rope have this (`pos_bind`, `use_dev`); Inkling's own ops
  (`InkKvPrologue`, `RelSinkAttn` / `InkAttnDecode`, the ring slot index) take host `pos0` today. This is the expensive part, and
  an op-contract change for m-attn and dfuse.

Order of value: S1 and S2 sit in the middle of the cycle, and S3 costs only the host work before the next draft. With the device
top-64 (correction above), S2's host part is a 3.6 KB read and arithmetic over 7 x 64 candidates. Which sync is worth removing
is for ink-mtp's per-cycle split to show (flush / block / head / markov / wall / ingest). Before any of
it, the per-cycle host ms should be measured. `INTARSIA_DSPARK_PROFILE=1` prints flush / block / head / step0 / ingest. The cycle
is about 20-25 ms modelled, so a 1 ms host gap per cycle is 4-5 %.

Batched decode (ink-batch keeps graphs off there): the same record per row, with the batch step's existing device top-k path
for sampled rows (`HeadMode::LastTopK`). Metal now has `topk_rows` (ink/mtp c770aaef3), but `supports_topk` stays false, so the
batch step does not take it on Metal yet.

Spec-loop host profile (commit below, approved by ink-mtp): with `INTARSIA_HOST_PROFILE=1` the solo loop's eager draft + verify
cycle (`Seq::prepare_with`, the branch DSpark and DFlash take) is its own class `spec-cycle`. Its columns:
- `draft` and `verify` (each including its device time and syncs), netted out of `prepare`
- `rows` = tokens per cycle
- `on_token` of the cycle's tokens
- `gpu-idle-gap` = host time from the verify's token download to the next draft's start

The S1 / S2 host work inside `draft` is ink-mtp's `INTARSIA_DSPARK_PROFILE` split.

## Item 6: the device sampler (Gumbel-max, opt-in; branch `ink/hostgap-gumbel`, a6cb2dce8; spark2 ran the same code as c4c1f48e6 before a rebase)

Built on a side branch so that #10943 still measures the `ink/hostgap` it was queued for (measAB builds b4 from that HEAD). It
merges into `ink/hostgap` after #10943 has run.

**What it does.** A plain-temperature request (temperature > 0, `top_k` 0, `top_p` 1, no constraint, no logit trace, so no
logprobs; not while it speculates; host rows up to `min_tokens`) draws each token on the device. Serve's default Inkling request
(the checkpoint's temperature 1.0, no top-k or top-p) is such a request. Only 4 bytes come back instead of the 200058-float row.
On Metal the step runs the greedy token-feedback pipeline (step N+1 is encoded before step N's token is read), so the 3.56 ms
host share of a t1.0 step (soft-cap 1.37 + sampler 2.10, item 4) and the 800 KB download leave the critical path.

**Definition** (`crates/intarsia-core/src/gumbel.rs`; the cpu op calls it, the Metal kernel mirrors it):
token = argmax_i(c(l_i) / T + g_i), ties to the smaller id; c = the head's softcap, exactly `sampler::softcap`'s formula (cap
0 = none); g_i = -ln(E_i), E_i = -ln(1 - w_i) (its series to w^5 below 1/16), w_i = min((f32(x_i) + 0.5) 2^-32, 1 - 2^-24);
x_i = word 0 of Philox4x32-10 with key (seed lo, seed hi) and counter (token id i, absolute position, 0, 0). The position is the
number of tokens before the sampled one. No batch row enters the key, so a row draws the same token solo or batched. A request
without a seed gets a fresh one (time mixed with a process counter).

**Where:**
- `Op::GumbelArgmax` (discriminant 96; 96..=99 reserved for this lane): logits `[R,V]` f32, keys `[R,4]` i32 (u32 bits: seed lo,
  seed hi, position, bits of 1/T), out `[R]` i32, attr `scale` = cap. cpu reference / fast (one shared function); Metal
  `gumbel_argmax_f32` (one threadgroup of 1024 threads per row, `precise::` log / tanh). CUDA and HIP list it unsupported.
  Admission sweep rows: R 1 / 5, V 100 / 101 / 4096 / 4099 / 200058, cap 0 / 30.
- `HeadMode::LastGumbel(GumbelHead { seed, inv_t })`: `Model::forward` (prefill head and plain steps), `feedback_step` (token
  feedback), `forward_batch` (per-row keys uploaded before the step, one `GumbelArgmax` over the step's rows).
  `Model::gumbel_ok`: the switch, one device (no peers, no remote rank, replicated head) and a selectable kernel, decided on
  first use. Otherwise every request keeps the host sampler.
- Keys: a per-request table, filled before the request's first device-sampled step (prefill, plain path) and, for feedback
  decode, once before the loop for every token the request may still sample. A feedback step only reads its own row and never
  writes. Metal's `upload` synchronizes, and a step is encoded while the one before runs, so a per-step write would serialize
  the pipeline or race it.
- `Seq::head_mode`: the eligibility above. A comment there says any future feature that changes logits before sampling
  (min_p, penalties, logit_bias) must exclude `LastGumbel` (coordinator).
- Switch: `INTARSIA_GUMBEL_SAMPLING=1`, default off for every model (docs/tuning.md row). It changes which token a seed draws,
  so a default flip is a separate decision.

**Checks so far:**
| check | where | result |
|---|---|---|
| core lib (Philox Random123 known answers; x = 0 and x = 2^32 - 1 edge words; g finite for 10^6 random words and both ends; series vs ln; chi-square of 40000 draws at T 0.7 and 1.0, softcap 30 on, and off; fixed seed; ties; Seq eligibility incl. logprobs / trace, top-k, top-p, min_tokens, speculation, switch off, model without the sampler; fresh seeds) | spark2 #11013 (c4c1f48e6) | 247 passed, 0 failed |
| cpu backend, whole suite (GumbelArgmax ref == fast == definition, rows independent of R) | spark2 #11018 | 17 binaries ok, 0 failed |
| `gumbel_engine` (tiny model: fixed seed same stream on the same and a fresh engine; another seed moves the streams; batched B 4 / 8 == solo, device and host rows mixed; switch off = host sampler, host rows unchanged) | spark2 #11013 | pass |
| `batch_engine` 10/10, `engine` 10/10, `batch` 6/6, `docs_env_vars` 2/2 | spark2 #11013 | pass |
| macOS compile: Metal crate with tests, cli `--features metal`, Metal kernel-list pin | bespin #11014 (no Xcode there: the shader compiles on endor) | pass |
| Metal: shader build, `tests/gumbel.rs` (Metal == definition at V 101 / 4099 / 200058, cap 0 / 30; rows independent; key at an offset; chi-square of the Metal draws at T 0.7 / 1.0, softcap on / off), admission, md5 gates (greedy and host-sampled unchanged; device-sampled feedback on == off, twice == once, 5k), t1.0 timing on / off / greedy (short, 5k, 5 reps), host profile, G1-G3, plan show and Metal model tests for other models | endor #11017, queued behind #10943 (`.scratch/endorgumbel.sh`, summary `.scratch/gmb/summary.txt`) | pending |

**Kernel cost (estimate, to be measured in #11017).** One threadgroup per row runs on one GPU core. Per element the kernel does
Philox (10 rounds), up to two `precise::log`, and `precise::tanh` with the cap. That is about 150 integer and float ops, roughly
0.15-0.3 ms for Inkling's 200058 rows, against the 3.5 ms it removes. If it shows up, the next step is a split over several
threadgroups per row with a second reduce pass. The argmax with ties to the smaller id does not depend on the partition, so
the tokens stay the same.

**Batched decode and MTP draft verify.** Batched rows are covered (`forward_batch`, cpu-tested at B 4 / 8). Inkling batching
waits for ink/batch. Keyed (seed-exact) MTP drafting and sampled speculation keep the host sampler (`Seq::head_mode`
requires no speculation). A device-sampled verify would need the same key per drafted row (position = row's absolute
position): the Gumbel draw of a verify row equals the plain step's draw only if both use that key. That is possible, but not
built.

## CPU checks (spark2)

The checks ran on spark2 under `~/intarsiaml-ink-hostgap/src`: an exact `git archive` of the commit, without the docs/results
raw dirs or golden/. spark2 has no /opt/code.

- **#10873, on 6c0f4a5d9:** all pass.
  - hostprof 2/2.
  - `dflash_cpu` 7/7, with and without `INTARSIA_HOST_PROFILE=1`; the spec-on == spec-off id tests pass under the profiler.
    93 `spec-cycle` lines.
  - docs_env_vars 2/2 and the cli build pass.
  - This run showed 18-20 ms "gaps" from stale sync marks, fixed in 5b487b79c.
- **#10888, on 5b487b79c:**
  - `dflash_cpu` 7/7 both ways, with gaps now 0.007-0.033 ms. docs_env_vars and the cli build pass.
  - The new hostprof test failed: its expected count was wrong (2 gaps, not 1), not the code. Fixed in cd886d57b.
- **#10899, on cd886d57b:** hostprof 3/3, `dflash_cpu` with the profiler 7/7.

## Batched decode and MTP draft verify

- **Batched decode:** there was no host-gap work for it in this lane. ink-batch keeps graphs off there. A batched step would need
  the per-row form of the same pieces: a device-side token read per row, which `forward_batch` already does with its single
  download per step, and device top-k for sampled rows (`HeadMode::LastTopK`). Metal has `topk_rows` now (ink/mtp c770aaef3),
  but `supports_topk` stays false.
- **MTP draft verify:** see item 5 (dropped: host time is under 5 % of a cycle).

## Not done (lane stopped by the coordinator after the three jobs)

- **localeval:** speed numbers are from `generate` (engine-timed decode tok/s), not localeval. No `localeval speed` run was made.
  The brief asked for localeval; the coordinator's rescope replaced it with measurement A.
- **Gates:** G1, G2 long and G3 were not rerun on Metal; only G2 short was (PASS, self-golden bitwise base = branch). The argmax
  change picks tokens and does not touch logits, the greedy md5s are identical, and the branch adds no other numerics change.
- **Other-model tests:** the Qwen3 and 35B MoE real-model Metal tests were not run. Instead: `plan show` is identical for Qwen3-8B
  / 35B / Inkling, Qwen3-8B's greedy md5 is the same at the new default and at `--profile-every 16`, and Qwen3 / the 35B keep
  `argmax_f32_v4` (V % 4 == 0).
- **Encode-ahead, chunked commits, cheaper encoding, ICB replay:** not built. Greedy decode has no host gap to remove (0.015 ms
  per step), and Metal already commits in chunks (`FIRST_COMMIT` 4, `COMMIT_EVERY` 128).
- **The sampled path:** not built. It is the remaining real gap, 2.4-3.6 ms per token (item 4). The bitwise option is to split
  the per-element work (soft-cap, sort keys, exp, divides) across threads, with the radix sort stable per chunk and the f64 sum
  kept sequential. A device sampler would change the draw and needs a decision.

## Defaults and flags

- **Metal `generate --profile-every` default 16 -> 0** (63e0d1895): every model on Metal; CUDA / HIP / cpu unchanged. Coordinator
  go-ahead, LOG.md row. Revert that commit alone to undo it.
- **`argmax_f32_v4t`** (942f72f31): on by default, confirmed by the coordinator. It applies to f32 argmax rows with V % 4 != 0,
  on any model. **Revert switch: `INTARSIA_METAL_ARGMAX_V4T=0`**, the old scalar kernel. Of the fleet's Metal models
  only Inkling-2-Mini has such a head. Qwen3 / the 35B (V % 4 == 0) run the kernel they ran before. `INTARSIA_METAL_ARGMAX_V4T=0`
  turns it off. Same result by construction and by test.
- **`INTARSIA_HOST_PROFILE`** (e6e5d4314, 6c0f4a5d9, 5b487b79c, cd886d57b): opt-in, measures only.

## Commits (ink/hostgap, not pushed)

| commit | what |
|---|---|
| 942f72f31 | Metal `argmax_f32_v4t` for V % 4 != 0, `INTARSIA_METAL_ARGMAX_V4T`, tests, admission V = 101 / 4099, `examples/argmax_cost.rs` |
| e6e5d4314 | `INTARSIA_HOST_PROFILE` in the solo decode loop, `softcap` column |
| 63e0d1895 | Metal `generate` defaults to `--profile-every 0`; `top ops: off` line; docs/benchmarking.md note; LOG.md row |
| c1841a9b3 | ignored `sampler_host_cost` microbench |
| 6c0f4a5d9 | host profile: the spec cycle as its own class (engine.rs only, approved by ink-mtp) |
| 5b487b79c | host profile: no gap from a stale sync mark; greedy token read split into sync |
| cd886d57b | host profile: test expectation fix |
| f2c338e39, 2229ae919, 6855e486a, f71919c9a, 78e50d649, a3ba38768, this one | results doc |
