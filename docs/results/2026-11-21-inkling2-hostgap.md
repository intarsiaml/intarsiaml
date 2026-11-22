# Inkling-2-Mini Metal decode: host gap (lane ink-hostgap, 2026-11-21)

Status: in progress. Branch `ink/hostgap` (from ink/metal a88db4620, fast-forwarded to e61f392fd), not pushed.

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

MEASUREMENT_A_PLACEHOLDER

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

GPU results: ARGMAX_RESULTS_PLACEHOLDER

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

Host cost estimate: HOSTCOST_PLACEHOLDER

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

## Item 5: the speculative draft cycle without a host round trip (proposal, not built)

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
