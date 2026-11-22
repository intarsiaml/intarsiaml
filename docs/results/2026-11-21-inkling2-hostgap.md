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

## Item 5: the speculative draft cycle without a host round trip (proposal, not built)

Read from `ink/mtp` 7c0a2a956 (`crates/intarsia-core/src/model/dspark.rs`, `model/dflash.rs`); the DSpark cycle has three host
syncs, each followed by host work the GPU waits through:

| sync | where | host work before the GPU has more |
|---|---|---|
| S1 | `dspark_ingest` (lazy flush of the verified rows) | download 4 taps x rows x H, `mix` (alpha-weighted tap sum + rms norm) per draft layer, upload per layer |
| S2 | `dspark_draft`, after the block pass and the base head | download hw / fw / base logits (kmax x V bf16, 2.8 MB at kmax 7), top-64 per row, the r-dim Markov recurrence and the gate |
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
- M2. The Markov head on the device (top-64 of each base row, the recurrence, the gate), writing drafts and k_eff into the
  record. Metal has no `topk_rows` today (`supports_topk` false), which is why ink-mtp's device top-k falls back to the host.
- M3. A verify of kmax + 1 rows always (rows past k_eff computed and ignored), so the verify can be encoded before k_eff is
  known, and an accept op: drafts vs the verify argmax, giving n_acc and the bonus token.
- M4. Every position-dependent op of the next cycle takes its position from a device buffer, because `st.len` after a cycle
  depends on n_acc. The generic Metal attention / KV append / rope have this (`pos_bind`, `use_dev`); Inkling's own ops
  (`InkKvPrologue`, `RelSinkAttn` / `InkAttnDecode`, the ring slot index) take host `pos0` today. This is the expensive part, and
  an op-contract change for m-attn and dfuse.

Order of value: S2 is worth removing first. Its host work is the longest (2.8 MB download, top-64 over 7 rows of 200058, the
recurrence) and it sits in the middle of the cycle. S1 and S3 each cost only the host work before the next encode. Before any of
it, the per-cycle host ms should be measured. `INTARSIA_DSPARK_PROFILE=1` prints flush / block / head / step0 / ingest. The cycle
is about 20-25 ms modelled, so a 1 ms host gap per cycle is 4-5 %.

Batched decode (ink-batch keeps graphs off there): the same record per row, with the batch step's existing device top-k path
for sampled rows (`HeadMode::LastTopK`, CUDA / HIP only). Metal would need `topk_rows` first.
