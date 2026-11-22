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
