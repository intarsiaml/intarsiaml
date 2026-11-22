# Inkling-2-Mini speculative decoding with its DSpark draft (lane ink-mtp, 2026-11-21)

Status: in progress. Branch `ink/mtp` (not pushed, not merged).

## Briefs received

- 2026-11-21 (coordinator, SendMessage): Fable's decode review ranks this lane #2 (`.worktrees/ink-fable/decode-profile-review.md`,
  item 2, sections 3.5 and 4; read section 4 before designing the verify path).
  1. Shape: short confidence-gated drafts, 2-3 verify rows; the drafter rows share the verify's unembed launch.
  2. Estimated saving 3.0-4.4 ms of the real 15.76 ms step at acceptance 0.6-0.7, about 6 ms if consecutive tokens share experts.
  3. #10743 decides it: report acceptance by draft position and the expert-union size for 2 and 3 consecutive tokens
     (union over the 8-of-128 per layer) as soon as available.
  4. Device-side accept / verify readback is ink-hostgap's job: send interface needs to the coordinator, who relays.
  5. Measure Metal decode with `--profile-every 0` (real greedy about 63.4 tok/s; the default profiler adds about 4 ms/token).

- 2026-11-21 (coordinator, SendMessage): endor's queue is about 5 h deep. Move the real-model CPU token-identity job (#10807) to
  spark2's CPU (released on endor, resubmitted there); keep the Metal decode tok/s job (#10806) on endor. Every other CPU-only
  job of this lane goes to a Spark.
- 2026-11-21 (coordinator, SendMessage): priorities in order. (1) Fold the drafter's base-logit read into the verify's unembed
  (Fable §4.3; a separate 922 MB read is about 1.6 ms a cycle). (2) Move the tap mixes and the top-64 Markov head to the device;
  send the interface needs (what the device hands back per cycle: accepted count, next ids, draft confidences) after the
  per-cycle profile. (3) Tune the policy on a mixed set (greedy golden + prose + chat turns), report tokens per pass and real
  tok/s per policy; the default must never be slower than spec off on prose; the p1 >= 0.6 gate is the right shape.
  (4) Sampling: confirm whether temperature > 0 uses exact speculative rejection sampling; if not, keep spec greedy-only and
  say so. Speculation stays opt-in until #10806's timed numbers; default-on is the coordinator's call. If drafter changes land
  before #10806 starts, edit its script rather than queue a new job (within its --expect).

## 1. Acceptance (Python oracle, #10743 on spark2's CPU)

`tools/inkling2_dspark_ref.py` (modes `taps`, `draft`, `analyze`, `tiny`) runs the target oracle (`inkling2_ref.py`, W4A16,
fp32 math) over fixed token streams. It saves the four DSpark taps per position (the partial entering visits 5, 23, 41, and the
final-norm output), the target argmax and the routing. It then drafts at every anchor exactly as vLLM's decode step does: context
K/V from the taps of positions < t, block = anchor + 7 masks, f_rnn from position t-1, 7 sequential Markov steps.

Streams: vLLM's greedy outputs on the 12 greedy golden prompts (2564 anchors on the target's greedy path) and the two human prose
texts of the long golden (875 anchors). A draft prefix counts as accepted when every draft equals the target's argmax and the text
stays on the target's path. Off-path anchors (where vLLM's W4A4 near-ties differ from the W4A16 oracle) are reported separately.
Raw output: `docs/results/2026-11-21-inkling2-mtp/analysis.{md,json}`.

P(at least j drafts accepted):

| set | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy golden | 0.722 | 0.521 | 0.384 | 0.293 | 0.230 | 0.189 | 0.152 |
| prose | 0.408 | 0.158 | 0.067 | 0.030 | 0.009 | 0.002 | 0.000 |
| all on-path | 0.642 | 0.428 | 0.303 | 0.226 | 0.174 | 0.141 | 0.113 |

On greedy, the acceptance of position j given j-1 is 0.72, 0.72, 0.74, 0.76, 0.79, 0.82, 0.80. Calibration of the position-1
top-1 probability (greedy, bucket: accept rate): <0.3: 0.17, 0.3-0.5: 0.29, 0.5-0.7: 0.43, 0.7-0.8: 0.53, 0.8-0.9: 0.62,
0.9-0.95: 0.75, >=0.95: 0.95 (1382 of 2564 anchors).

Expert union per visit over R consecutive rows (real routing, 8 of 128 per visit):

| R | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVFP4 visits | 8.0 | 12.5 | 16.0 | 18.9 | 21.4 | 23.7 | 25.7 | 27.6 |
| MXFP8 visits | 8.0 | 12.6 | 16.3 | 19.2 | 21.7 | 23.9 | 25.9 | 27.7 |
| independent routing | 8.0 | 15.5 | 22.5 | 29.1 | 35.3 | 41.1 | 46.5 | 51.6 |

Consecutive tokens share experts strongly: at R = 2 the second row adds 4.5 new experts, not 7.5.

Tokens per verify step (accepted + 1) and modelled speedup. The cost model is the m-moe expert timings at the real union, the
m-attn attention per row, the dense bytes at 480 GB/s plus 1.5 ms fixed (14.4 ms plain step), and the native drafter (block 0.52
GB, base logits 0.92 GB, fold 0.1 GB per step). "Cheap" is a 0.5 GB drafter (perf-strategy §6). These are modelled numbers:
the measured ones are in section 3.

| policy (greedy) | mean k | tokens / step | speedup | cheap drafter |
|---|---:|---:|---:|---:|
| fixed k=1 | 1.00 | 1.722 | 1.23 | 1.38 |
| fixed k=2 | 2.00 | 2.242 | 1.46 | 1.64 |
| fixed k=3 | 3.00 | 2.626 | 1.59 | 1.78 |
| gate p1 >= 0.6, kmax 2 | 1.42 | 2.121 | 1.46 | 1.64 |
| gate p1 >= 0.6, kmax 3 | 1.94 | 2.460 | 1.62 | 1.83 |
| gate p1 >= 0.8, kmax 3 | 1.54 | 2.298 | 1.57 | 1.77 |
| gate p1 >= 0.6, kmax 4 | 2.37 | 2.720 | 1.74 | 1.96 |
| gate p1 >= 0.6, kmax 7 | 3.29 | 3.223 | 1.95 | 2.21 |

On prose every policy is 0.99-1.09x modelled (fixed k=3: 0.99; gate 0.6 kmax 3: 1.07). The gate keeps prose from going below 1.

RESULTS_PLACEHOLDER
