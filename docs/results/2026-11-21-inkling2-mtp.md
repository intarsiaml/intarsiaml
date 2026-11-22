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

RESULTS_PLACEHOLDER
