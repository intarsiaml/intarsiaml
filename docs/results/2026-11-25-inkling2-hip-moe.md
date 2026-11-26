# Inkling-2-Mini experts on HIP (lane ink-hip-moe, package C, 2026-11-25)

Branch `ink/hip-moe`, base origin/main `d8c3c7148` + `ink/hip` `6c5ee5d67` (lead's skeleton). Ops: `MoeExperts` situ (NVFP4),
`MoeExpertsMxfp8` (72). W4A16 only.

## 0. Briefs received (kept here against context loss)

- Launch brief (coordinator, ~21:50 EST): package C; `moe_experts_situ_bf16` + `_rows`, `moe_experts_mxfp8_bf16` + `_rows`,
  registered `.situ()`, never `.act_quant()`; 128 experts, top-8, D 2304, I 1152. Order: decode path correct and fast (bench vs
  627 GB/s on the active experts' bytes), then a correct prefill path, speed second; WMMA prefill is speed package S2.
  Files: `kernels/hip/ink_moe.hip`, `kernels/hip/ik_hip_abi_inkmoe.h`, `crates/intarsia-backend-hip/src/inkmoe.rs`,
  `crates/intarsia-backend-hip/tests/ink_moe*.rs`, this doc, `HANDOFF.md`.
- Coordinator message after the skeleton landed: skeleton is `6c5ee5d67` on `ink/hip`, merge it; module shape and notes are in
  the last section of `.worktrees/hip-swarm/COMMON.md`. The router is the lead's, in its own module (`src/inkroute.rs`,
  `ink_route.hip`), so `inkmoe.rs` is this lane's alone. Write a SIMT `ink_moe_gu` / `dn` correctness form first, at any pair
  count, with `SITU_TOL (1.5e-2, 2e-3)`; the fast decode form after that.

- Coordinator message (from the ink-hip lead, after moe_route's first window): tatooine is ready for integration; when the expert
  ops pass their unit window, send the coordinator "ready to merge at <commit>". Checkpoint on tatooine (read-only):
  `/data/models/intarsia/Inkling-2-Mini` (in the dev image `/models/Inkling-2-Mini`). Window scripts to copy, not edit:
  `docs/results/2026-11-25-inkling2-hip/raw/window.sh` (stops radiance, capped stage in its own process group, restore trap, polls
  /v1/models on :8080) and `run.sh` (dev image, card 0), both on `ink/hip`. Own `INTARSIA_TARGET_VOL=intarsia-cargo-target-<lane>`
  and own container name. `ink/hip` head has the scripts plus the hip arm of the tiny tests on top of `6c5ee5d67`; no re-merge
  needed. The lead merges each lane once its ops pass their own unit window, then runs admission + tiny G1 + real-model G2 in one
  window once ops 67-74 and the experts exist; the experts block that window, so correct first, fast after. moe_route passed its
  window (#11924, ADMITTED 19/19). Window rules unchanged: fleetq `--takes-down radiance`, 20 min or less, at most 3 windows,
  none after 06:00, radiance back up before release.

## 1. Status

(open)
