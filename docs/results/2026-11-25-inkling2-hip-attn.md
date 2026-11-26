# Inkling-2-Mini attention on HIP (lane ink-hip-attn, package A, 2026-11-25)

Branch `ink/hip-attn` (origin/main `d8c3c7148` + `ink/hip` skeleton `6c5ee5d67`). Ops `InkKvPrologue` (70), `RelSinkAttn` (71);
stretch `InkAttnDecode` (75). Brief: `.worktrees/hip-swarm/COMMON.md`; port map: `docs/results/2026-11-25-inkling2-hip.md` (ink/hip).

## Briefs received (kept against context loss)

- 2026-11-25 ~21:45 EST, coordinator: the lead's skeleton is committed, `6c5ee5d67` on `ink/hip`; run `git merge ink/hip` now
  (done). Module shape and notes are in the last section of COMMON.md. Op 75 is optional for G2, so land 70 and 71 first; use the
  scalar split-KV `ink_attn_s1_k` + comb form first. Write this message into the results doc (this entry).

## Step 0

`git log --all -i --grep=hip` and a grep of every `ink/*` branch for `ink_kv_prologue` / `rel_sink_attn` under `kernels/hip` and
`crates/intarsia-backend-hip`: no earlier HIP Inkling attention work exists.

## Status

(open)
