#!/usr/bin/env bash
# kvflip tp2 gate, one fleetq job over both Sparks: the knob unset (A, the base binary: KV-once off) against INTARSIA_CUDA_ATTN_KV_ONCE=1 (B), ABBA, a fresh server pair per window.
#   Every window: ids (greedy + seeded sampled replies of 700 tokens at ~4k / 32.5k / 65.3k prompts, token ids), then sampled T=0.7 and greedy decode tok/s at 4k / 32k / 64k, 3 reps.
#   FLEETQ_OWNER=claude-kvflip fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=45,spark2=45 --expect 90 --label "kvflip tp2 ABBA" -- bash scripts/kvflip/jobAB.sh
D="$(dirname "${BASH_SOURCE[0]}")"
ON="INTARSIA_CUDA_ATTN_KV_ONCE=1"
C="ids 0.7:4096,32768,65536:3 0:4096,32768,65536:3"
bash "$D/tp2.sh" a1 - $C
bash "$D/tp2.sh" b1 "$ON" $C
bash "$D/tp2.sh" b2 "$ON" $C
bash "$D/tp2.sh" a2 - $C
