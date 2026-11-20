#!/usr/bin/env bash
# kvonce served gate: the knob off (A) against on (B), order ABBA BAAB, a fresh server per window, shipping command line (window.sh).
#   a1 / b1 also: seeded sampled replies (T=0.7, seed 11, 700 tokens) and greedy replies (T=0, 700 tokens) for byte-for-byte identity,
#   prompts of ~4000 / 32500 / 65300 tokens so the decode crosses a chunk change (4096, 32768, 65536 keys).
#   Every window: sampled T=0.7 and greedy, 4k / 32k / 64k, 3 reps -> 12 reps per arm per cell. Then one `--spec off` window (sampled + greedy 32k / 64k).
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 90 --label "kvonce ABBA" -- bash scripts/kvonce/jobAB.sh
D="$(dirname "${BASH_SOURCE[0]}")"
ON="INTARSIA_CUDA_ATTN_KV_ONCE=1"
ID="seeded:4000,32500,65300 greedyid:4000,32500,65300"
C="0.7:4096,32768,65536:3 0:4096,32768,65536:3"
bash "$D/window.sh" a1 "" - $ID $C
bash "$D/window.sh" b1 "" "$ON" $ID $C
bash "$D/window.sh" b2 "" "$ON" $C
bash "$D/window.sh" a2 "" - $C
bash "$D/window.sh" b3 "" "$ON" $C
bash "$D/window.sh" a3 "" - $C
bash "$D/window.sh" a4 "" - $C
bash "$D/window.sh" b4 "" "$ON" $C
bash "$D/window.sh" o1 "--spec off" - 0.7:32768,65536:4 0:32768,65536:4
