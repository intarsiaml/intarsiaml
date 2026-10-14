#!/usr/bin/env bash
# Window 2 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): the dev build's nsys decode profile, then the 35B A/B
# of the base (arch/plan tip) and dev builds in 3 interleaved rounds (base dev / dev base / base dev), as the prefill-cuda
# doc did. Steps are skipped when the deadline is near (W2_BUDGET_MIN, default 14).
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
DEADLINE=$(( $(date +%s) + ${W2_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
need 3 "nsys dev" && bash $D/w-profile.sh dev:dev
need 3 "A/B round 1" && bash $D/w-ab.sh 35b 1 base:base dev:dev
need 3 "A/B round 2" && bash $D/w-ab.sh 35b 2 dev:dev base:base
need 3 "A/B round 3" && bash $D/w-ab.sh 35b 3 base:base dev:dev
echo "== w2 done $(date +%T), $(left) min left"
