#!/usr/bin/env bash
# Window 2 (spark1, vLLM down, inside scripts/cuda-decode/window.sh): the dev build's nsys decode profile, an early-abort
# guard (launch count must have dropped; generated text must equal the base build's), then the 35B A/B of the base
# (arch/plan tip) and dev builds, interleaved rounds (base dev / dev base). A step is skipped when the deadline is near
# (W2_BUDGET_MIN, default 16; a round needs 7 min).
set -u
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
DEADLINE=$(( $(date +%s) + ${W2_BUDGET_MIN:-16} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
need() { if (( $(left) < $1 )); then echo "SKIP $2: $(left) min left, needs $1"; return 1; fi; echo "== $2 ($(left) min left) $(date +%T)"; return 0; }
need 3 "nsys dev" && bash $D/w-profile.sh dev:dev
L=$(python3 -I -c "import json;print(int(json.load(open('$P/out/prof-dev/steps.json'))['launches']))" 2>/dev/null || echo 0)
echo "dev launches/step: $L (base 944)"
if cmp -s $P/out/prof-base/nsys.out $P/out/prof-dev/nsys.out; then echo "TEXT: dev generated text == base generated text (160 greedy tokens)"; else echo "TEXT: dev generated text DIFFERS from base"; fi
if (( L == 0 || L > 900 )); then echo "ABORT: fused ops not in effect (launches $L)"; exit 1; fi
need 7 "A/B round 1" && bash $D/w-ab.sh 35b 1 base:base dev:dev
need 7 "A/B round 2" && bash $D/w-ab.sh 35b 2 dev:dev base:base
echo "== w2 done $(date +%T), $(left) min left"
