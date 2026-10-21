#!/usr/bin/env bash
# Lane G2b window 4, Mac side: wait for stages-w4.sh's READY file on tatooine, run localeval against the tp=2 server
# (same settings as the 2026-10-21 standings HIP arms: sizes 128..16384, gen 128, reps 3, thinking off), then DONE.
# Runs inside the window's fleetq job (FLEETQ_JOB inherited), so localeval does not queue for its own slot.
set -uo pipefail
: "${FLEETQ_JOB:?}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$S/../../docs/results/2026-10-21-35b-tp2-hip/raw/w4
mkdir -p "$RAW"
for i in $(seq 1 600); do ssh -o BatchMode=yes tatooine 'test -f ~/g2b/out/w4/READY' 2>/dev/null && break; sleep 5; done
(cd /opt/code/deepeval && timeout 540 uv run localeval run --url http://tatooine:8090/v1 --model qwen36-35b-g2b \
  --sizes 128,1024,4096,16384 --gen 128 --reps 3 --thinking off --label g2b-35b-tp2-hip) > "$RAW/localeval-tp2.out" 2>&1
echo "localeval exit $?" >> "$RAW/localeval-tp2.out"
ssh -o BatchMode=yes tatooine 'touch ~/g2b/out/w4/DONE'
