#!/bin/bash
# endor EXCLUSIVE fleetq slot 1, lane ink-metal-dflash2: (a) the Inkling-2 tiny suites on the Metal backend (INTARSIA_INKLING2_BACKEND=metal; the
# DFlash2 file is this lane's), (b) greedy md5: golden `code` + the 4108-id prompt, --ignore-eos, 256 tokens (64 = the first 64 ids), arms
# off / dspark (the Metal default, k3) / dfl3 / dfl4 / dfl5 / dfl7 (the DFlash2 drafter at --spec-k K).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
trap 'for c in $(pgrep -P $$ 2>/dev/null); do pkill -P $c 2>/dev/null; kill $c 2>/dev/null; done' EXIT
W=/opt/code/intarsiaml/.worktrees/ink-metal-dflash2; R=$W/docs/results/2026-11-26-inkling2-metal-dflash2/raw
cd $W
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
echo "slot1 start $(date +%T)"
if [ -z "${SKIP_TINY:-}" ]; then
  mkdir -p .scratch/metal/s1
  echo "== tiny suites on metal $(date +%T)"
  for t in inkling2_dflash2 inkling2_dspark inkling2_verify_softcap inkling2; do
    env INTARSIA_INKLING2_BACKEND=metal cargo test --release --no-fail-fast -p intarsia-models --test $t > .scratch/metal/s1/tiny-$t.txt 2>&1
    echo "  $t rc $? $(date +%T) | $(grep -aE '^test result' .scratch/metal/s1/tiny-$t.txt | tr '\n' ' ')"
    grep -aE "FAILED|panicked" .scratch/metal/s1/tiny-$t.txt | head -5
  done
fi
echo "== md5 $(date +%T)"
NAME=s1-md5 PROMPTS=md5 TOK=256 IDLE=0 LIMIT=900 ARMS="off@r1 dspark@r1 dfl4@r1 dfl7@r1 dfl3@r1 dfl5@r1" $R/slot.sh
echo "slot1 end $(date +%T)"
