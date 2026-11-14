#!/usr/bin/env bash
# mtp2s A/B round 2 (one fleetq slot): does the reduced-draft-head + batched-verify win hold for production-like traffic?
#   t1:    temperature 1.0, seeded, ABBA O A V VD VD V A O (O = drafting off: the seeded-sampled ids check against it)
#   g0:    greedy O, one visit (drafting off; compare its reply shas with round 1's A / VD greedy visits)
#   long:  greedy, ~3k-token prompts, A VD
#   cjk:   greedy, Chinese prompts, A VD (acceptance with the 65536-row draft head)
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 50 --label "mtp2s: ..." -- scripts/mtp2s/ab2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(git -C "$H" rev-parse --show-toplevel)/docs/results/2026-11-14-mtp2s/raw"
for part in ${PARTS:-t1 g0 long cjk}; do
  case $part in
    t1)   OUT=$R/ab2-t1   TEMP=1.0 SET=en   ARMS="O1 A1 V1 VD1 VD2 V2 A2 O2" LOGNAME_=ab2-t1 bash "$H/ab.sh" ;;
    g0)   OUT=$R/ab2-g0   TEMP=0   SET=en   ARMS="O1"                       LOGNAME_=ab2-g0 bash "$H/ab.sh" ;;
    long) OUT=$R/ab2-long TEMP=0   SET=long ARMS="A1 VD1"                   LOGNAME_=ab2-long bash "$H/ab.sh" ;;
    cjk)  OUT=$R/ab2-cjk  TEMP=0   SET=cjk  ARMS="A1 VD1"                   LOGNAME_=ab2-cjk bash "$H/ab.sh" ;;
  esac
done
