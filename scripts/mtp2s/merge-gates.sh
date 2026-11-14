#!/usr/bin/env bash
# mtp2s after merging #371: build main (bin/intarsia) and the merged branch (bin/intarsia-topk), then gates.sh plan + ids
# (ref = --spec off, vd, vdt = the full opt-in set), then remove ~/intarsia-mtp2s. One fleetq slot:
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 35 --label "mtp2s: post-merge gates" -- scripts/mtp2s/merge-gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV=origin/main OUTBIN=intarsia bash "$H/build.sh" || exit 1
REV=HEAD OUTBIN=intarsia-topk bash "$H/build.sh" || exit 1
OUT="$(git -C "$H" rev-parse --show-toplevel)/docs/results/2026-11-14-mtp2s/raw/gates-merged" bash "$H/gates.sh"
ssh -n -o BatchMode=yes spark2 'rm -rf ~/intarsia-mtp2s; ls -d ~/intarsia-mtp2s 2>/dev/null || echo "removed ~/intarsia-mtp2s"'
