#!/usr/bin/env bash
# Lane lmhead-fp8-default: remove what cbuild.sh / the CUDA gates left on spark1 (~/lmh-cuda: trees, targets, the copied 27B, outputs).
#   fleetq run --host spark1 --load none --mem 0 --expect 5 --label "..." -- scripts/lmhead/cclean.sh
set -uo pipefail
ssh -n -o BatchMode=yes spark1 'rm -rf ~/lmh-cuda; ls -d ~/lmh-cuda 2>&1 | head -1; df -h ~ | tail -1'
