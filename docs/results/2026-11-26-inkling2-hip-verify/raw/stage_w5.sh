#!/usr/bin/env bash
# Window 5 (lane ink-hip-verify), card 0: main's (80719483a) hip tiny suite `inkling2`, one thread, the same command as the branch's
# run in window 4 (which failed g1b_teacher_forced_bf16 and the two screened-head tests): is that main's result too?
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w5
rm -rf "$O"; mkdir -p "$O"
EXE=$(grep -aoE '/target/release/deps/inkling2-[0-9a-f]+' "$REPO/logs/build-main-tiny.log" | tail -1)
[ -z "$EXE" ] && { echo "[stage] no main tiny exe"; exit 3; }
echo "[stage] main tiny inkling2 $(date +%T) ($EXE)"
HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env INTARSIA_INKLING2_BACKEND=hip $EXE --test-threads 1 > $O/tiny-main.log 2>&1
echo "rc $?"; grep -aE "^test |test result" $O/tiny-main.log | head -24
echo "[stage] done $(date +%T)"
