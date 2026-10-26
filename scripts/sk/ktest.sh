#!/usr/bin/env bash
# Lane sk: `intarsia kernels test --backend cuda --quick` of a tree on spark2:  fleetq run --host spark2 --mem 20 --expect 15 --label "sk: kernels test" -- scripts/sk/ktest.sh <tree> <outname> [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=$1; OUTN=$2; shift 2; HOST=${HOST:-spark2}
RAW=/opt/code/intarsiaml/.worktrees/sk2/docs/results/2026-10-25-sk2-serve-macro/raw; mkdir -p "$RAW"
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; env ${*:-X=1} timeout 1500 \$HOME/intarsiaml-sk-$TREE/target/acc/release/intarsia kernels test --backend cuda --quick < /dev/null" > "$RAW/$OUTN.log" 2>&1
echo "exit $?"; grep -aE 'FAIL|REJECT|kernels,|cases|QUICK DONE' "$RAW/$OUTN.log" | tail -4
