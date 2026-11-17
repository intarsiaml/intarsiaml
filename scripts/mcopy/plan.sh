#!/usr/bin/env bash
# mcopy2 gate (b): `plan show --json` of main's binary and the new one, INTARSIA_SPEC_COPY unset, must be byte-identical; with the knob set
# the new binary's plan differs only by the per-row recurrent buffers. Writes $R/out/plan/*.json and prints the cmp / diff result.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 15 --label "mcopy2: plan show" -- ssh spark1 'bash -s' < scripts/mcopy/plan.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mcopy2; O=$R/out/plan; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
show() { # <name> <binary> [ENV=val]
  env ${3:-X=1} $2 plan show --model $M35 --backend cuda --spec mtp --spec-k 2 --mem-budget 40 --max-seq 8192 --json < /dev/null > $O/$1.json 2> $O/$1.err
  echo "$1: exit $? $(wc -c < $O/$1.json) bytes sha $(sha256sum < $O/$1.json | cut -c1-16)"
}
show main $R/bin/intarsia-main
show new $R/bin/intarsia
show new-n20 $R/bin/intarsia INTARSIA_SPEC_COPY=20
show new-n20L15 $R/bin/intarsia INTARSIA_SPEC_COPY=20,15
show new-n20L3 $R/bin/intarsia INTARSIA_SPEC_COPY=20,3
show main-n20 $R/bin/intarsia-main INTARSIA_SPEC_COPY=20
show main2 $R/bin/intarsia-main
cmp $O/main.json $O/main2.json && echo "main twice: byte-identical (plan show is deterministic)" || diff <(python3 -m json.tool $O/main.json) <(python3 -m json.tool $O/main2.json) | head
cmp $O/main.json $O/new.json && echo "PASS plan show knob unset: main and new byte-identical" || { echo "FAIL plan show differs:"; diff <(python3 -m json.tool $O/main.json) <(python3 -m json.tool $O/new.json) | head -20; }
diff <(python3 -m json.tool $O/main.json) <(python3 -m json.tool $O/new-n20.json) | head -20; echo "--- main vs new n20L3"; diff <(python3 -m json.tool $O/main.json) <(python3 -m json.tool $O/new-n20L3.json) | head -20; echo "--- main vs new n20L15"; diff <(python3 -m json.tool $O/main.json) <(python3 -m json.tool $O/new-n20L15.json) | head -20
echo "--- main n20 vs new n20 (rows only widened by verify_rows before)"; diff <(python3 -m json.tool $O/main-n20.json) <(python3 -m json.tool $O/new-n20.json) | head -20
