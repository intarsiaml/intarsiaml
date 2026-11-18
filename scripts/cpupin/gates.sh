#!/usr/bin/env bash
# cpupin gates for #436 (the mcdiag hooks), on spark1: (a) `plan show --json` of main's binary (bin/intarsia-main) and the lane's (bin/intarsia), every new knob unset,
# must be byte-identical; (b) greedy ids of `generate --spec mtp --spec-k 2` (default config, knobs unset) of both binaries on the 7 mcopy cells must be equal.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cpupin: gates" -- ssh spark1 'bash ~/intarsiaml-cpupin/src/scripts/cpupin/gates.sh'
# CELLS overrides the cell list. Output in ~/intarsiaml-cpupin/out/gates/.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cpupin; O=$R/out/gates; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
nohup bash $R/bin/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
python3 $R/bin/prompts.py $R/cells > /dev/null
show() { env X=1 $2 plan show --model $M35 --backend cuda --spec mtp --spec-k 2 --mem-budget 40 --max-seq 8192 --json < /dev/null > $O/plan-$1.json 2> $O/plan-$1.err
  echo "plan $1: exit $? $(wc -c < $O/plan-$1.json) bytes sha $(sha256sum < $O/plan-$1.json | cut -c1-16)"; }
show main $R/bin/intarsia-main; show new $R/bin/intarsia; show main2 $R/bin/intarsia-main
cmp $O/plan-main.json $O/plan-main2.json && echo "main twice: byte-identical"
cmp $O/plan-main.json $O/plan-new.json && echo "PASS plan show knobs unset: main and lane byte-identical" || echo "FAIL plan show differs"
fails=0
for n in ${CELLS:-copy3k agent-batch agent-edit json-xform chat-short reason code-gen}; do
  P=$R/cells/$n.txt
  case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  for b in main new; do
    B=$R/bin/intarsia; [ $b = main ] && B=$R/bin/intarsia-main
    $B generate -m $M35 --backend cuda --mem-budget 40 --max-seq $MS --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 --show-ids \
      -p "$(cat $P)" --max-tokens $NT < /dev/null > $O/gen-$n-$b.out 2> $O/gen-$n-$b.err
    grep -a '^ids:' $O/gen-$n-$b.err > $O/gen-$n-$b.ids
  done
  if cmp -s $O/gen-$n-main.ids $O/gen-$n-new.ids && [ -s $O/gen-$n-main.ids ]; then echo "PASS ids $n: $(tr ',' '\n' < $O/gen-$n-new.ids | wc -l) ids identical (main vs lane)"
  else echo "FAIL ids $n: differ or empty"; fails=$((fails+1)); fi
done
echo "gates ids fails=$fails"
