#!/usr/bin/env bash
# postchain2 gates (INTARSIA_SIDE_EAGER), on spark1: (a) `plan show --json` of main's binary (bin/intarsia-main) and the lane's (bin/intarsia), every new knob unset,
# must be byte-identical; (b) greedy ids of `generate --spec mtp --spec-k 2` (default config, knobs unset) of both binaries on the 7 mcopy cells must be equal.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "postchain2: gates" -- ssh spark1 'bash ~/intarsiaml-postchain2/src/scripts/postchain2/gates.sh'
# CELLS overrides the cell list. Output in ~/intarsiaml-postchain2/out/gates/.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-postchain2; O=$R/out/gates; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
nohup bash $R/src/scripts/postchain2/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
show() { env X=1 $2 plan show --model $M35 --backend cuda --spec mtp --spec-k 2 --mem-budget 40 --max-seq 8192 --json < /dev/null > $O/plan-$1.json 2> $O/plan-$1.err
  echo "plan $1: exit $? $(wc -c < $O/plan-$1.json) bytes sha $(sha256sum < $O/plan-$1.json | cut -c1-16)"; }
show main $R/bin/intarsia-main; show new $R/bin/intarsia; show main2 $R/bin/intarsia-main
cmp $O/plan-main.json $O/plan-main2.json && echo "main twice: byte-identical"
cmp $O/plan-main.json $O/plan-new.json && echo "PASS plan show knobs unset: main and lane byte-identical" || echo "FAIL plan show differs"
fails=0
for n in ${CELLS:-copy3k json-xform chat-short reason code-gen}; do
  P=$R/cells/$n.txt
  case $n in copy3k) MS=8192; NT=3800 ;; json-xform) MS=8192; NT=2600 ;; *) MS=4096; NT=500 ;; esac
  for b in main x xc off; do
    B=$R/bin/intarsia; E=X=1; SP="--spec mtp --spec-k 2"
    case $b in main) B=$R/bin/intarsia-main ;; x) E=INTARSIA_SIDE_EAGER=0 ;; xc) E="INTARSIA_SIDE_EAGER=0 INTARSIA_SPEC_COPY=20 INTARSIA_SPEC_TRACE=1" ;; off) SP="--spec off" ;; esac
    env $E $B generate -m $M35 --backend cuda --mem-budget 40 --max-seq $MS $SP --chat --no-think --temperature 0 --profile-every 0 --show-ids \
      -p "$(cat $P)" --max-tokens $NT < /dev/null > $O/gen-$n-$b.out 2> $O/gen-$n-$b.err
    grep -a '^ids:' $O/gen-$n-$b.err > $O/gen-$n-$b.ids
  done
  ok=1; for b in x xc off; do cmp -s $O/gen-$n-main.ids $O/gen-$n-$b.ids || ok=0; done
  if [ $ok = 1 ] && [ -s $O/gen-$n-main.ids ]; then echo "PASS ids $n: $(tr ',' '\n' < $O/gen-$n-main.ids | wc -l) ids identical (main, lane SIDE_EAGER=0, + COPY=20, lane --spec off); copy chains in xc: $(grep -ac '^spec: copy' $O/gen-$n-xc.err)"
  else echo "FAIL ids $n: differ or empty ($(for b in main x xc off; do echo -n "$b $(md5sum < $O/gen-$n-$b.ids | cut -c1-8) "; done))"; fails=$((fails+1)); fi
done
echo "gates ids fails=$fails"
