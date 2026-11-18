#!/usr/bin/env bash
# sideeager gate, on spark1: greedy ids of `generate --spec mtp --spec-k 2` with the knobs unset (base) and with INTARSIA_SIDE_EAGER=0 (x), same binary, on the cells that have a
# single-message prompt (the agent cells are multi-message: their served replies are compared by sha in the ABBA, see ab_an.py).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 15 --label "sideeager: gates" -- ssh spark1 'bash ~/intarsiaml-sideeager/src/scripts/sideeager/gates.sh'
# CELLS overrides the cell list. Output in ~/intarsiaml-sideeager/out/gates/.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-sideeager; O=$R/out/gates; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
nohup bash $R/src/scripts/sideeager/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
fails=0
for n in ${CELLS:-chat-short reason code-gen}; do
  P=$R/cells/$n.txt
  for b in base x; do
    E=X=1; [ $b = x ] && E=INTARSIA_SIDE_EAGER=0
    env $E $R/bin/intarsia generate -m $M35 --backend cuda --mem-budget 40 --max-seq 4096 --spec mtp --spec-k 2 --chat --no-think --temperature 0 --profile-every 0 --show-ids \
      -p "$(cat $P)" --max-tokens 500 < /dev/null > $O/gen-$n-$b.out 2> $O/gen-$n-$b.err
    grep -a '^ids:' $O/gen-$n-$b.err > $O/gen-$n-$b.ids
  done
  if cmp -s $O/gen-$n-base.ids $O/gen-$n-x.ids && [ -s $O/gen-$n-base.ids ]; then echo "PASS ids $n: $(tr ',' '\n' < $O/gen-$n-base.ids | wc -l) ids identical (base, INTARSIA_SIDE_EAGER=0)"
  else echo "FAIL ids $n: differ or empty ($(for b in base x; do echo -n "$b $(md5sum < $O/gen-$n-$b.ids | cut -c1-8) "; done))"; fails=$((fails+1)); fi
done
echo "gates ids fails=$fails"
