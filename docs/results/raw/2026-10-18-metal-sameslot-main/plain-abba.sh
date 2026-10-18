#!/bin/zsh
# Plain-prefill ABBA, 4k prompt: CAND (2026-10-15 intarsia-cand) vs main 5eabb1a7, one exclusive fleetq slot.
# Per run: generate --max-tokens 1 --profile-every 0 at the shipped chunk (nothing passed), same flags as
# docs/results/raw/2026-10-15-metal-final/gates.sh prun(). One discarded warm-up per binary, then 8 rounds with the
# order flipped each round (C M, M C, C M, ...), 20 s between runs. GPU idle (3 samples) and a 100 ms macmon trace per run.
B=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
CAND=$B/final/intarsia-cand
MAIN=$B/intarsia-main
OUT=/opt/code/intarsiaml/docs/results/raw/2026-10-18-metal-sameslot-main/plain
PF=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-c/prompt-4k.txt
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 60 --label "intarsiaml plain-prefill ABBA 4k: 10-15 CAND vs main 5eabb1a7" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
{ echo "date: $(date -Iseconds)"; echo "cand md5: $(md5 -q $CAND)"; echo "main md5: $(md5 -q $MAIN)"; echo "prompt md5: $(md5 -q $PF)"; sw_vers | tr '\n' ' '; echo; } > $OUT/versions.txt
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1; sleep 1; done | tr '\n' ' '; echo; }
run() { # label bin
  local f=$OUT/$1
  echo "idle: $(idle3)  zoom: $(pgrep -fl zoom.us | tr '\n' ';')" > $f.idle
  macmon pipe -i 100 > $f.macmon.jsonl 2>/dev/null & local MP=$!
  date +%s.%N > $f.t0
  $2 generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $PF)" > $f.out 2> $f.err
  date +%s.%N > $f.t1
  kill $MP 2>/dev/null; wait $MP 2>/dev/null
  echo "$1: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $f.err | tail -1) $(grep -o 'prefill chunk [0-9]*' $f.err | head -1) | $(cat $f.idle)"
}
run warm-cand $CAND; sleep 20
run warm-main $MAIN; sleep 20
i=0
for b in C M M C C M M C C M M C C M M C; do
  i=$((i+1))
  if [[ $b == C ]]; then run cand-$(printf %02d $i) $CAND; else run main-$(printf %02d $i) $MAIN; fi
  sleep 20
done
echo "idle end: $(idle3)"
echo PLAIN_DONE
