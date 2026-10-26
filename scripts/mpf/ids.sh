#!/usr/bin/env bash
# Lane mpf ids gate ON spark1 (35B NVFP4, CUDA, tp=1): for each prompt of scripts/mpf/ids-prompts (PROMPTS overrides; scripts/dfh/ids-prompts is the first set) (short-0..5, 4k-0..5, 16k-0..5) greedy
# ids (256, --ignore-eos) per arm; every arm but `off` must equal `off`. Arms (ARMS, default "off mtp"): off = --spec off, mtp = --spec mtp
# --spec-k $K (default 2), auto = no --spec (the recipe default: its stderr must carry the "default of recipe qwen3_5_moe on cuda" line
# and a spec: stats line). BIN: the CLI (default target-dev), O: output dir, SRC: tree that holds the prompts, OFF_FROM: a dir whose
# gen-<name>-off.{ids,err} stand in for the off arm (a mutant binary is compared with the clean run's spec-off ids).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "mpf: ids" -- ssh spark1 'ARMS="off auto" bash -s short-0 4k-0' < scripts/mpf/ids.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mpf
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/target-dev/release/intarsia}; O=${O:-$R/out}; SRC=${SRC:-$R/src}; mkdir -p "$O"
P=${PROMPTS:-$SRC/scripts/mpf/ids-prompts}; fails=0
for n in "$@"; do
  case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; 34k*) MS=36864 ;; *) MS=20480 ;; esac
  for arm in ${ARMS:-off mtp}; do
    if [ "$arm" = off ] && [ -n "${OFF_FROM:-}" ]; then cp $OFF_FROM/gen-$n-off.ids $OFF_FROM/gen-$n-off.err $O/; continue; fi
    case $arm in off) A="--spec off" ;; mtp*) A="--spec mtp --spec-k ${arm#mtp}"; [ "$arm" = mtp ] && A="--spec mtp --spec-k ${K:-2}" ;; auto) A="" ;; esac
    $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $P/$n.txt)" --max-tokens ${NTOK:-256} < /dev/null > $O/gen-$n-$arm.out 2> $O/gen-$n-$arm.err
    grep -a '^ids:' $O/gen-$n-$arm.err > $O/gen-$n-$arm.ids
  done
  pt=$(grep -a -o 'prompt[ _a-z]*[=:] *[0-9]*' $O/gen-$n-off.err | head -1)
  for arm in ${ARMS:-off mtp}; do
    [ "$arm" = off ] && continue
    st=$(grep -a '^spec:' $O/gen-$n-$arm.err | tail -1 | cut -c1-110)
    extra=""
    if [ "$arm" = auto ]; then
      grep -aq 'default of recipe qwen3_5_moe on cuda at tp=1' $O/gen-$n-$arm.err && [ -n "$st" ] || extra=" NO-DEFAULT-LINE-OR-NO-SPEC-STATS"
    fi
    if cmp -s $O/gen-$n-off.ids $O/gen-$n-$arm.ids && [ -s $O/gen-$n-off.ids ] && [ -z "$extra" ]; then echo "PASS ids $n $arm ($pt) $(tr ',' '\n' < $O/gen-$n-off.ids | wc -l) ids identical | $st"; else echo "FAIL ids $n $arm ($pt) differ/empty$extra | $st"; fails=$((fails+1)); fi
  done
done
echo "ids fails=$fails"
