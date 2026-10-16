#!/bin/zsh
# Job F2: position-balanced speed runs after the DVFS finding of job F (the first run after an idle gate ran at a lower GPU clock).
# COLD: fresh `intarsia generate` (1 token, cb-mode trace). 4 arms (base, n1, n1np = n1 with INTARSIA_METAL_PREWIRE=0, n1ns = n1 with
#   INTARSIA_METAL_AHEAD=0), a Latin square over 8 rounds (every arm in every position twice, forward and reversed), a discarded burn-in
#   prefill before each cell, no idle gate between the runs of a cell. One macmon for the whole job; per-run intervals in markers.txt.
# WARM A/B: cbgap-bench inside one process alternates staged uploads on (--ab 2) and off (0) per request (ABBA), so clock drift hits both.
# speed2.sh <outdir> <new tag> <rounds>   (inside an exclusive slot)
set -u
zmodload zsh/datetime
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; TAG=$2; N=$3
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
mkdir -p $O/cold $O/warm
macmon pipe -i 200 > $O/macmon.jsonl 2>/dev/null & MP=$!
trap 'kill $MP 2>/dev/null' EXIT
: > $O/markers.txt
BIN=(); ENVV=()
arm_cfg() {
  case $1 in
    base) BIN=$SC/intarsia-base; ENVV=(A=1);;
    n1)   BIN=$SC/intarsia-$TAG; ENVV=(A=1);;
    n1np) BIN=$SC/intarsia-$TAG; ENVV=(INTARSIA_METAL_PREWIRE=0);;
    n1ns) BIN=$SC/intarsia-$TAG; ENVV=(INTARSIA_METAL_AHEAD=0);;
  esac
}
one() { # arm round size chunk
  local A=$1 r=$2 S=$3 C=$4; arm_cfg $A
  local F=$O/cold/$A-r$r-$S-c$C t0 t1
  t0=$EPOCHREALTIME
  env $ENVV INTARSIA_METAL_PROFILE=cb:$F.trace.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
    --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
  t1=$EPOCHREALTIME
  echo "$A r$r $S c$C $t0 $t1" >> $O/markers.txt
}
ORDERS=("base n1 n1np n1ns" "n1 n1np n1ns base" "n1np n1ns base n1" "n1ns base n1 n1np" "n1ns n1np n1 base" "base n1ns n1np n1" "n1 base n1ns n1np" "n1np n1 base n1ns")
for r in $(seq 1 $N); do
  ARMS=(${=ORDERS[$(( (r - 1) % 8 + 1 ))]})
  for C in 256 2048; do for S in 1024 4062; do
    wait_idle || echo "cell r$r $S c$C not idle" >> $O/markers.txt
    arm_cfg n1; env $ENVV $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 --profile-every 0 --prefill-chunk $C \
      --prompt "$(cat $R/prompt-4062.txt)" > /dev/null 2>&1   # burn-in: brings the GPU clock up, discarded
    for A in $ARMS; do one $A $r $S $C; done
  done; done
  echo "cold round $r done $(date +%T)"
done
for C in 256 2048; do for p in 1 2 3; do
  wait_idle || touch $O/warm/c$C-p$p.contaminated
  P=$O/warm/c$C-p$p
  INTARSIA_METAL_PROFILE=cb:$P.trace.jsonl $SC/cbgap-bench-$TAG --model $M --backend metal --max-seq 8192 --profile-every 0 --prefill-chunk $C \
    --docs /opt/code/intarsiaml-cbgap/tools/prompts/corpus_docs.txt --lens 1024,4062 --reps 12 --warmup 1 --ab 2 --out $P.req.jsonl > $P.log 2>&1
  echo "warm c$C p$p done $(date +%T)"
done; done
echo SPEED2_DONE
