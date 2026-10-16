#!/bin/zsh
# Job F: speed, interleaved. Arms: base (70b15eb), n1 (this branch), n1np (n1 with INTARSIA_METAL_PREWIRE=0), n1ns (n1 with INTARSIA_METAL_AHEAD=0).
# COLD: a fresh `intarsia generate` per run (1 token, cb-mode trace, macmon). WARM: one cbgap-bench process per run (1 untimed request per
# length, then 3 timed ones per length, serving path), cb-mode trace, macmon. Prompt sizes 1024 and 4062 tokens, chunk 256 and 2048.
# Per round every (size, chunk) cell runs each arm once; the arm order alternates forward / reversed between rounds.
# speed.sh <outdir> <rounds>   (inside an exclusive slot)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; N=$2
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
mkdir -p $O/cold $O/warm
arm_cfg() { # arm -> sets BIN BENCH ENVV
  case $1 in
    base) BIN=$SC/intarsia-base; BENCH=$SC/cbgap-bench-base; ENVV=(A=1);;
    n1)   BIN=$SC/intarsia-n1;   BENCH=$SC/cbgap-bench-n1;   ENVV=(A=1);;
    n1np) BIN=$SC/intarsia-n1;   BENCH=$SC/cbgap-bench-n1;   ENVV=(INTARSIA_METAL_PREWIRE=0);;
    n1ns) BIN=$SC/intarsia-n1;   BENCH=$SC/cbgap-bench-n1;   ENVV=(INTARSIA_METAL_AHEAD=0);;
  esac
}
for r in $(seq 1 $N); do
  if [ $((r % 2)) = 1 ]; then ARMS=(base n1 n1np n1ns); else ARMS=(n1ns n1np n1 base); fi
  for C in 256 2048; do
    for S in 1024 4062; do for A in $ARMS; do
      arm_cfg $A
      F=$O/cold/$A-r$r-$S-c$C
      wait_idle || touch $F.contaminated
      macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null & MP=$!
      env $ENVV INTARSIA_METAL_PROFILE=cb:$F.trace.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
        --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-$S.txt)" > $F.out 2> $F.err
      kill $MP 2>/dev/null
    done; done
    for A in $ARMS; do
      arm_cfg $A
      $R/trace-run.sh $BENCH $O/warm/$A-r$r-c$C $C 1024,4062 3 1 $ENVV
    done
  done
  echo "round $r done $(date +%T)"
done
echo SPEED_DONE
