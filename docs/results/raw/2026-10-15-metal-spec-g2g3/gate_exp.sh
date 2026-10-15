#!/bin/zsh
# usage (inside a slot): gate_exp.sh <bin> <outdir> ; gate parameter sweep, one generate per (config, case)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
BIN=$1; O=$2; mkdir -p $O
P=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g1/prompts
M=$HOME/models/Qwen3-8B
typeset -A FL PF
FL=(open-256 "--chat --no-think" grounded-256 "--chat --no-think" golden-docs-256 "--ignore-eos" golden-code-256 "--ignore-eos" 4k-256 "--ignore-eos" short-256 "--ignore-eos")
PF=(open-256 open grounded-256 grounded golden-docs-256 golden-docs golden-code-256 golden-code 4k-256 4k short-256 short)
CONFIGS=("off" "ngram|0,4,64,1" "ngram|2,4,64,1" "ngram|3,4,64,1" "ngram|2,8,64,1" "ngram|0,4,64,2" "ngram|2,4,64,2" "ngram|1,4,64,1" "off")
i=0
for cfg in $CONFIGS; do
  i=$((i+1))
  mode=${cfg%%|*}; g=${cfg#*|}; [ $mode = off ] && g=0,4,64,1
  tag=$i-${cfg//[|,]/_}
  for c in open-256 grounded-256 golden-docs-256 golden-code-256 4k-256 short-256; do
    INTARSIA_SPEC_GATE=$g INTARSIA_SPEC_TRACE=1 $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --show-ids --max-tokens 256 ${=FL[$c]} --spec $mode --prompt "$(cat $P/${PF[$c]}.txt)" < /dev/null > $O/$tag-$c.out 2> $O/$tag-$c.err
    echo "$tag $c $(grep '^finish=' $O/$tag-$c.err | sed 's/.*decode/decode/') passes=$(grep -c '^spec: token' $O/$tag-$c.err)"
  done
done
