#!/bin/zsh
# Engine half of the DSpine tap-replay test (Metal): one `intarsia generate` per prompt with INTARSIA_DSPINE_DUMP (+ _LAYERS), greedy,
# DSpine k7, forced length, one dump directory per prompt under <out>. Inside a fleetq slot:
#   fleetq run --host endor --quiet gpu,cpu --mem 30 --expect 40 --label "dspine tap dump" -- zsh tools/dspine/dump_cats.sh <binary> <out> [category files...]
# Prompts: every line of the category files (default scripts/dfh/cats/*.jsonl, the files scripts/dt1/cat-window.sh runs) as <file>-<name>, and, with
# LONG=1, scripts/hip-decode/prompt4k.txt as `long4k` (TOKENS_LONG, default 256). TOKENS (default 512) are forced (--ignore-eos --min-tokens).
# Then `python -I tools/dspine/replay_taps.py ... <out>/*/` (the reference half).
set -u
BIN=${1:?binary}; O=${2:?out dir}; shift 2; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}
DRAFT=${DRAFT:?DRAFT=<DSpine checkpoint dir>}
TOKENS=${TOKENS:-512}; K=${K:-7}
FILES=(${@:-$R/scripts/dfh/cats/*.jsonl})
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
run() { # run <name> <prompt> <tokens> <max-seq>
  local n=$1 p=$2 tok=$3 ms=$4 d=$O/$1
  [[ -f $d/dump/ids-0.json ]] && { say "$n: done already"; return; }
  rm -rf $d; mkdir -p $d
  INTARSIA_DSPINE_DUMP=$d/dump INTARSIA_DSPINE_DUMP_LAYERS=${LAYERS:-1} INTARSIA_SPEC_TRACE=0 $BIN generate -m $M --backend metal --max-seq $ms --chat --no-think \
    --profile-every 0 --ignore-eos --min-tokens $tok --show-ids -p "$p" --max-tokens $tok --spec dflash --spec-k $K --spec-draft $DRAFT \
    > $d/gen.out 2> $d/gen.err < /dev/null
  say "$n exit $? | $(grep -a '^prompt:' $d/gen.err) | $(grep -a '^spec:' $d/gen.err) | $(grep -a 'decode' $d/gen.err | tail -1 | grep -o 'decode.*')"
}
for f in $FILES; do
  b=${f:t:r}
  python3 - $f $b > $O/.prompts-$b.tsv <<'PY'
import json, sys
for l in open(sys.argv[1]):
    if l.strip():
        j = json.loads(l)
        print(sys.argv[2] + "-" + j["name"] + "\t" + j["input"].replace("\t", " ").replace("\n", " "))
PY
  while IFS=$'\t' read -r n p; do run $n "$p" $TOKENS 4096; done < $O/.prompts-$b.tsv
done
if [[ ${LONG:-0} == 1 ]]; then run long4k "$(cat $R/scripts/hip-decode/prompt4k.txt)" ${TOKENS_LONG:-256} 8192; fi
say "all done"
