#!/bin/zsh
# copyflipm narrowing gate (the default only for a recipe that opts in), inside a fleetq slot on endor (--quiet gpu,cpu), no flags (the recipe's MTP default):
#  - Qwen3.8-27B: unset on the new binary == unset on base (origin/main), greedy ids on copy3k and code-gen, and no copy drafts in the log
#  - Qwen3.6-35B-A3B: unset on the new binary == `--spec off`, copy3k, and the log says copy drafts are on by default (chain of 15)
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-copyflipm/raw/ids-narrow; mkdir -p $RAW
BASE=$W/target-cf-base/intarsia-base; NEW=$W/target-cf/release/intarsia; CELLS=$W/target-cf/cells
mkdir -p $CELLS; [ -f $CELLS/copy3k.txt ] || python3 $W/scripts/mcopy/prompts.py $CELLS > /dev/null
fails=0
gen() { # name model bin cell max_seq max_tokens extra-args...
  local n=$1 m=$2 b=$3 c=$4 ms=$5 nt=$6; shift 6; local f=$RAW/$n-$c
  INTARSIA_SPEC_TRACE=1 $b generate -m $m --backend metal --max-seq $ms "$@" --chat --no-think --temperature 0 --profile-every 0 --show-ids -p "$(cat $CELLS/$c.txt)" --max-tokens $nt < /dev/null > $f.out 2> $f.err
  echo "$n $c: exit $? $(grep -a 'tok/s' $f.err | tail -1 | sed -E 's/\| prefill.*\| decode [0-9]+ ms //' | cut -c1-120)"; grep -a '^ids:' $f.err > $f.ids
}
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for c in copy3k code-gen; do
  case $c in copy3k) MS=8192; NT=1000 ;; *) MS=4096; NT=300 ;; esac
  gen base27 $M27 $BASE $c $MS $NT; gen new27 $M27 $NEW $c $MS $NT
  if [ -s $RAW/base27-$c.ids ] && cmp -s $RAW/base27-$c.ids $RAW/new27-$c.ids; then echo "PASS 27B $c: $(tr ',' '\n' < $RAW/new27-$c.ids | wc -l | tr -d ' ') ids identical, new unset == base unset"; else echo "FAIL 27B $c ids"; fails=$((fails+1)); fi
  grep -aq "copy drafts on" $RAW/new27-$c.err && { echo "FAIL 27B $c: copy drafts on"; fails=$((fails+1)); } || echo "PASS 27B $c: copy drafts off in the log"
done
gen off35 $M35 $NEW copy3k 8192 3800 --spec off; gen new35 $M35 $NEW copy3k 8192 3800 --spec mtp --spec-k 2
if [ -s $RAW/off35-copy3k.ids ] && cmp -s $RAW/off35-copy3k.ids $RAW/new35-copy3k.ids; then echo "PASS 35B copy3k: $(tr ',' '\n' < $RAW/new35-copy3k.ids | wc -l | tr -d ' ') ids identical, new unset == --spec off"; else echo "FAIL 35B copy3k ids"; fails=$((fails+1)); fi
grep -aq "copy drafts on: a chain of up to 15 tokens.*default on Metal" $RAW/new35-copy3k.err && echo "PASS 35B copy3k: copy drafts on by default, chain 15" || { echo "FAIL 35B copy3k: not on by default"; fails=$((fails+1)); }
# the 35B with no --spec flag (recipe MTP default) is on too
gen newns35 $M35 $NEW code-gen 4096 200
grep -aq "copy drafts on: a chain of up to 15 tokens.*default on Metal" $RAW/newns35-code-gen.err && echo "PASS 35B no --spec: copy drafts on by default" || { echo "FAIL 35B no --spec: not on"; fails=$((fails+1)); }
echo "narrow gate fails=$fails"
