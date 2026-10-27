#!/bin/bash
# G3 ids: base vs cand greedy 256 ids; arms off / mtp(k2) / ngram / c512. usage (in a fleetq slot): ids.sh OUT prompt...
OUT=$1; shift; mkdir -p $OUT
R=/opt/code/intarsiaml/.worktrees/mgd/run; P=/opt/code/intarsiaml/.worktrees/mgd/scripts/mpf/ids-prompts
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for t in cand base; do for n in "$@"; do
  case $n in short*) MS=4096;; 4k*) MS=8192;; *) MS=40960;; esac
  for arm in off mtp ngram c512; do
    case $arm in off) A="--spec off";; mtp) A="--spec mtp --spec-k 2";; ngram) A="--spec ngram --spec-k 4";; c512) A="--spec off --prefill-chunk 512";; esac
    $R/intarsia-$t generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $P/$n.txt)" --max-tokens 256 < /dev/null > $OUT/$t-$n-$arm.out 2> $OUT/$t-$n-$arm.err
    grep -a '^ids:' $OUT/$t-$n-$arm.err > $OUT/$t-$n-$arm.ids
    echo "$t $n $arm rc=$? n=$(tr ',' '\n' < $OUT/$t-$n-$arm.ids | wc -l)"
  done
done; done
python3 - "$OUT" "$@" <<'PY'
import sys
out=sys.argv[1]; names=sys.argv[2:]
def ids(t,n,a): return open(f'{out}/{t}-{n}-{a}.ids').read().strip()
for n in names:
    ref=ids('cand',n,'off')
    for a in ['mtp','ngram','c512']: print('PASS' if ids('cand',n,a)==ref else 'FAIL','cand',n,a,'== cand off')
    for a in ['off','mtp','ngram','c512']: print('PASS' if ids('cand',n,a)==ids('base',n,a) else 'FAIL','cand',n,a,'== base',a)
PY
