#!/bin/bash
# G3 ids: for BIN in base cand: greedy 256 ids per prompt for arms off / mtp(k2) / ngram / c512 (--prefill-chunk 512, spec off).
# usage (inside a fleetq slot): ids2.sh OUT prompt...
OUT=$1; shift; mkdir -p $OUT
N=/opt/code/intarsiaml/.worktrees/nxp/.nxp-out
P=/opt/code/intarsiaml/.worktrees/nxp/scripts/mpf/ids-prompts
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for t in cand base; do
  for n in "$@"; do
    case $n in short*) MS=4096;; 4k*) MS=8192;; *) MS=20480;; esac
    for arm in off mtp ngram c512; do
      case $arm in off) A="--spec off";; mtp) A="--spec mtp --spec-k 2";; ngram) A="--spec ngram --spec-k 4";; c512) A="--spec off --prefill-chunk 512";; esac
      $N/intarsia-$t generate -m $M --backend metal --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $P/$n.txt)" --max-tokens 256 < /dev/null > $OUT/$t-$n-$arm.out 2> $OUT/$t-$n-$arm.err
      grep -a '^ids:' $OUT/$t-$n-$arm.err > $OUT/$t-$n-$arm.ids
      echo "$t $n $arm rc=$? ids=$(tr ',' '\n' < $OUT/$t-$n-$arm.ids | wc -l) $(grep -a '^spec:' $OUT/$t-$n-$arm.err | tail -1 | cut -c1-100)"
    done
  done
done
python3 - "$OUT" "$@" <<'PY'
import sys
out=sys.argv[1]; names=sys.argv[2:]
def ids(t,n,a):
    s=open(f'{out}/{t}-{n}-{a}.ids').read().strip()
    return [int(x) for x in s.replace('ids:','').replace('[','').replace(']','').replace(' ','').split(',') if x]
for t in ['base','cand']:
    for n in names:
        o=ids(t,n,'off')
        for a in ['mtp','ngram','c512']:
            x=ids(t,n,a); print('PASS' if x==o and o else 'FAIL',t,n,a,'== off',len(o),'ids')
for n in names:
    b,c=ids('base',n,'off'),ids('cand',n,'off')
    d=next((i for i,(p,q) in enumerate(zip(b,c)) if p!=q),None)
    print('cand vs base (off):',n,'identical' if b==c else f'first divergence at id {d} of {len(b)}; equal ids {sum(p==q for p,q in zip(b,c))}/{len(b)}')
PY
