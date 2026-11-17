#!/bin/zsh
# Lane mfp8split gates (Metal), adapted from scripts/mchunk4/gates.sh, for the 27B and the 35B. Run under fleetq (--quiet gpu,cpu).
# Binaries (copied aside by build.sh): NEW = .base/bin-new/intarsia (this branch, split on unless INTARSIA_METAL_FP8_ROWS_SPLIT=0), BASE = .base/bin-main/intarsia (origin/main)
#   solo     : `generate --spec off --show-ids` (256 tokens, greedy) on each of the 16 prompts of served_ids.py, NEW binary
#   served   : serve arms new-mb16, new-mb16-off (INTARSIA_METAL_FP8_ROWS_SPLIT=0), base-mb16, new-mb8, base-mb8; 1 / 4 / 6 / 8 / 9 / 10 / 12 / 16 concurrent requests
#              (even prompts ~99 tokens, odd ~4666), ids per stream, plan JSON of each server
#   compare  : every served stream vs its solo ids; arms vs each other; plan diffs
#   route    : kernel-mode profile of the batched decode (short prompts) at 10 / 12 streams, split on and off: matmul_fp8_gemv_rows launches by rows
#   routedef : the same 10-stream route check with the env var UNSET (the shipped default)
#   mtp      : `generate` with the default engine (MTP), NEW vs BASE ids, short + 4k
# usage: gates.sh <27b|35b> solo | served [arm...] | compare | route | mtp
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
MODEL=${1:?27b|35b}; STEP=${2:?step}; shift 2
W=/opt/code/intarsiaml/.worktrees/mfp8split; R=$W/docs/results/2026-11-17-mfp8split/raw/gates-$MODEL; mkdir -p $R/prompts $R/solo
BASE=$W/.base/bin-main/intarsia; NEW=$W/.base/bin-new/intarsia
case $MODEL in 27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4 ;; 35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4 ;; esac
TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
MAXSEQ=${MAXSEQ:-8192}
py() { python3 $W/scripts/mfp8split/served_ids.py "$@"; }
PORT=18677
start() { local O=$1 B=$2; shift 2; mkdir -p $O
  $B serve --backend metal --model $M --served-model-name intarsia --max-seq $MAXSEQ --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $O/server.out 2> $O/server.err < /dev/null &
  TP=$!
  for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $O: $(tail -3 $O/server.err)"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
case $STEP in
  solo)
    py prompts --text $TXT --dir $R/prompts
    for i in {0..15}; do
      $NEW generate --model $M --backend metal --max-seq $MAXSEQ --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos \
        --show-ids --spec off --prompt "$(cat $R/prompts/p$i.txt)" > $R/solo/ids-p$i.out 2> $R/solo/ids-p$i.err || echo "solo p$i FAILED"
      echo "solo p$i: $(grep -aoE '^prompt: [0-9]+ tokens' $R/solo/ids-p$i.err)"
    done 2>&1
    py solo $R/solo/ids-p*.err --json $R/solo.json
    ;;
  served)
    ARMS=${*:-new-mb16 new-mb16-off base-mb16 new-mb8 base-mb8}
    for arm in ${=ARMS}; do
      B=$NEW; [ ${arm%%-*} = base ] && B=$BASE
      MB=${${arm#*-mb}%%-*}
      K=1; [[ $arm == *-off ]] && K=0
      export INTARSIA_METAL_FP8_ROWS_SPLIT=$K
      start $R/served-$arm $B --max-batch $MB || continue
      curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $R/plan-$arm.json
      py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $R/served-$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
      grep -a "context clamped\|reserved for the recurrent\|specul" $R/served-$arm/server.err | sed "s/^/$arm: /" | cut -c1-200
      gzip -f $R/served-$arm/server.err
      unset INTARSIA_METAL_FP8_ROWS_SPLIT
    done
    ;;
  compare)
    for f in $R/served-*.json; do py vs-solo $f $R/solo.json; done
    py compare $R/served-base-mb16.json $R/served-new-mb16.json
    py compare $R/served-base-mb8.json $R/served-new-mb8.json
    py compare $R/served-new-mb16-off.json $R/served-new-mb16.json
    for a in new-mb8 new-mb16 new-mb16-off base-mb16; do
      diff <(python3 -m json.tool $R/plan-base-mb8.json) <(python3 -m json.tool $R/plan-$a.json) > $R/plan-diff-base-mb8-vs-$a.txt
      echo "plan base-mb8 vs $a: $(grep -c '^[<>]' $R/plan-diff-base-mb8-vs-$a.txt) differing lines"
    done
    ;;
  route)
    for spec in on-10 on-12 off-10 off-12 on-8; do
      K=1; [[ $spec == off-* ]] && K=0; N=${spec##*-}
      O=$R/route-$spec; mkdir -p $O
      INTARSIA_METAL_FP8_ROWS_SPLIT=$K INTARSIA_METAL_PROFILE=kernel:$O/prof.jsonl start $O $NEW --max-batch 16 || continue
      py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $O/ids.json --gen 24 --short --counts $N 2>&1 | sed "s/^/$spec: /"
      stop
      echo "route $spec: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$O/prof.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op']=='matmul_fp8_gemv_rows': c[r['t']]+=1
print(sorted(c.items()))")"
      gzip -f $O/prof.jsonl $O/server.err
    done
    ;;
  routedef)
    # the shipped default path: INTARSIA_METAL_FP8_ROWS_SPLIT unset, 10 streams, kernel-mode profile
    spec=def-10; N=10; O=$R/route-$spec; mkdir -p $O
    unset INTARSIA_METAL_FP8_ROWS_SPLIT
    INTARSIA_METAL_PROFILE=kernel:$O/prof.jsonl start $O $NEW --max-batch 16 || exit 1
    py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $O/ids.json --gen 24 --short --counts $N 2>&1 | sed "s/^/$spec: /"
    stop
    echo "route $spec (env unset): $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$O/prof.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op']=='matmul_fp8_gemv_rows': c[r['t']]+=1
print(sorted(c.items()))")"
    rm -f $O/prof.jsonl; gzip -f $O/server.err
    ;;
  mtp)
    for pi in 0 1; do
      ref=""
      for arm in base new; do
        B=$NEW; [ $arm = base ] && B=$BASE
        o=$R/mtp-p$pi-$arm
        $B generate --model $M --backend metal --max-seq $MAXSEQ --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
          --prompt "$(cat $R/prompts/p$pi.txt)" > $o.out 2> $o.err
        got="$(grep -a '^ids' $o.err | md5 -q)"; [ -z "$ref" ] && ref=$got
        echo "mtp p$pi $arm: $got $(grep -aoE 'specul[^,]*' $o.err | head -1) $([ $got = $ref ] && echo SAME || echo DIFFER)"
      done
    done
    ;;
esac
echo "## done $MODEL $STEP"
