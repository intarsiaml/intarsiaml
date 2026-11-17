#!/bin/zsh
# Lane mb16 gates (27B, Metal). Run under fleetq. Binaries: NEW = this branch built with --features metal, copied to .base/bin-mb16/intarsia (a later cargo build without the feature overwrites target/release), BASE = main c60d26caf built
# into .base/target (git archive, scripts/mb16/build-base.sh).
#   solo     : `generate --spec off --show-ids` (256 tokens, greedy) on each of the 16 prompts of served_ids.py, NEW binary: the
#              solo reference of every served stream
#   served   : serve NEW --max-batch 16, NEW --max-batch 8, BASE --max-batch 8 (each --spec off is implied: Metal batching runs no MTP);
#              1 / 4 / 8 / 9 / 12 / 16 concurrent requests, ids per stream; plan JSON of each server
#   compare  : every served stream vs its solo ids; NEW mb8 vs BASE mb8; plan diffs
#   route    : routing traces (INTARSIA_METAL_PROFILE=cb) of a 125-token prefill on NEW and BASE (matmul_mxfp4 calls by kernel and M)
#   mtp      : `generate` with the default engine (MTP verify rows now take matmul_mxfp4_gemv_rows), NEW vs BASE ids, short + 4k
# usage: gates.sh solo | served | compare | route | mtp
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mb16; R=$W/docs/results/2026-11-16-mb16/raw/gates; mkdir -p $R/prompts
BASE=$W/.base/target/release/intarsia; NEW=$W/.base/bin-mb16/intarsia
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
MAXSEQ=${MAXSEQ:-8192}
py() { python3 $W/scripts/mb16/served_ids.py "$@"; }
PORT=18675
start() { local O=$1 B=$2; shift 2; mkdir -p $O
  $B serve --backend metal --model $M27 --served-model-name intarsia --max-seq $MAXSEQ --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $O/server.out 2> $O/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $O: $(tail -3 $O/server.err)"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
case ${1:?step} in
  solo)
    py prompts --text $TXT --dir $R/prompts
    for i in {0..15}; do
      $NEW generate --model $M27 --backend metal --max-seq $MAXSEQ --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos \
        --show-ids --spec off --prompt "$(cat $R/prompts/p$i.txt)" > $R/solo/ids-p$i.out 2> $R/solo/ids-p$i.err || echo "solo p$i FAILED"
      echo "solo p$i: $(grep -aoE '^prompt: [0-9]+ tokens' $R/solo/ids-p$i.err)"
    done 2>&1
    py solo $R/solo/ids-p*.err --json $R/solo.json
    ;;
  served)
    for arm in new-mb16 new-mb8 base-mb8; do
      B=$NEW; [ ${arm%%-*} = base ] && B=$BASE
      start $R/served-$arm $B --max-batch ${arm##*-mb} || continue
      curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $R/plan-$arm.json
      py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $R/served-$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
      grep -a "context clamped\|reserved for the recurrent\|specul" $R/served-$arm/server.err | sed "s/^/$arm: /" | cut -c1-200
      gzip -f $R/served-$arm/server.err
    done
    ;;
  compare)
    for arm in new-mb16 new-mb8 base-mb8; do py vs-solo $R/served-$arm.json $R/solo.json; done
    py compare $R/served-base-mb8.json $R/served-new-mb8.json
    for a in new-mb8 new-mb16; do
      diff <(python3 -m json.tool $R/plan-base-mb8.json) <(python3 -m json.tool $R/plan-$a.json) > $R/plan-diff-base-mb8-vs-$a.txt
      echo "plan base-mb8 vs $a: $(grep -c '^[<>]' $R/plan-diff-base-mb8-vs-$a.txt) differing lines"
    done
    ;;
  route)
    P="$(head -c 520 $TXT)"
    for arm in new base; do
      B=$NEW; [ $arm = base ] && B=$BASE
      INTARSIA_METAL_PROFILE=cb:$R/route-$arm.jsonl INTARSIA_METAL_COMMIT_EVERY=1 $B generate --model $M27 --backend metal --max-seq $MAXSEQ \
        --profile-every 0 --temperature 0 --max-tokens 2 --ignore-eos --spec off --prompt "$P" > $R/route-$arm.out 2> $R/route-$arm.err
      echo "route $arm: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$R/route-$arm.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op'].startswith('matmul_mxfp4'): c[(r['op'],r['t'])]+=1
print(sorted(c.items()))")"
      gzip -f $R/route-$arm.jsonl
    done
    ;;
  mtp)
    for pi in 0 1; do
      ref=""
      for arm in base new; do
        B=$NEW; [ $arm = base ] && B=$BASE
        o=$R/mtp-p$pi-$arm
        $B generate --model $M27 --backend metal --max-seq $MAXSEQ --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
          --prompt "$(cat $R/prompts/p$pi.txt)" > $o.out 2> $o.err
        got="$(grep -a '^ids' $o.err | md5 -q)"; [ -z "$ref" ] && ref=$got
        echo "mtp p$pi $arm: $got $(grep -aoE 'specul[^,]*' $o.err | head -1) $([ $got = $ref ] && echo SAME || echo DIFFER)"
      done
    done
    ;;
esac
echo "## done $*"
