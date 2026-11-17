#!/bin/zsh
# Lane mchunk4 gates (27B, Metal), adapted from scripts/mb16/gates.sh. Run under fleetq (--quiet gpu,cpu). Binaries (copied aside, so a
# later `cargo test --workspace --release` that rebuilds target/release/intarsia without --features metal cannot hurt):
#   NEW  = .base/bin-mchunk4/intarsia (this branch), BASE = .base/bin-main/intarsia (origin/main 12590c55e)
#   solo     : `generate --spec off --show-ids` (256 tokens, greedy) on each of the 16 prompts of served_ids.py, NEW binary
#   served   : serve arms new-mb16, new-mb8, new-mb16-chunk0, new-mb8-chunk0 (INTARSIA_METAL_MXFP4_ROWS_CHUNK=0), base-mb8; 1 / 4 / 6 / 8 / 9 / 12 / 16
#              concurrent requests (even prompts ~99 tokens, odd ~4666), ids per stream, plan JSON of each server
#   compare  : every served stream vs its solo ids; arms vs each other; plan diffs
#   route    : kernel-mode profile of the batched decode at 6 / 8 streams (mb8), 12 streams (mb16), knob on and off: matmul_mxfp4_gemv_rows launches by M
#   mtp      : `generate` with the default engine (MTP), NEW vs BASE ids, short + 4k
# usage: gates.sh solo | served [arm...] | compare | route | mtp
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mchunk4; R=$W/docs/results/2026-11-16-mchunk4/raw/gates; mkdir -p $R/prompts $R/solo
BASE=$W/.base/bin-main/intarsia; NEW=$W/.base/bin-mchunk4/intarsia
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
MAXSEQ=${MAXSEQ:-8192}
py() { python3 $W/scripts/mchunk4/served_ids.py "$@"; }
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
    shift; ARMS=${*:-new-mb16 new-mb8 new-mb16-chunk0 new-mb8-chunk0 base-mb8}
    for arm in ${=ARMS}; do
      B=$NEW; [ ${arm%%-*} = base ] && B=$BASE
      MB=${${arm#*-mb}%%-*}
      K=1; [[ $arm == *-chunk0 ]] && K=0
      export INTARSIA_METAL_MXFP4_ROWS_CHUNK=$K
      start $R/served-$arm $B --max-batch $MB || continue
      curl -s http://127.0.0.1:$PORT/v1/intarsia/plan > $R/plan-$arm.json
      py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $R/served-$arm.json 2>&1 | sed "s/^/$arm: /"
      stop
      grep -a "context clamped\|reserved for the recurrent\|specul" $R/served-$arm/server.err | sed "s/^/$arm: /" | cut -c1-200
      gzip -f $R/served-$arm/server.err
      unset INTARSIA_METAL_MXFP4_ROWS_CHUNK
    done
    ;;
  compare)
    for f in $R/served-*.json; do py vs-solo $f $R/solo.json; done
    py compare $R/served-base-mb8.json $R/served-new-mb8.json
    py compare $R/served-new-mb8.json $R/served-new-mb8-chunk0.json
    py compare $R/served-new-mb16.json $R/served-new-mb16-chunk0.json
    for a in new-mb8 new-mb8-chunk0 new-mb16 new-mb16-chunk0; do
      diff <(python3 -m json.tool $R/plan-base-mb8.json) <(python3 -m json.tool $R/plan-$a.json) > $R/plan-diff-base-mb8-vs-$a.txt
      echo "plan base-mb8 vs $a: $(grep -c '^[<>]' $R/plan-diff-base-mb8-vs-$a.txt) differing lines"
    done
    ;;
  route)
    # kernel-mode profile: every launch is its own command buffer, labelled with the kernel and `t` (the launch's rows).
    # Short prompts only (odd streams' 4k prompts would only add prefill records, which carry other kernels).
    for spec in mb8-on-6 mb8-on-8 mb8-off-6 mb8-off-8 mb16-on-12 mb16-off-12; do
      MB=${${spec%%-*}#mb}; K=1; [[ $spec == *-off-* ]] && K=0; N=${spec##*-}
      O=$R/route-$spec; mkdir -p $O
      INTARSIA_METAL_MXFP4_ROWS_CHUNK=$K INTARSIA_METAL_PROFILE=kernel:$O/prof.jsonl start $O $NEW --max-batch $MB || continue
      py run --url http://127.0.0.1:$PORT/v1 --text $TXT --out $O/ids.json --gen 8 --counts $N 2>&1 | sed "s/^/$spec: /"
      stop
      echo "route $spec: $(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$O/prof.jsonl'):
    if l.startswith('{'):
        r=json.loads(l)
        if r['op'].startswith('matmul_mxfp4'): c[(r['op'],r['t'])]+=1
print(sorted(c.items()))")"
      gzip -f $O/prof.jsonl $O/server.err
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
