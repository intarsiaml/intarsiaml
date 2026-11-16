#!/bin/zsh
# Lane mnr end-to-end arms on endor (Qwen3.8-27B MXFP4, Metal), knob INTARSIA_METAL_NAX_ROWS off / on, same binary, fresh process
# per arm. Run inside a fleetq slot:
#   fleetq run --host endor --expect 30 --mem 30 --quiet gpu --label "mnr: ..." -- scripts/mnr/e2e.sh <arm>...
# Arms: acc-off acc-on   accuracy gate against the HF W4A16 golden (prefill + decode KL21 / top-1)
#       b8-off b8-on     serve --max-batch 8, localeval speed at 1k, 1 / 2 / 4 / 8 streams (repeat arms for ABBA: b8-off b8-on b8-on b8-off)
#       ids-on ids-off   256 greedy ids, --spec off against the MTP default and MTP k4 (short prompt and 1k), must be equal
#       mmlu-on          localeval quality mmlu, limit 25 (n = 1425), MTP default
set -u
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h}/../..; W=${W:A}; BIN=$W/target-bin/intarsia-mnr; R=$W/docs/results/2026-11-15-mnr-metal-nax-rows/raw; mkdir -p $R
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
G=/opt/code/intarsiaml/docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
NAT=/opt/code/deepeval/tests/data/prompts/natural-v1
PORT=18641
typeset -A SEQ
knob_env() { [[ $1 == on ]] && echo INTARSIA_METAL_NAX_ROWS=1 || echo MNR_KNOB_OFF=1 }

serve_arm() { # tag knob localeval-subcommand-args... -- serve args...
  local tag=$1 knob=$2; shift 2
  local LA=(); while [[ $1 != -- ]]; do LA+=($1); shift; done; shift
  SEQ[$tag]=$(( ${SEQ[$tag]:-0} + 1 )); local O=$R/$tag-${SEQ[$tag]}; mkdir -p $O
  local CMD=($BIN serve --backend metal --model $M27 --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@")
  echo "## $tag (knob $knob) $(date +%T): ${CMD[*]}"
  env $(knob_env $knob) $CMD > $O/server.out 2> $O/server.err < /dev/null &
  local TP=$! up=0
  for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { echo "SERVER FAILED $tag"; tail -5 $O/server.err; kill $TP 2>/dev/null; return 1; }
  (cd /opt/code/deepeval && uv run localeval $LA --url http://127.0.0.1:$PORT/v1 --model intarsia --thinking off --no-unim \
     --label "mnr-$tag" --recipe "$(knob_env $knob) ${CMD[*]}" > $O/localeval.out 2> $O/localeval.err; echo "localeval exit $?")
  kill -INT $TP; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null
  grep -aE "error|panick" $O/server.err | head -3
  tail -14 $O/localeval.out
}

gen_ids() { # out knob prompt-file extra generate args...
  local out=$1 knob=$2 pf=$3; shift 3
  env $(knob_env $knob) $BIN generate --model $M27 --backend metal --max-seq 20000 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids "$@" \
    --prompt "$(cat $pf)" > $out.out 2> $out.err < /dev/null
  echo "$(basename $out): exit $? $(grep -aoE '(prefill|decode) [0-9]+ ms \([0-9.]+ tok/s\)' $out.err | tr '\n' ' ')"
}

for arm in "$@"; do
  knob=${arm##*-}
  case $arm in
    acc-*)
      echo "## $arm $(date +%T)"
      env $(knob_env $knob) $BIN accuracy --backend metal --model $M27 --max-seq 4096 --mxfp4-act none --golden $G --mode both --decode-positions 16 \
        > $R/acc-$knob.out 2> $R/acc-$knob.err < /dev/null; echo "accuracy $knob exit $?"
      grep -aE "^== mode|^ALL|^all|mean|total|PASS|FAIL" $R/acc-$knob.out | head -20 ;;
    b8-*)   serve_arm b8-$knob $knob speed --sizes 1024 --gen 128 --reps 1 --concurrency 2,4,8 -- --max-batch 8 ;;
    mmlu-*) serve_arm mmlu-$knob $knob quality --structured none --bench mmlu --limit 25 -- ;;
    ids-*)
      mkdir -p $R/ids; print -r -- "Write a short story about a lighthouse keeper who finds a message in a bottle." > $R/ids/short.txt
      for p in short n1024; do
        pf=$R/ids/short.txt; [[ $p == n1024 ]] && pf=$NAT/n1024.txt
        gen_ids $R/ids/$p-$knob-off $knob $pf --spec off
        gen_ids $R/ids/$p-$knob-mtp2 $knob $pf
        gen_ids $R/ids/$p-$knob-mtp4 $knob $pf --spec mtp --spec-k 4
        for v in mtp2 mtp4; do
          cmp -s $R/ids/$p-$knob-off.out $R/ids/$p-$knob-$v.out && echo "IDS SAME $p knob $knob: $v == spec off" || echo "IDS DIFF $p knob $knob: $v vs spec off"
        done
      done ;;
  esac
done
echo "## done $(date +%T)"
