#!/bin/zsh
# Lane df, Phase 1 on endor (M5 Max): ONE exclusive slot.
#   fleetq run --host endor --exclusive --mem 40 --expect 90 --label "df metal p1" -- zsh scripts/df/metal-p1.sh <binary> <out dir>
# Part A (ours, MXFP4 W4A16): generate traces (INTARSIA_SPEC_TRACE=1) at short-0 and 4k-1, off / mtp2 / mtp7 with the eager MTP step
#   (INTARSIA_METAL_MTP_GPU_ACCEPT=0: prints draft and verify+rollback per pass), mtp2 on the default pipe path; ids vs off; one
#   op table each for off and mtp7.
# Part B (end to end, localeval suite cells speed-plain = filler, speed-spec = natural-v1): ours off, ours mtp2 (default);
#   TensorFold (the standing service's binary and checkpoint, Vontra/Qwen3.8-27B-MLX-4bit + z-lab/Qwen3.8-27B-DFlash2, own port,
#   offline) with DFlash2 (its default) and with --drafter none.
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}; PORT=8297
TF=${TF:-/Users/josef/.local/share/tensorfold-mac-all/tensorfold}
TOK=(--tokenizer-repo Vontra/Qwen3.8-27B-MLX-4bit --tokenizer-revision 70ae7fac63274ff2eac54152031433374cb80f2f)
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT
if [[ ${PARTS:-AB} == *A* ]]; then
for n in short-0 4k-1; do
  P="$(cat $R/scripts/dfh/ids-prompts/$n.txt)"
  for a in "off:--spec off" "mtp2e:--spec mtp --spec-k 2" "mtp7e:--spec mtp --spec-k 7" "mtp2:--spec mtp --spec-k 2"; do
    tag=${a%%:*}; args=(${=a#*:}); envs=(INTARSIA_SPEC_TRACE=1); [[ $tag == *e ]] && envs+=(INTARSIA_METAL_MTP_GPU_ACCEPT=0)
    env $envs $BIN generate -m $M --backend metal --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos --show-ids -p "$P" --max-tokens 256 $args \
      > $O/gen-$n-$tag.out 2> $O/gen-$n-$tag.err < /dev/null
    rc=$?; grep -a '^ids:' $O/gen-$n-$tag.err > $O/gen-$n-$tag.ids
    same=FAIL; [[ -s $O/gen-$n-off.ids ]] && cmp -s $O/gen-$n-off.ids $O/gen-$n-$tag.ids && same=PASS
    say "$n $tag exit $rc ids-vs-off $same | $(grep -a 'verify + rollback' $O/gen-$n-$tag.err | tail -n +5 | awk '{s+=$(NF-1); d+=$4; c++} END{if(c)printf "mean draft %.2f verify+rollback %.2f ms n=%d", d/c, s/c, c}') | $(grep -a 'mtp pipe step' $O/gen-$n-$tag.err | tail -n +5 | awk '{s+=$5; c++} END{if(c)printf "pipe step %.2f ms n=%d", s/c, c}') | $(grep -aE 'tok/s' $O/gen-$n-$tag.err | tail -2 | cut -c1-150 | tr '\n' '|')"
  done
done
P="$(cat $R/scripts/dfh/ids-prompts/short-0.txt)"
for a in "off:--spec off" "mtp7e:--spec mtp --spec-k 7"; do
  tag=${a%%:*}; args=(${=a#*:})
  env INTARSIA_METAL_MTP_GPU_ACCEPT=0 $BIN generate -m $M --backend metal --max-seq 8192 --chat --no-think --profile-every 1 --op-table --ignore-eos -p "$P" --max-tokens 96 $args \
    > $O/optable-$tag.out 2> $O/optable-$tag.err < /dev/null
  say "optable $tag exit $?"
done
fi
up() { local i; for i in {1..300}; do curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; kill -0 $SPID 2>/dev/null || return 1; sleep 1; done; return 1; }
down() { kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done; }
suite() { # suite <tag> <model id> <lane> <drafter> <recipe> <cells>
  local tag=$1 mid=$2 lane=$3 dr=$4 rec=$5 cells=$6 c
  for c in ${=cells}; do
    ( cd /opt/code/deepeval && uv run localeval suite --url http://127.0.0.1:$PORT/v1 --model $mid --thinking off $TOK --lane $lane --drafter $dr --max-num-seqs 1 \
        --max-context 20480 --power-mode default --tier adhoc --cells $c --label "df-metal-$tag-$c" --recipe "$rec" --wait-idle 60 ) > $O/suite-$tag-$c.log 2>&1 < /dev/null
    say "suite $tag $c exit $? $(grep -aoE 'results/[^`]+' $O/suite-$tag-$c.log | tail -1)"
  done
}
if [[ ${PARTS:-AB} == *B* ]]; then
for v in ${=ARMS:-off mtp2 tf tfnone}; do
  say "serve $v"
  case $v in
    off|mtp2)
      extra=(); [[ $v == off ]] && extra=(--spec off)
      $BIN serve --backend metal --model $M --served-model-name qwen38-27b --max-seq 20480 --profile-every 0 --no-supervisor $extra --host 127.0.0.1 --port $PORT \
        > $O/serve-$v.out 2> $O/serve-$v.err < /dev/null &
      SPID=$!; mid=qwen38-27b; rec="intarsia serve --backend metal --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 20480 ${extra[*]} (default MTP k2 for mtp2; lane df)" ;;
    tf|tfnone)
      extra=(); [[ $v == tfnone ]] && extra=(--drafter none)
      HF_HUB_OFFLINE=1 $TF serve Vontra/Qwen3.8-27B-MLX-4bit --host 127.0.0.1 --port $PORT --context 20480 --name bench $extra > $O/serve-$v.out 2> $O/serve-$v.err < /dev/null &
      SPID=$!; mid=bench; rec="tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --context 20480 --name bench ${extra[*]} (binary of the endor service, $(basename $(readlink -f $TF 2>/dev/null || echo $TF)); DFlash2 z-lab/Qwen3.8-27B-DFlash2 unless --drafter none; lane df)" ;;
  esac
  if ! up; then say "serve $v failed"; tail -20 $O/serve-$v.err >> $O/steps.log; down; continue; fi
  [[ $v == tf || $v == tfnone ]] && mid=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  case $v in off|tfnone) suite $v $mid plain none "$rec" "speed-plain" ;; mtp2) suite $v $mid spec:mtp mtp "$rec" "speed-plain speed-spec" ;; tf) suite $v $mid spec:dflash dflash2 "$rec" "speed-plain speed-spec" ;; esac
  down
done
fi
say done
