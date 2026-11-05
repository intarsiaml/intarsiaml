#!/usr/bin/env bash
# Lane d8 gates on the CUDA box (GB10). Runs ON that box (scripts/d8/spark1.sh gates). Binaries: dev = ~/intarsiaml-d8-target (this branch), base = ~/intarsiaml-d8-target-base (origin/main).
#   gates.sh ids          default (no --spec) vs --spec off vs INTARSIA_SPEC_DEFAULT=0, greedy ids, 256 tokens, 8B prompts of scripts/dc, scripts/hg
#   gates.sh fallback     server cases: --max-batch 4 / missing draft / bad INTARSIA_DFLASH_DRAFT / explicit --spec off / sampled request: all answer, no verify pass
#   gates.sh plan         plan show (json + text), base vs dev, flags off: byte-identical
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
SRC=$HOME/intarsiaml-d8-src; O=$HOME/intarsiaml-d8-out; mkdir -p $O
DEV=$HOME/intarsiaml-d8-target/release/intarsia; BASE=$HOME/intarsiaml-d8-target-base/release/intarsia
md() { md5sum "$1" | cut -c1-8; }
gen() { # gen <out-prefix> <prompt-file> <binary> [args...]; env from ENVX
  local pre=$1 f=$2 b=$3; shift 3
  env $ENVX $b generate -m $M8 --backend cuda --mem-budget 30 --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
    -p "$(cat $f)" --max-tokens ${NTOK:-256} "$@" < /dev/null > /dev/null 2> $pre.err
  grep -a '^ids:' $pre.err > $pre.ids
}
case "${1:?}" in
ids)
  fails=0
  for id in ${IDS:-short 4k math1 math2 code1 chat1 prose-lighthouse}; do
    f=$SRC/scripts/dc/prompts/$id.txt; [ -f "$f" ] || f=$SRC/scripts/hg/prompts/$id.txt
    ENVX= gen $O/ids-$id-default $f $DEV
    ENVX= gen $O/ids-$id-off $f $DEV --spec off
    ENVX=INTARSIA_SPEC_DEFAULT=0 gen $O/ids-$id-envoff $f $DEV
    st=$(grep -a '^spec:' $O/ids-$id-default.err | tail -1 | cut -c1-150); on=$(grep -ac 'the default of recipe' $O/ids-$id-default.err)
    fin=$(grep -a 'finish=' $O/ids-$id-default.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60); fo=$(grep -a 'finish=' $O/ids-$id-off.err | tail -1 | sed 's/.*| decode/decode/' | cut -c1-60)
    if [ -s $O/ids-$id-off.ids ] && cmp -s $O/ids-$id-off.ids $O/ids-$id-default.ids && cmp -s $O/ids-$id-off.ids $O/ids-$id-envoff.ids && [ "$on" -ge 1 ]; then
      echo "PASS ids $id: default == --spec off == INTARSIA_SPEC_DEFAULT=0 ($(tr ',' '\n' < $O/ids-$id-off.ids | wc -l) ids, md5 $(md $O/ids-$id-off.ids)); default line seen; off: $fo | default: $fin | $st"
    else echo "FAIL ids $id (default line count $on) | $st"; fails=$((fails+1)); fi
  done
  echo "ids fails=$fails" ;;
fallback)
  fails=0; PORT=8098
  serve() { # serve <tag> <env> <args...>
    local tag=$1 e=$2; shift 2
    env $e $DEV serve --backend cuda --model $M8 --model-id Qwen3-8B --port $PORT --host 0.0.0.0 --max-seq 4096 --mem-budget 24 "$@" > $O/fb-$tag.log 2>&1 < /dev/null &
    echo $! > $O/fb.pid
    for _ in $(seq 150); do curl -sf http://localhost:$PORT/health > /dev/null && return 0; sleep 2; done; return 1
  }
  stopsrv() { p=$(cat $O/fb.pid); kill $p 2>/dev/null; for _ in $(seq 60); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; }
  steps() { curl -s http://localhost:$PORT/metrics | grep -E '^intarsia_spec_steps_total' | awk '{s+=$2} END {print s+0}'; }
  ask() { # ask <temperature> -> prints the content length and the completion tokens
    curl -s http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"Qwen3-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that checks whether a number is prime, with a short explanation.\"}],\"max_tokens\":96,\"temperature\":$1,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['choices'][0]['message']['content']), d['usage']['completion_tokens'])"
  }
  case_() { # case_ <tag> <env> <expect-dflash 0|1> <serve args...>
    local tag=$1 e=$2 want=$3; shift 3
    if ! serve "$tag" "$e" "$@"; then echo "FAIL fallback $tag: server did not come up"; tail -5 $O/fb-$tag.log; stopsrv; fails=$((fails+1)); return; fi
    s0=$(steps); g=$(ask 0); s1=$(steps); t=$(ask 0.8); s2=$(steps)
    line=$(grep -a -E 'default of recipe|not applied|--spec dflash on' $O/fb-$tag.log | head -2 | cut -c1-230 | tr '\n' '|')
    ok=1; [ -n "$g" ] && [ -n "$t" ] || ok=0
    if [ "$want" = 1 ]; then [ "$s1" -gt "$s0" ] || ok=0; [ "$s2" = "$s1" ] || ok=0; else [ "$s2" = "$s0" ] || ok=0; fi
    [ $ok = 1 ] && echo "PASS fallback $tag: greedy '$g' sampled '$t', verify passes greedy $((s1-s0)) sampled $((s2-s1)) (want dflash=$want) | $line" || { echo "FAIL fallback $tag: greedy '$g' sampled '$t' steps $s0 $s1 $s2 | $line"; fails=$((fails+1)); }
    stopsrv
  }
  case_ default "" 1 --max-batch 1
  case_ batch4 "" 0 --max-batch 4
  case_ nodraft "HF_HUB_CACHE=$O/emptyhub" 0 --max-batch 1
  case_ baddraft "INTARSIA_DFLASH_DRAFT=$O/nothing" 0 --max-batch 1
  case_ specoff "" 0 --max-batch 1 --spec off
  case_ envoff "INTARSIA_SPEC_DEFAULT=0" 0 --max-batch 1
  echo "fallback fails=$fails" ;;
plan)
  fails=0
  for t in base dev; do
    if [ $t = base ]; then B=$BASE; else B=$DEV; fi
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 --json < /dev/null > $O/plan-$t.json 2> /dev/null
    $B plan show --backend cuda --model $M8 --max-seq 8192 --mem-budget 30 < /dev/null > $O/plan-$t.txt 2> /dev/null
  done
  for k in json txt; do cmp -s $O/plan-base.$k $O/plan-dev.$k && [ -s $O/plan-dev.$k ] && echo "PASS plan $k byte-identical ($(wc -c < $O/plan-dev.$k) B)" || { echo "FAIL plan $k"; fails=$((fails+1)); }; done
  for p in short 4k; do
    f=$SRC/scripts/hg/prompts/$p.txt
    ENVX= gen $O/noop-$p-base $f $BASE; ENVX= gen $O/noop-$p-dev $f $DEV --spec off
    cmp -s $O/noop-$p-base.ids $O/noop-$p-dev.ids && [ -s $O/noop-$p-dev.ids ] && echo "PASS noop ids $p: main == branch --spec off ($(tr ',' '\n' < $O/noop-$p-dev.ids | wc -l) ids)" || { echo "FAIL noop ids $p"; fails=$((fails+1)); }
  done
  echo "plan fails=$fails" ;;
esac
