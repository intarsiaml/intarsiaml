#!/usr/bin/env bash
# Lane rip GPU job (spark2, --quiet gpu,cpu): the speed A/B of row-invariant prefill, same binary, arm A =
# INTARSIA_ROW_INVARIANT_PREFILL=0 (the ordinary prefill, BI8's reuse rule), arm B = INTARSIA_ROW_INVARIANT_PREFILL=1 (opt-in; the
# measured run had it as the default, `0f94ac8`).
# Per round and arm: prefill on a solo server (localeval speed --sizes 1024,4096,16384 --gen 16 --reps 3, one stream), then the
# 16-stream decode cell on a --max-batch 16 server (BI8 / SB8's cell: --sizes 1024 --gen 256 --reps 3 --concurrency 16).
# ROUNDS (default 3) rounds A B for the 8B. The 35B has row-invariant prefill off (the plan finds no row-invariant MoE / NVFP4
# kernels), so the env changes nothing there: its arms are A = ~/intarsiaml-rip2/bin/base (origin/main), B = the fix binary,
# ROUNDS35 (default 1) rounds. localeval reports cached_tokens_max per size (a nonce leads every prompt; > 64 cached is flagged).
#   OUT=... fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 75 --label "rip2: speed A/B" -- scripts/rip2/speed.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
OUT=${OUT:?}; mkdir -p "$OUT"
HUB=/home/nvidia/.cache/huggingface/hub
M8=$HUB/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
M35=$HUB/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BINDIR=/home/nvidia/intarsiaml-rip2/bin
BIN=$BINDIR/intarsia
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
start() { # log env args
  SPID=$(rsh "nohup env $2 ${SBIN:-$BIN} serve $3 > ~/intarsiaml-rip2/out/speed-$1.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && return 0; sleep 2; done
  echo "server $1 did not come up"; stop_server; return 1
}
le() { # label sizes gen conc recipe
  (cd /opt/code/deepeval && uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes "$2" --gen "$3" --reps 3 \
    --concurrency "$4" --label "rip2-$1" --recipe "$5" 2>&1 | grep -E "saved|unim|tok/s|cached|^\|" | tail -16)
}
arm() { # model-tag model round arm
  local tag=$1 M=$2 r=$3 a=$4 env args
  SBIN=$BIN
  if [[ $tag == 35b ]]; then
    env="INTARSIA_ROW_INVARIANT_PREFILL=default"; [[ $a == A ]] && SBIN=$BINDIR/base
  elif [[ $a == A ]]; then env="INTARSIA_ROW_INVARIANT_PREFILL=0"; else env="INTARSIA_ROW_INVARIANT_PREFILL=1"; fi
  args="--backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --profile-every 0"
  local solo="$args --max-seq 17408 --mem-budget 40" mb="$args --max-seq 8192 --mem-budget 40 --max-batch 16"
  if start "$tag-prefill-$a-r$r" "$env" "$solo"; then
    le "$tag-prefill-$a-r$r" 1024,4096,16384 16 1 "env $env $SBIN serve $solo" | tee "$OUT/$tag-prefill-$a-r$r.log"
    stop_server
  fi
  if start "$tag-decode-$a-r$r" "$env" "$mb"; then
    le "$tag-decode-$a-r$r" 1024 256 16 "env $env $SBIN serve $mb" | tee "$OUT/$tag-decode-$a-r$r.log"
    stop_server
  fi
}
for r in $(seq 1 "${ROUNDS:-3}"); do for a in A B; do arm 8b "$M8" "$r" "$a"; done; done
for r in $(seq 1 "${ROUNDS35:-0}"); do for a in A B; do arm 35b "$M35" "$r" "$a"; done; done
echo "== done"
