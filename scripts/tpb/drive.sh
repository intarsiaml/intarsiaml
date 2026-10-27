#!/usr/bin/env bash
# Lane TPB window driver, run FROM THE MAC inside ONE fleetq job that holds both R9700s and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 40 \
#     --label "tpb ..." -- env W=wa ITEMS="..." scripts/tpb/drive.sh
# ITEMS = space separated "ARM[+mod...]:ACTION", run in order; a fresh `serve` starts whenever ARM+mods changes.
#   ARM   f32 (INTARSIA_TP_PARTIAL=f32) | bf16 (INTARSIA_TP_PARTIAL=bf16) | rec (no env: the recipe decides)
#         | any other name NAME: envs from ENV_NAME (e.g. ENV_hyb="INTARSIA_TP_PARTIAL=bf16 INTARSIA_X=1")
#   mods  spoff (--spec off) | mbN (--max-batch N) | bin=NAME (binary ~/intarsiaml-tpb/NAME instead of tpb-intarsia)
#   ACTION speed (localeval speed --sizes $SIZES --gen 128 --reps 3) | mmlu (localeval quality --bench mmlu --limit 25 --thinking off)
#          | texts (texts.py: md5 of 256 greedy forced tokens for the 18 gate prompts) | mixids (mixids.py: texts of A and B when B prefills beside A)
#          | kl (stops the server, runs scripts/tpb/acc2.sh style KL for the arm's env on tatooine; needs the engine, not the server)
# kl: ARM is ignored (KLARMS env lists the acc2.sh arms)
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}" "${ITEMS:?ITEMS=...}"
SIZES=${SIZES:-1024,4096,16384}; MS=${MS:-20480}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-27-tpb-hip-tp-partials/raw/$W; mkdir -p "$OUT" "$REPO/tpb-out"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO=$(mktemp -u "$REPO/tpb-out/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/tpb-bin' < /dev/null; scp -q "$HERE/guard.sh" tatooine:tpb-bin/
$SSH 'bash ~/tpb-bin/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stop_serve() { $SSH 'docker rm -f tpb-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() { stop_serve; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

armenv() { # armenv <base arm> -> docker -e args
  case $1 in
    f32) echo "-e INTARSIA_TP_PARTIAL=f32" ;;
    bf16) echo "-e INTARSIA_TP_PARTIAL=bf16" ;;
    rec) echo "" ;;
    *) local v="ENV_$1" e=""; for kv in ${!v:-}; do e="$e -e $kv"; done; echo "$e" ;;
  esac
}
serve() { # serve <arm+mods>
  local full=$1 IFS='+' parts; read -ra parts <<< "$full"; local arm=${parts[0]} bin=tpb-intarsia mb=1 extra="" m
  for m in "${parts[@]:1}"; do
    case $m in
      spoff) extra="$extra --spec off" ;;
      mb*) mb=${m#mb} ;;
      bin=*) bin=${m#bin=} ;;
      *) echo "unknown mod $m" >&2; return 1 ;;
    esac
  done
  local envs; envs=$(armenv "$arm")
  RECIPE="HIP_VISIBLE_DEVICES=0,1 $bin serve --backend hip --model 27B --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS$extra (lane tpb, arm $full, env ${envs:-none}, docker $IMG)"
  stop_serve
  $SSH "docker run -d --name tpb-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 $envs \
    -v \$HOME/intarsiaml-tpb:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model $M27 \
    --served-model-name qwen38-27b-tpb --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS$extra --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$full.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 tpb-serve' > "$OUT/serve-fail-$full.log" 2>&1 < /dev/null; return 1
}
tag() { echo "$1" | tr '+:=' '___'; }
cur=""; n=0
for item in $ITEMS; do
  n=$((n + 1)); full=${item%%:*}; act=${item#*:}; t="$n-$(tag "$full")-$act"
  if [[ $full != "$cur" ]]; then
    say "item $n: starting server $full"
    if ! serve "$full"; then say "item $n $full: server did not come up"; cur=""; continue; fi
    $SSH 'docker logs tpb-serve 2>&1 | head -80' < /dev/null > "$OUT/serve-$(tag "$full")-$n.log" 2>&1
    printf '[targets.tpb-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen38-27b-tpb"\n' "$(tag "$full")" "$PORT" > "$OUT/targets-$(tag "$full").toml"
    cur=$full
  fi
  TG=$OUT/targets-$(tag "$full").toml; TN=tpb-$(tag "$full")
  case $act in
    speed)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval speed "$TN" --sizes "$SIZES" --gen 128 --reps 3 \
          --label "tpb-$W-$(tag "$full")-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)" ;;
    mmlu)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval quality "$TN" --bench mmlu --limit 25 --thinking off \
          --label "tpb-$W-$(tag "$full")-mmlu-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/mmlu-$t.log" 2>&1 < /dev/null
      say "item $n $full mmlu: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/mmlu-$t.log" | tail -1)" ;;
    texts)
      python3 "$HERE/texts.py" "http://tatooine:$PORT" qwen38-27b-tpb "$REPO/scripts/cmtp/ids-prompts" > "$OUT/texts-$t.txt" 2>&1
      say "item $n $full texts: exit $? $(wc -l < "$OUT/texts-$t.txt") lines" ;;
    mixids)
      python3 "$HERE/mixids.py" "http://tatooine:$PORT" qwen38-27b-tpb "$REPO/scripts/cmtp/ids-prompts/4k-0.txt" "$REPO/scripts/cmtp/ids-prompts/16k-3.txt" "$REPO/scripts/cmtp/ids-prompts/4k-3.txt" > "$OUT/mixids-$t.txt" 2>&1
      say "item $n $full mixids: $(tr '\n' '|' < "$OUT/mixids-$t.txt" | cut -c1-300)" ;;
    kl)   # engine KL against the W4A16 golden (acc2.sh, both arms), no server: stop it first
      stop_serve; cur=""
      $SSH "O=\$HOME/intarsiaml-tpb-logs/acc2-$W ARMS='${KLARMS:-f32 bf16}' bash ~/intarsiaml-tpb/scripts/tpb/acc2.sh" < /dev/null > "$OUT/kl-$t.txt" 2>&1
      scp -q "tatooine:intarsiaml-tpb-logs/acc2-$W/acc-*.log" "$OUT/" < /dev/null
      say "item $n kl: $(tr '\n' '|' < "$OUT/kl-$t.txt" | cut -c1-500)" ;;
    *) say "item $n: unknown action $act" ;;
  esac
done
say "done"
