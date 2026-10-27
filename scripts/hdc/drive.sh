#!/usr/bin/env bash
# Lane HDC window driver, run FROM THE MAC inside ONE fleetq job that holds both R9700s and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 40 \
#     --label "tpb ..." -- env W=wa ITEMS="..." scripts/hdc/drive.sh
# ITEMS = space separated "ARM[+mod...]:ACTION", run in order; a fresh `serve` starts whenever ARM+mods changes.
#   ARM   rec (no env: the recipe decides) | NAME: envs from ENV_NAME (e.g. ENV_old="INTARSIA_HIP_MXFP4_A8_PD=1")
#   mods  spoff (--spec off) | mbN (--max-batch N) | chunkN | bin=NAME (binary ~/intarsiaml-hdc/NAME instead of hdc-base-cli)
#   ACTION speed (localeval speed --sizes $SIZES --gen 128 --reps 3) | mmlu (localeval quality --bench mmlu --limit 25 --thinking off)
#          | texts (texts.py: md5 of 256 greedy forced tokens for the 18 gate prompts) | mixids (mixids.py: texts of A and B when B prefills beside A)
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}" "${ITEMS:?ITEMS=...}"
SIZES=${SIZES:-1024,4096,16384}; MS=${MS:-20480}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-27-hdc-hip-27b-decode/raw/$W; mkdir -p "$OUT" "$REPO/hdc-out"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO=$(mktemp -u "$REPO/hdc-out/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/hdc-bin' < /dev/null; scp -q "$HERE/guard.sh" tatooine:hdc-bin/
$SSH 'bash ~/hdc-bin/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stop_serve() { $SSH 'docker rm -f hdc-serve' >/dev/null 2>&1 < /dev/null; }
cleanup() { stop_serve; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

armenv() { # armenv <base arm> -> docker -e args
  case $1 in
    f32) echo "-e INTARSIA_TP_PARTIAL=f32" ;;
    bf16) echo "-e INTARSIA_TP_PARTIAL=bf16" ;;
    hyb) echo "-e INTARSIA_TP_PARTIAL=bf16-prefill" ;;
    rec) echo "" ;;
    *) local v="ENV_$1" e=""; for kv in ${!v:-}; do e="$e -e $kv"; done; echo "$e" ;;
  esac
}
serve() { # serve <arm+mods>
  local full=$1 parts; IFS='+' read -ra parts <<< "$full"; local arm=${parts[0]} bin=hdc-base-cli mb=1 extra="" m
  for m in "${parts[@]:1}"; do
    case $m in
      spoff) extra="$extra --spec off" ;;
      chunk*) extra="$extra --prefill-chunk ${m#chunk}" ;;
      mb*) mb=${m#mb} ;;
      bin=*) bin=${m#bin=} ;;
      *) echo "unknown mod $m" >&2; return 1 ;;
    esac
  done
  local envs; envs=$(armenv "$arm")
  local QENV="-e GPU_MAX_HW_QUEUES=1"; [[ "$envs" == *GPU_MAX_HW_QUEUES* ]] && QENV=""
  RECIPE="$bin serve tp2 mb$mb$(tr -d " " <<< "$extra") [$arm] (lane hdc)"  # short: localeval stats the --recipe string as a path
  stop_serve
  $SSH "docker run -d --name hdc-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp $QENV -e HIP_VISIBLE_DEVICES=0,1 $envs \
    -v \$HOME/intarsiaml-hdc:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model $M27 \
    --served-model-name qwen38-27b-hdc --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS$extra --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$full.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 hdc-serve' > "$OUT/serve-fail-$full.log" 2>&1 < /dev/null; return 1
}
tag() { echo "$1" | tr '+:=' '___'; }
cur=""; n=0
for item in $ITEMS; do
  n=$((n + 1)); full=${item%%:*}; act=${item#*:}; t="$n-$(tag "$full")-$act"
  if [[ $full != "$cur" && $act != attr && $act != gates ]]; then  # gates and kprof use the cards themselves
    say "item $n: starting server $full"
    if ! serve "$full"; then say "item $n $full: server did not come up"; cur=""; continue; fi
    $SSH 'docker logs hdc-serve 2>&1 | head -80; echo queues=$(docker exec hdc-serve printenv GPU_MAX_HW_QUEUES)' < /dev/null > "$OUT/serve-$(tag "$full")-$n.log" 2>&1
    printf '[targets.hdc-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen38-27b-hdc"\n' "$(tag "$full")" "$PORT" > "$OUT/targets-$(tag "$full").toml"
    cur=$full
  fi
  TG=$OUT/targets-$(tag "$full").toml; TN=hdc-$(tag "$full")
  case $act in
    speed)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval speed "$TN" --sizes "$SIZES" --gen 128 --reps 3 \
          --label "hdc-$W-$(tag "$full")-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)" ;;
    mmlu)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval quality "$TN" --bench mmlu --limit 25 --thinking off \
          --label "hdc-$W-$(tag "$full")-mmlu-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/mmlu-$t.log" 2>&1 < /dev/null
      say "item $n $full mmlu: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/mmlu-$t.log" | tail -1)" ;;
    texts)
      python3 "$HERE/texts.py" "http://tatooine:$PORT" qwen38-27b-hdc "$REPO/scripts/cmtp/ids-prompts" > "$OUT/texts-$t.txt" 2>&1
      say "item $n $full texts: exit $? $(wc -l < "$OUT/texts-$t.txt") lines" ;;
    mixids)
      python3 "$HERE/mixids.py" "http://tatooine:$PORT" qwen38-27b-hdc "$REPO/scripts/cmtp/ids-prompts/4k-0.txt" "$REPO/scripts/cmtp/ids-prompts/16k-3.txt" "$REPO/scripts/cmtp/ids-prompts/4k-3.txt" > "$OUT/mixids-$t.txt" 2>&1
      say "item $n $full mixids: $(tr '\n' '|' < "$OUT/mixids-$t.txt" | cut -c1-300)" ;;
    attr)    # attr.sh sections ASEC (plan op prof gen) on the free cards; logs and rocprof CSVs copied back
      stop_serve; cur=""
      $SSH "NOSTOP=1 W=$W SEC='${ASEC:-plan op prof gen}' NTOK=${NTOK:-130} bash ~/intarsiaml-hdc/scripts/hdc/attr.sh" < /dev/null > "$OUT/attr-$n.txt" 2>&1
      mkdir -p "$OUT/attr-$n"; scp -q -r "tatooine:intarsiaml-hdc-logs/$W/." "$OUT/attr-$n/" < /dev/null
      say "item $n attr: $(grep -a '^\[hdc\]\|^==' "$OUT/attr-$n.txt" | cut -c1-160 | tr '\n' '|')" ;;
    gates)   # gates.sh sections GSEC (fuse rows adm plan time) on the free cards
      stop_serve; cur=""
      $SSH "NOSTOP=1 W=$W SEC='${GSEC:-fuse rows adm plan time}' bash ~/intarsiaml-hdc/scripts/hdc/gates.sh" < /dev/null > "$OUT/gates-$n.txt" 2>&1
      mkdir -p "$OUT/gates-$n"; scp -q -r "tatooine:intarsiaml-hdc-logs/$W/." "$OUT/gates-$n/" < /dev/null
      say "item $n gates: $(grep -a '^==' "$OUT/gates-$n.txt" | cut -c1-150 | tr '\n' '|')" ;;
    slog)    # the running server's stderr (host-profile lines)
      $SSH 'docker logs hdc-serve 2>&1' < /dev/null > "$OUT/slog-$t.txt" 2>&1
      say "item $n slog: $(grep -ac host-profile "$OUT/slog-$t.txt") host-profile lines" ;;
    *) say "item $n: unknown action $act" ;;
  esac
done
say "done"
