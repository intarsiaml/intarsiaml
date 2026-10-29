#!/usr/bin/env bash
# Lane NQ27 (from ARX) window driver (copy of scripts/pf27/drive.sh), run FROM THE MAC inside ONE fleetq job that holds both R9700s and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 40 \
#     --label "tpb ..." -- env W=wa ITEMS="..." scripts/pf27/drive.sh
# ITEMS = space separated "ARM[+mod...]:ACTION", run in order; a fresh `serve` starts whenever ARM+mods changes.
#   ARM   rec (no env: the recipe decides) | NAME: envs from ENV_NAME (e.g. ENV_old="INTARSIA_HIP_MXFP4_A8_PD=1")
#   mods  spoff (--spec off) | mbN (--max-batch N) | chunkN | bin=NAME (binary ~/intarsiaml-nq27/NAME instead of nq27-base-cli)
#         | m35 (Qwen3.6-35B-A3B NVFP4 instead of the 27B)
#   ACTION speed (localeval speed --sizes $SIZES --gen 128 --reps 3) | mmlu (localeval quality --bench mmlu --limit 25 --thinking off)
#          | texts (texts.py: md5 of 256 greedy forced tokens for the 18 gate prompts) | mixids (mixids.py: texts of A and B when B prefills beside A)
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}" "${ITEMS:?ITEMS=...}"
SIZES=${SIZES:-1024,4096,16384}; MS=${MS:-20480}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
M35=/models/intarsia/Qwen3.6-35B-A3B-NVFP4
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-29-nq27-quant-handoff/raw/$W; mkdir -p "$OUT" "$REPO/.lane"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
scp -q "$HERE"/*.sh tatooine:intarsiaml-nq27/scripts/nq27/
for f in ${BINS:-nq27-base-cli nq27-dev-cli}; do
  $SSH "test -x ~/intarsiaml-nq27/$f" < /dev/null || { echo "MISSING $f: not starting the window"; exit 3; }
done
FIFO=$(mktemp -u "$REPO/.lane/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/intarsiaml-nq27/nq27-bin' < /dev/null; scp -q "$HERE/guard.sh" tatooine:intarsiaml-nq27/nq27-bin/
$SSH 'bash ~/intarsiaml-nq27/nq27-bin/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stop_serve() { $SSH 'docker rm -f nq27-serve' >/dev/null 2>&1 < /dev/null; }
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
  local full=$1 parts; IFS='+' read -ra parts <<< "$full"; local arm=${parts[0]} bin=nq27-base-cli mb=1 extra="" m mp=$M27 mt=27B
  for m in "${parts[@]:1}"; do
    case $m in
      spoff) extra="$extra --spec off" ;;
      chunk*) extra="$extra --prefill-chunk ${m#chunk}" ;;
      mb*) mb=${m#mb} ;;
      bin=*) bin=${m#bin=} ;;
      m35) mp=$M35; mt=35B ;;
      r*) ;;  # a tag only: a new tag starts a fresh server
      *) echo "unknown mod $m" >&2; return 1 ;;
    esac
  done
  local envs; envs=$(armenv "$arm")
  local QENV="-e GPU_MAX_HW_QUEUES=1"; [[ "$envs" == *GPU_MAX_HW_QUEUES* ]] && QENV=""
  RECIPE="HIP_VISIBLE_DEVICES=0,1 $bin serve --backend hip --model $mt --tp 2 --devices 0,1 --max-batch $mb --mem-budget 29 --max-seq $MS$extra ($envs; lane arx)"
  stop_serve
  $SSH "docker run -d --name nq27-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp $QENV -e HIP_VISIBLE_DEVICES=0,1 $envs \
    -v \$HOME/intarsiaml-nq27:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model $mp \
    --served-model-name qwen-nq27 --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS$extra --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$full.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 nq27-serve' > "$OUT/serve-fail-$full.log" 2>&1 < /dev/null; return 1
}
tag() { echo "$1" | tr '+:=' '___'; }
cur=""; n=0; T0=$(date +%s)
for item in $ITEMS; do
  n=$((n + 1)); full=${item%%:*}; act=${item#*:}; t="$n-$(tag "$full")-$act"
  # lane arx: the window cap (DEADLINE s after the start, default 41 min): later items are skipped
  if (( $(date +%s) - T0 > ${DEADLINE:-2460} )); then say "item $n $full:$act: skipped (window cap)"; continue; fi
  if [[ $full != "$cur" ]]; then
    say "item $n: starting server $full"
    if ! serve "$full"; then say "item $n $full: server did not come up"; cur=""; continue; fi
    $SSH 'docker logs nq27-serve 2>&1 | head -80; echo queues=$(docker exec nq27-serve printenv GPU_MAX_HW_QUEUES)' < /dev/null > "$OUT/serve-$(tag "$full")-$n.log" 2>&1
    printf '[targets.nq27-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen-nq27"\n' "$(tag "$full")" "$PORT" > "$OUT/targets-$(tag "$full").toml"
    cur=$full
  fi
  TG=$OUT/targets-$(tag "$full").toml; TN=nq27-$(tag "$full")
  case $act in
    speed)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run python "$HERE/leval.py" speed "$TN" --sizes "$SIZES" --gen 128 --reps 3 \
          --label "nq27-$W-$(tag "$full")-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)" ;;
    conc)  # lane arx: decode with CONC parallel streams (the server needs mbN >= the largest), 1k prompt
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run python "$HERE/leval.py" speed "$TN" --sizes 1024 --gen 128 --reps 3 --concurrency "${CONC:-4,8,16}" \
          --label "nq27-$W-$(tag "$full")-conc-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/conc-$t.log" 2>&1 < /dev/null
      say "item $n $full conc: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/conc-$t.log" | tail -1)" ;;
    mmlu)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval quality "$TN" --bench mmlu --limit 25 --thinking off \
          --label "nq27-$W-$(tag "$full")-mmlu-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/mmlu-$t.log" 2>&1 < /dev/null
      say "item $n $full mmlu: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/mmlu-$t.log" | tail -1)" ;;
    p4107)  # lane arx2: TTFT of localeval's 4k filler prompt at 182 sentences (4107 tokens) and 181 (4084), P4107N reps each
      ( cd /opt/code/deepeval && uv run python "$HERE/p4107.py" "http://tatooine:$PORT/v1" qwen-nq27 "${P4107N:-3}" ) > "$OUT/p4107-$t.txt" 2>&1 < /dev/null
      say "item $n $full p4107: $(grep -a '^p4107' "$OUT/p4107-$t.txt" | cut -c1-200 | tr '\n' '|')" ;;
    texts)
      python3 "$HERE/../pf27/texts.py" "http://tatooine:$PORT" qwen-nq27 "$REPO/scripts/cmtp/ids-prompts" > "$OUT/texts-$t.txt" 2>&1
      say "item $n $full texts: exit $? $(wc -l < "$OUT/texts-$t.txt") lines" ;;
    texts2)  # lane arx: texts.py on TNAMES only (e.g. "4k-0 16k-0")
      python3 "$HERE/../pf27/texts.py" "http://tatooine:$PORT" qwen-nq27 "$REPO/scripts/cmtp/ids-prompts" ${TNAMES:-4k-0 16k-0} > "$OUT/texts2-$t.txt" 2>&1
      say "item $n $full texts2: $(tr '\n' '|' < "$OUT/texts2-$t.txt" | cut -c1-200)" ;;
    mixids)
      python3 "$HERE/../pf27/mixids.py" "http://tatooine:$PORT" qwen-nq27 $(for f in ${MIXF:-4k-0 16k-3 4k-3}; do echo "$REPO/scripts/cmtp/ids-prompts/$f.txt"; done) > "$OUT/mixids-$t.txt" 2>&1
      say "item $n $full mixids: $(tr '\n' '|' < "$OUT/mixids-$t.txt" | cut -c1-300)" ;;
    gates)   # micro.sh sections GSEC with GENV extra env, cards free (the server stopped), radiance stays down (NORAD=1)
      stop_serve; cur=""
      $SSH "NORAD=1 W=$W-gates SEC='$GSEC' ${GENV:-} bash ~/intarsiaml-nq27/scripts/nq27/micro.sh" < /dev/null > "$OUT/gates-$n.txt" 2>&1
      mkdir -p "$OUT/gates-$n"; scp -q -r "tatooine:intarsiaml-nq27/nq27-out/$W-gates/." "$OUT/gates-$n/" < /dev/null
      say "item $n gates: $(grep -a '^==' "$OUT/gates-$n.txt" | cut -c1-200 | tr '\n' '|')" ;;
    *) say "item $n: unknown action $act" ;;
  esac
done
say "done"
