#!/usr/bin/env bash
# Lane PF27 window driver, run FROM THE MAC inside ONE fleetq job that holds both R9700s and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 40 \
#     --label "tpb ..." -- env W=wa ITEMS="..." scripts/pf27/drive.sh
# ITEMS = space separated "ARM[+mod...]:ACTION", run in order; a fresh `serve` starts whenever ARM+mods changes.
#   ARM   rec (no env: the recipe decides) | NAME: envs from ENV_NAME (e.g. ENV_old="INTARSIA_HIP_MXFP4_A8_PD=1")
#   mods  spoff (--spec off) | mbN (--max-batch N) | chunkN | bin=NAME (binary ~/intarsiaml-pf27/NAME instead of pf27-base-cli)
#   ACTION speed (localeval speed --sizes $SIZES --gen 128 --reps 3) | mmlu (localeval quality --bench mmlu --limit 25 --thinking off)
#          | texts (texts.py: md5 of 256 greedy forced tokens for the 18 gate prompts) | mixids (mixids.py: texts of A and B when B prefills beside A)
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}" "${ITEMS:?ITEMS=...}"
SIZES=${SIZES:-1024,4096,16384}; MS=${MS:-20480}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-28-pf27-hip-27b-prefill/raw/$W; mkdir -p "$OUT" "$REPO/pf27-out"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO=$(mktemp -u "$REPO/pf27-out/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/intarsiaml-pf27/pf27-bin' < /dev/null; scp -q "$HERE/guard.sh" tatooine:intarsiaml-pf27/pf27-bin/
$SSH 'bash ~/intarsiaml-pf27/pf27-bin/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
stop_serve() { $SSH 'docker rm -f pf27-serve' >/dev/null 2>&1 < /dev/null; }
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
  local full=$1 parts; IFS='+' read -ra parts <<< "$full"; local arm=${parts[0]} bin=pf27-base-cli mb=1 extra="" m
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
  RECIPE="HIP_VISIBLE_DEVICES=0,1 $bin serve --backend hip --model 27B --tp 2 --devices 0,1 --max-batch $mb --mem-budget 29 --max-seq $MS$extra ($envs; lane pf27)"
  stop_serve
  $SSH "docker run -d --name pf27-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp $QENV -e HIP_VISIBLE_DEVICES=0,1 $envs \
    -v \$HOME/intarsiaml-pf27:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model $M27 \
    --served-model-name qwen38-27b-pf27 --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS$extra --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$full.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 pf27-serve' > "$OUT/serve-fail-$full.log" 2>&1 < /dev/null; return 1
}
tag() { echo "$1" | tr '+:=' '___'; }
cur=""; n=0
for item in $ITEMS; do
  n=$((n + 1)); full=${item%%:*}; act=${item#*:}; t="$n-$(tag "$full")-$act"
  if [[ $full != "$cur" ]]; then
    say "item $n: starting server $full"
    if ! serve "$full"; then say "item $n $full: server did not come up"; cur=""; continue; fi
    $SSH 'docker logs pf27-serve 2>&1 | head -80; echo queues=$(docker exec pf27-serve printenv GPU_MAX_HW_QUEUES)' < /dev/null > "$OUT/serve-$(tag "$full")-$n.log" 2>&1
    printf '[targets.pf27-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen38-27b-pf27"\n' "$(tag "$full")" "$PORT" > "$OUT/targets-$(tag "$full").toml"
    cur=$full
  fi
  TG=$OUT/targets-$(tag "$full").toml; TN=pf27-$(tag "$full")
  case $act in
    speed)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval speed "$TN" --sizes "$SIZES" --gen 128 --reps 3 \
          --label "pf27-$W-$(tag "$full")-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)" ;;
    mmlu)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$TG" uv run localeval quality "$TN" --bench mmlu --limit 25 --thinking off \
          --label "pf27-$W-$(tag "$full")-mmlu-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/mmlu-$t.log" 2>&1 < /dev/null
      say "item $n $full mmlu: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/mmlu-$t.log" | tail -1)" ;;
    texts)
      python3 "$HERE/texts.py" "http://tatooine:$PORT" qwen38-27b-pf27 "$REPO/scripts/cmtp/ids-prompts" > "$OUT/texts-$t.txt" 2>&1
      say "item $n $full texts: exit $? $(wc -l < "$OUT/texts-$t.txt") lines" ;;
    mixids)
      python3 "$HERE/mixids.py" "http://tatooine:$PORT" qwen38-27b-pf27 "$REPO/scripts/cmtp/ids-prompts/4k-0.txt" "$REPO/scripts/cmtp/ids-prompts/16k-3.txt" "$REPO/scripts/cmtp/ids-prompts/4k-3.txt" > "$OUT/mixids-$t.txt" 2>&1
      say "item $n $full mixids: $(tr '\n' '|' < "$OUT/mixids-$t.txt" | cut -c1-300)" ;;
    kprof)   # rocprof counter passes on the a8 GEMM test binary (w1.sh SEC=prof): PROFENV = extra env of the call, e.g. CANDS / GROUP / PVARS
      stop_serve; cur=""
      $SSH "NOSTOP=1 BIN=/work/pf27-bin/mxfp4_a8-dev W=$W-kprof SEC='list prof' $PROFENV bash ~/intarsiaml-pf27/scripts/pf27/w1.sh" < /dev/null > "$OUT/kprof-$n.txt" 2>&1
      scp -q -r "tatooine:intarsiaml-pf27-logs/$W-kprof/." "$OUT/kprof-$n/" < /dev/null
      say "item $n kprof: $(grep -ac 'pass' "$OUT/kprof-$n.txt") passes" ;;
    gates)   # w1.sh sections GSEC (corr hash mut adm plan time ...) with GENV extra env (TVARS=...); cards free
      stop_serve; cur=""
      $SSH "NOSTOP=1 BIN=/work/pf27-bin/mxfp4_a8-dev W=$W-gates SEC='$GSEC' $GENV bash ~/intarsiaml-pf27/scripts/pf27/w1.sh" < /dev/null > "$OUT/gates-$n.txt" 2>&1
      scp -q -r "tatooine:intarsiaml-pf27-logs/$W-gates/." "$OUT/gates-$n/" < /dev/null
      say "item $n gates: $(grep -a '^==' "$OUT/gates-$n.txt" | cut -c1-150 | tr '\n' '|')" ;;
    *) say "item $n: unknown action $act" ;;
  esac
done
say "done"
