#!/usr/bin/env bash
# Lane lmhead-fp8: HIP window driver (Qwen3.8-27B MXFP4 on tatooine's two R9700s), run FROM THE MAC inside ONE fleetq slot that takes radiance down:
#   J=$(fleetq acquire --host tatooine --takes-down radiance --mem 30 --quiet gpu,cpu --expect 75 --label "..." --wait)
#   FLEETQ_JOB=$J W=w1 ITEMS="..." scripts/lmhead/hdrive.sh ; fleetq release $J      (radiance is restored by the EXIT trap; a deadman restarts it after DEADMAN s)
# ITEMS = space separated "BIN.Q.TP.SPEC.MB:ACTION", run in order; a fresh `serve` starts whenever the part before ":" changes.
#   BIN   head (this branch, ~/lmhead-bin/intarsia) | base (main 5390e65b4, ~/lmhbase-bin/intarsia); head35 / base35: the same binaries on the Qwen3.6-35B-A3B NVFP4
#   Q     off (variable unset) | fp8 | mxfp4  = INTARSIA_LMHEAD_QUANT
#   TP    1 | 2     SPEC  off (--spec off) | mtp (tp2: the recipe's default k2; tp1: --spec mtp --spec-k 2)     MB  --max-batch
#   ACTION speed (localeval speed 1k/4k/16k, gen 128, REPS reps) | speed16 (1k prompt, 1/4/8/16 streams) | texts (md5 of 256 greedy tokens, 18 prompts)
#          | mmlu (10 per task) | gsm8k (100) | plan (the server's startup log) | quality (accuracy KL vs the bf16 head on $OUT/prompts.safetensors, which `gen` writes; tp from the item) | gen
# Steps and results: $OUT/steps.log ("item N <arm> speed: exit 0 results/<run>" lines feed scripts/hdc/speedtab2.py).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq (see the header)}" "${W:?W=<window name>}" "${ITEMS:?ITEMS=...}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-11-05-lmhead-fp8-27b/raw/hip-$W}; mkdir -p "$OUT"
LE=/opt/code/deepeval; PORT=8297; REPS=${REPS:-3}; MS=${MS:-20480}; DEADMAN=${DEADMAN:-4800}
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
SSH="ssh -n -o BatchMode=yes tatooine"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
stop_serve() { $SSH 'docker rm -f lmh-serve' >/dev/null 2>&1; }
restore() {
  stop_serve; $SSH 'docker ps -q --filter name=lmh-run- | xargs -r docker kill >/dev/null 2>&1'
  $SSH "[ -f ~/lmh-deadman.pid ] && kill \$(cat ~/lmh-deadman.pid) 2>/dev/null; rm -f ~/lmh-deadman.pid"
  say "starting radiance"
  $SSH "FLEETQ_JOB=$FLEETQ_JOB systemctl --user start radiance"
  for i in $(seq 1 48); do
    if $SSH "curl -s -m 3 http://localhost:8080/v1/models" | grep -q '"data":\['; then say "radiance answers (after $((i * 5)) s): $($SSH 'curl -s -m 3 http://tatooine:8080/v1/models' | head -c 120)"; return 0; fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 240 s"; return 1
}
trap restore EXIT
trap 'exit 130' INT TERM
$SSH "nohup bash -c 'sleep $DEADMAN; systemctl --user start radiance' >/dev/null 2>&1 & echo \$! > ~/lmh-deadman.pid"
say "job $FLEETQ_JOB window $W: stopping radiance"
$SSH "FLEETQ_JOB=$FLEETQ_JOB systemctl --user stop radiance" && sleep 3
START=$SECONDS; CAP=${CAP:-3900}   # no new item starts after CAP s (75 min limit minus the last item and the restore)

serve() { $SSH 'docker rm -f lmh-serve' >/dev/null 2>&1; }
restore() {
  stop_serve; $SSH 'docker ps -q --filter name=lmh-run- | xargs -r docker kill >/dev/null 2>&1'
  $SSH "[ -f ~/lmh-deadman.pid ] && kill \$(cat ~/lmh-deadman.pid) 2>/dev/null; rm -f ~/lmh-deadman.pid"
  say "starting radiance"
  $SSH "FLEETQ_JOB=$FLEETQ_JOB systemctl --user start radiance"
  for i in $(seq 1 48); do
    if $SSH "curl -s -m 3 http://localhost:8080/v1/models" | grep -q '"data":\['; then say "radiance answers (after $((i * 5)) s): $($SSH 'curl -s -m 3 http://tatooine:8080/v1/models' | head -c 120)"; return 0; fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 240 s"; return 1
}
trap restore EXIT
trap 'exit 130' INT TERM
$SSH "nohup bash -c 'sleep $DEADMAN; systemctl --user start radiance' >/dev/null 2>&1 & echo \$! > ~/lmh-deadman.pid"
say "job $FLEETQ_JOB window $W: stopping radiance"
$SSH "FLEETQ_JOB=$FLEETQ_JOB systemctl --user stop radiance" && sleep 3
START=$SECONDS; CAP=${CAP:-3900}   # no new item starts after CAP s (75 min limit minus the last item and the restore)

dock() { # dock <gpus> <extra docker args> -- <cmd...>   one container; the two binaries are mounted
  local g=$1 name=lmh-run-$RANDOM; shift
  echo "docker run --name $name --network host --device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$g"
}
serve() { # serve <item head>
  IFS=. read -r bin q tp spec mb <<< "$1"
  local MD=$M27; [[ $bin == *35 ]] && { MD=/models/intarsia/Qwen3.6-35B-A3B-NVFP4; bin=${bin%35}; }   # head35 / base35: the Qwen3.6-35B-A3B NVFP4, default flags
  local b=lmhead-bin; [[ $bin == base ]] && b=lmhbase-bin
  local envs=""; [[ $q != off ]] && envs="-e INTARSIA_LMHEAD_QUANT=$q"
  local gpus=0 extra="--max-batch $mb"
  [[ $tp == 2 ]] && { gpus=0,1; extra="$extra --tp 2 --devices 0,1"; }
  case $spec in off) extra="$extra --spec off" ;; mtp) [[ $tp == 1 ]] && extra="$extra --spec mtp --spec-k 2" ;; esac
  RECIPE="intarsia serve (lane lmhead-fp8) $bin q=$q tp=$tp spec=$spec mb=$mb"
  stop_serve
  $SSH "docker run -d --name lmh-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $envs \
    -v \$HOME/$b:/b:ro -v /data/models:/models:ro $IMG /b/intarsia serve --backend hip --model $MD --served-model-name qwen38-27b \
    --profile-every 0 --mem-budget 29 --max-seq $MS $extra --no-supervisor --host 0.0.0.0 --port $PORT" > "$OUT/docker-$1.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 60 lmh-serve' > "$OUT/serve-fail-$1.log" 2>&1; return 1
}
cur=""; n=0
for item in $ITEMS; do
  n=$((n + 1)); full=${item%%:*}; act=${item#*:}; t="$n-$full-$act"
  (( SECONDS - START > CAP )) && { say "cap $CAP s reached: skipping item $n ($item) and the rest"; break; }
  if [[ $act == gates ]]; then
    stop_serve; cur=""
    say "item $n gates: start"
    ssh -o BatchMode=yes tatooine "bash -s" < "$HERE/hgates.sh" > "$OUT/gates.txt" 2>&1
    say "item $n gates: exit $? $(grep -a '^GATES' "$OUT/gates.txt" | cut -c1-120 | tr '\n' '|')"
    mkdir -p "$OUT/gates"; scp -q -r "tatooine:lmh-gates/." "$OUT/gates/" < /dev/null
    continue
  fi
  if [[ $act == quality ]]; then
    stop_serve; cur=""
    say "item $n $full quality: start"
    IFS=. read -r bin q tp spec mb <<< "$full"
    GOLDEN=${GOLDEN:-$OUT/prompts.safetensors}
    [[ -s ${GOLDEN:-/nonexistent} ]] || { say "item $n quality: no GOLDEN (prompts.safetensors from scripts/lmhead/mquality.sh)"; continue; }
    $SSH 'mkdir -p ~/lmh-golden' && scp -q "$GOLDEN" tatooine:lmh-golden/prompts.safetensors
    ssh -o BatchMode=yes tatooine "TP=$tp BIN=$bin DEC=${DEC:-128} ARMS='${QARMS:-fp8 mxfp4 off}' SKIPWRITE=${SKIPWRITE:-0} bash -s" < "$HERE/hquality.sh" > "$OUT/quality-$full.txt" 2>&1
    say "item $n $full quality: exit $? $(grep -a '^QUALITY' "$OUT/quality-$full.txt" | tail -3 | tr '\n' '|')"
    continue
  fi
  if [[ $full != "$cur" ]]; then
    say "item $n: starting server $full"
    if ! serve "$full"; then say "item $n $full: server did not come up"; cur=""; continue; fi
    $SSH 'docker logs lmh-serve 2>&1 | head -120' > "$OUT/serve-$full-$n.log" 2>&1
    grep -aE "lm_head quantised|spec:|vocab-parallel" "$OUT/serve-$full-$n.log" | head -4 | tee -a "$OUT/steps.log"
    cur=$full
  fi
  U=http://tatooine:$PORT/v1
  case $act in
    speed)
      ( cd $LE && uv run localeval speed --url $U --model qwen38-27b --thinking off --sizes 1024,4096,16384 --gen 128 --reps $REPS \
          --label "lmh-$W-$full-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)" ;;
    speed16)
      ( cd $LE && uv run localeval speed --url $U --model qwen38-27b --thinking off --sizes 1024 --gen 128 --reps ${REPS16:-2} --concurrency 1,4,8,16 \
          --label "lmh-$W-$full-$n" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/speed-$t.log" 2>&1 < /dev/null
      say "item $n $full speed16: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/speed-$t.log" | tail -1)"
      grep -aE 'streams: aggregate' "$OUT/speed-$t.log" | tee -a "$OUT/steps.log" ;;
    texts)
      python3 "$REPO/scripts/hdc/texts.py" "http://tatooine:$PORT" qwen38-27b "$REPO/scripts/cmtp/ids-prompts" > "$OUT/texts-$t.txt" 2>&1
      say "item $n $full texts: exit $? $(wc -l < "$OUT/texts-$t.txt") lines" ;;
    mmlu)
      ( cd $LE && uv run localeval quality --url $U --model qwen38-27b --thinking off --bench mmlu --limit 10 \
          --label "lmh-$W-$full-mmlu-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/mmlu-$t.log" 2>&1 < /dev/null
      say "item $n $full mmlu: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/mmlu-$t.log" | tail -1)" ;;
    gsm8k)
      ( cd $LE && uv run localeval quality --url $U --model qwen38-27b --thinking off --bench gsm8k --limit 100 \
          --label "lmh-$W-$full-gsm8k-$n" --recipe "$RECIPE" --wait-idle 600 ) > "$OUT/gsm8k-$t.log" 2>&1 < /dev/null
      say "item $n $full gsm8k: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/gsm8k-$t.log" | tail -1)" ;;
    gen)   # 128 greedy tokens per prompt from the running (bf16-head) server, then the prompts-only golden (prompt + generated text) on the Mac
      python3 "$HERE/hgen.py" "http://tatooine:$PORT" qwen38-27b "$REPO/scripts/cmtp/ids-prompts" "$OUT/gen" ${GENNAMES:-4k-0 4k-1 4k-2 16k-0 16k-1 16k-2} > "$OUT/gen-$t.txt" 2>&1
      ( cd $REPO && uv run --with tokenizers --with numpy --with safetensors python scripts/lmhead/mkgolden.py ${MODEL_LOCAL:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4} "$OUT/prompts.safetensors" \
          $(for g in ${GENNAMES:-4k-0 4k-1 4k-2 16k-0 16k-1 16k-2}; do echo "$g=$REPO/scripts/cmtp/ids-prompts/$g.txt:$OUT/gen/$g.txt"; done) ) >> "$OUT/gen-$t.txt" 2>&1
      say "item $n $full gen: exit $? $(tail -1 "$OUT/gen-$t.txt" | cut -c1-200)" ;;
    plan) say "item $n $full plan: see serve-$full-$n.log" ;;
    *) say "item $n: unknown action $act" ;;
  esac
done
say "done"
