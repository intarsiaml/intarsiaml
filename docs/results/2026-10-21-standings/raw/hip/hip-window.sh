#!/usr/bin/env bash
# 2026-10-21 standings: R9700 (tatooine, ONE card, PCI 03:00.0 = HIP device 0) intarsia HIP vs the engines already
# installed, Qwen3.6-35B-A3B. Runs on the Mac (it needs /opt/code/deepeval) as ONE fleetq command, so the trap restores
# radiance if this dies:
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "standings hip window" -- \
#     env OUT=<dir> bash scripts/standings/hip-window.sh
# Arms (each: start a server on GPU 0, `localeval run` = speed sweep (+ mmlu/gsm8k n=25 quick tier), stop the server):
#   intarsia          docker image intarsia-hip-dev, binary from ~/intarsiaml-standings (CPU slot build beforehand)
#   llamacpp-hip      llama.cpp v0.5.0 HIP build (~/bench-r9700/...-hip) run inside the intarsia-hip-dev image (ROCm libs)
#   llamacpp-vk       ~/.local/opt/llama.cpp (Vulkan, RADV) on MXFP4_MOE and UD-Q4_K_XL GGUF
# vLLM-ROCm is not installed (retired 2026-10-11) and radiance serves only the 27B, so neither is an arm.
# Window cap: WINDOW_S of radiance downtime INCLUDING the restart poll (<= 180 s); later arms are skipped when the
# remaining time is short and the skip is logged in $OUT/downtime.log.
set -uo pipefail
: "${OUT:?set OUT to the raw output dir}"
mkdir -p "$OUT"
T=tatooine
WINDOW_S=${WINDOW_S:-1150}
LOG=$OUT/downtime.log
LE=/opt/code/deepeval
SIZES=${SIZES:-128,1024,4096,16384}
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
capped() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }

IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
REPO=/home/josef/intarsiaml-standings
VOL=intarsia-cargo-target-standings
LVK=/home/josef/.local/opt/llama.cpp
LHIPBIN=/home/josef/bench-r9700/llama.cpp-v0.5.0-hip/build/bin
GG=/data/models/gguf/qwen3.6-35b-a3b
MXFP4=$GG/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
Q4KXL=$GG/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

# ---- preflight: everything that can fail cheaply fails BEFORE radiance is stopped ----
pf_fail=0
pf() { ssh $T "$1" >>"$OUT/preflight.out" 2>&1 || { echo "PREFLIGHT FAIL: $1" | tee -a "$OUT/preflight.out"; pf_fail=1; }; }
: >"$OUT/preflight.out"
pf "docker run --rm -v $VOL:/target:ro $IMG test -x /target/release/intarsia"
pf "test -d /data/models/intarsia/Qwen3.6-35B-A3B-NVFP4 && test -f $MXFP4 && test -f $Q4KXL"
pf "$LVK/llama-server --version"
pf "test -x $LHIPBIN/llama-server"
HIP_OK=1
ssh $T "docker run --rm -v $LHIPBIN:/lcpp:ro $IMG bash -c 'ldd /lcpp/libggml-hip.so.0 | grep -c \"not found\"' | grep -qx 0" >>"$OUT/preflight.out" 2>&1 || {
  HIP_OK=0; echo "preflight: llama.cpp HIP build has unresolved libs in the dev image, ROCm arm will be skipped" | tee -a "$OUT/preflight.out"; }
ssh $T 'systemctl --user is-active radiance; lspci | grep -c 7551' >>"$OUT/preflight.out" 2>&1
if [ $pf_fail = 1 ]; then echo "preflight failed, radiance untouched"; exit 3; fi
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then echo "preflight ok (HIP_OK=$HIP_OK)"; exit 0; fi

t0=$(date +%s)
left() { echo $(( WINDOW_S - ($(date +%s) - t0) )); }

stop_servers() {
  ssh $T 'docker rm -f intarsia-serve llamacpp-hip-serve >/dev/null 2>&1; [ -f /tmp/standings-llama.pid ] && kill $(cat /tmp/standings-llama.pid) 2>/dev/null; sleep 1; [ -f /tmp/standings-llama.pid ] && kill -9 $(cat /tmp/standings-llama.pid) 2>/dev/null; rm -f /tmp/standings-llama.pid; true' >/dev/null 2>&1
}
restore() {
  stop_servers
  say "starting radiance"
  ssh $T 'systemctl --user start radiance'
  local i
  for i in $(seq 1 36); do
    if ssh $T 'curl -s -m 3 http://localhost:8080/v1/models' | grep -q '"data":\['; then
      say "radiance answers (after $((i * 5)) s); window total $(( $(date +%s) - t0 )) s"
      return 0
    fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 180 s"
  return 1
}

wait_up() { # wait_up <url> <secs>
  local i
  for ((i = 0; i < $2; i += 3)); do
    curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0
    sleep 3
  done
  return 1
}

# arm <label> <model-id> <port> <min-seconds-needed> <quality 1|0> <start-cmd run on tatooine>
arm() {
  local label=$1 mid=$2 port=$3 need=$4 q=$5 cmd=$6
  if (( $(left) < need )); then say "SKIPPED arm $label: $(left) s left, needs $need"; return 0; fi
  say "arm $label: starting server"
  stop_servers
  ssh $T "$cmd" >"$OUT/$label-server-start.out" 2>&1
  if ! wait_up "http://$T:$port/health" 240; then
    say "arm $label: server did not come up"
    ssh $T 'docker logs --tail 30 intarsia-serve 2>&1; docker logs --tail 30 llamacpp-hip-serve 2>&1; tail -30 /tmp/standings-llama.log 2>&1' >"$OUT/$label-server.err" 2>&1
    stop_servers
    return 1
  fi
  say "arm $label: server up ($(left) s left)"
  local qargs=()
  [ "$q" = 1 ] && qargs=(--bench mmlu,gsm8k --limit 25 --judge none)
  local cap=$(( $(left) - 40 )); (( cap > 600 )) && cap=600
  (cd $LE && capped "$cap" uv run localeval run --url "http://$T:$port/v1" --model "$mid" \
    --sizes "$SIZES" --gen 128 --reps 3 --thinking off "${qargs[@]}" --label "standings-$label") >"$OUT/$label-localeval.out" 2>&1
  say "arm $label: localeval exit $? ($(left) s left)"
  curl -s "http://$T:$port/metrics" 2>/dev/null | grep -iE "graph|eager" | head >"$OUT/$label-metrics.txt" 2>&1
  stop_servers
  sleep 3
}

say "job ${FLEETQ_JOB:-none}; window cap ${WINDOW_S} s; stopping radiance"
trap restore EXIT
ssh $T 'systemctl --user stop radiance'
sleep 2
say "radiance stopped"
ssh $T "rocm-smi --showmeminfo vram 2>/dev/null | head -12; $LVK/llama-server --version 2>&1 | head -3" >"$OUT/versions-tatooine.txt" 2>&1

DEV="--device /dev/kfd --device /dev/dri --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp"
INTARSIA_CMD="docker run -d --name intarsia-serve --network host $DEV -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0 \
 -v $REPO:/work -v /data/models/intarsia:/models:ro -v $VOL:/target -w /work $IMG \
 /target/release/intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --profile-every 0 --mem-budget 29 \
 --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-intarsia-hip --max-seq 17408"
LARGS="-ngl 999 -fa on -c 20480 --cache-ram 0 -np 1 --host 0.0.0.0 --port 8091"
llama_hip_cmd() { # <gguf> <alias>
  echo "docker run -d --name llamacpp-hip-serve --network host $DEV -e HIP_VISIBLE_DEVICES=0 -v $LHIPBIN:/lcpp:ro -v $GG:/gguf:ro $IMG /lcpp/llama-server -m /gguf/$(basename "$1") --alias $2 $LARGS"
}
llama_vk_cmd() { # <gguf> <alias>
  echo "GGML_VK_VISIBLE_DEVICES=0 nohup $LVK/llama-server -m $1 --alias $2 $LARGS >/tmp/standings-llama.log 2>&1 </dev/null & echo \$! >/tmp/standings-llama.pid"
}

arm intarsia-35b-nvfp4 qwen36-35b-intarsia-hip 8090 420 1 "$INTARSIA_CMD"
[ $HIP_OK = 1 ] && arm llamacpp-hip-35b-mxfp4moe MXFP4_MOE 8091 360 1 "$(llama_hip_cmd $MXFP4 MXFP4_MOE)"
arm llamacpp-vk-35b-mxfp4moe MXFP4_MOE 8091 360 1 "$(llama_vk_cmd $MXFP4 MXFP4_MOE)"
arm llamacpp-vk-35b-q4kxl UD-Q4_K_XL 8091 240 0 "$(llama_vk_cmd $Q4KXL UD-Q4_K_XL)"
say "arms done"
