#!/usr/bin/env bash
# Lane dp: Qwen3.6-35B-A3B-NVFP4 as ONE tp2 server (arm T) against TWO tp1 servers, one per R9700 (arm D), on tatooine. Runs on endor INSIDE a
# fleetq slot that takes radiance down (the two cards are radiance's), e.g.
#   fleetq run --host tatooine --takes-down radiance --mem 30 --exclusive --expect 60 --label "..." -- scripts/dp/window.sh <outdir>
# Needs ~/intarsiaml-dp/dp-intarsia (scripts/dp/build.sh). Protocol: docs/results/2026-11-06-hip35-two-tp1.md.
# Visits (ABAB, a fresh server or servers per visit): T1 D1 T2 D2. Per visit: identity prompts, then load.py at 1 / 8 / 16 / 32 total streams
# (warm-up burst + 3 bursts each; D splits N evenly over its two servers), peak VRAM per card sampled from sysfs every second.
# D1 also runs the CPU-starvation check (16 streams on server 0 alone, then 32 on both with docker stats / top sampled).
# Then the max-seq probe on one card (32768, then 65536 if that works; stepping down if not), then, if time allows, one extra T visit at
# --max-batch 32 (outside the decision rule). No deadman timers: radiance is restored by the EXIT / HUP / INT / TERM trap on every path.
# DRY=1 DRY_TAIL=--help: the same ssh / docker steps with radiance untouched and no model load (a CPU-slot check).
# Knobs: VISITS="T1 D1 T2 D2"  PROBE=0 (skip the max-seq probe)  EXTRA=0 (skip the max-batch 32 visit)  CAP=3000 (s after which no extra starts).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq (see the header)}"
OUT=${1:?outdir}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolute: the driver and the logs write here
H=$(cd "$(dirname "$0")" && pwd)
R='$HOME/intarsiaml-dp'   # expanded by the remote shell
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
PT=8300; P0=8301; P1=8302
UT=http://tatooine:$PT/v1; U0=http://tatooine:$P0/v1; U1=http://tatooine:$P1/v1
CAP=${CAP:-3000}
DRYM=-d; [[ ${DRY:-0} == 1 ]] && DRYM=--rm
SAMPLER=""
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
rsh() { ssh -n -o BatchMode=yes tatooine "$@"; }
vram_stop() { if [[ -n "$SAMPLER" ]]; then kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=""; fi; }
vram_start() { # vram_start <file>: epoch, then VRAM bytes used on card0 and card1, once a second, until vram_stop
  vram_stop
  ( while :; do echo "$(date +%s) $(ssh -n -o BatchMode=yes tatooine 'cat /sys/class/drm/card0/device/mem_info_vram_used /sys/class/drm/card1/device/mem_info_vram_used' 2>/dev/null | tr '\n' ' ')"; sleep 1; done ) >> "$1" 2>/dev/null &
  SAMPLER=$!
}
RESTORED=0
restore() {
  trap - EXIT; vram_stop
  rsh "docker rm -f dp-t dp-d0 dp-d1 dp-c >/dev/null 2>&1"
  if [[ ${DRY:-0} == 1 ]]; then log "DRY: radiance untouched"; return 0; fi
  log "starting radiance"
  rsh "FLEETQ_JOB=$FLEETQ_JOB systemctl --user start radiance"
  for i in $(seq 1 36); do
    if rsh "curl -s -m 3 http://localhost:8080/v1/models" | grep -q '"data":\['; then log "radiance answers on :8080 (after $((i * 5)) s)"; RESTORED=1; return 0; fi
    sleep 5
  done
  log "RADIANCE DID NOT ANSWER within 180 s"; return 1
}
rsh 'test -x $HOME/intarsiaml-dp/dp-intarsia' || { echo 'no ~/intarsiaml-dp/dp-intarsia: run scripts/dp/build.sh first (radiance untouched)'; exit 5; }
trap restore EXIT
trap 'exit 130' HUP INT TERM
log "fleetq job $FLEETQ_JOB; window.sh $(git -C "$H" rev-parse --short=9 HEAD 2>/dev/null)"
rsh "systemctl --user status radiance --no-pager" > "$OUT/radiance-status-before.txt" 2>&1
{ echo "nproc: $(rsh nproc)"; rsh "lscpu | grep -E 'Model name|^CPU\(s\)|Thread|NUMA node\(s\)'; free -g; pgrep -af 'systemctl --user start radiance' || echo 'no stray radiance-start timers'; for c in card0 card1; do echo \$c \$(grep PCI_SLOT_NAME /sys/class/drm/\$c/device/uevent); done"; } > "$OUT/host.txt" 2>&1
cat "$OUT/host.txt" | tee -a "$OUT/steps.log" >/dev/null
# another lane's deadman timer that is still pending would restart radiance inside the window: do not start then (pend.sh lists the seconds left)
PENDING=$(ssh -o BatchMode=yes tatooine bash -s < "$H/pend.sh")
log "pending foreign radiance-start timer (seconds until it fires): ${PENDING:-none}"
if [[ ${DRY:-0} != 1 ]] && [[ -n "$PENDING" ]]; then log "a radiance-start timer fires in ${PENDING} s: not starting (radiance untouched); rerun after it, or have its owner remove it"; exit 6; fi
if [[ ${DRY:-0} != 1 ]]; then
  log "stopping radiance"
  rsh "FLEETQ_JOB=$FLEETQ_JOB systemctl --user stop radiance" && sleep 5
  rsh "cat /sys/class/drm/card0/device/mem_info_vram_used /sys/class/drm/card1/device/mem_info_vram_used" | tr '\n' ' ' > "$OUT/vram-baseline.txt"
  log "VRAM used after the stop (card0 card1, bytes): $(cat "$OUT/vram-baseline.txt")"
  for b in $(cat "$OUT/vram-baseline.txt"); do (( b > 4000000000 )) && { log "a card still holds $b bytes after radiance stopped: abort"; exit 3; }; done
fi
srv() { # srv <name> <port> <HIP_VISIBLE_DEVICES> <max-seq> <max-batch> <extra serve args> [extra docker args]
  rsh "docker rm -f $1 >/dev/null 2>&1; docker run $DRYM --name $1 --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video \
    --ipc host --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
    -e HIP_VISIBLE_DEVICES=$3 ${7:-} -v $R:/work -v /data/models/intarsia:/models:ro -w /work \
    $IMG /work/dp-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 \
    --profile-every 0 --mem-budget ${MEMB:-29} --host 0.0.0.0 --port $2 --served-model-name intarsia --max-seq $4 --max-batch $5 ${6:-} ${DRY_TAIL:-}" \
    > "$OUT/serve-start-$1.txt" 2>&1
}
wait_health() { # wait_health <name> <port>
  for i in $(seq 1 100); do
    rsh "curl -sf -m 2 http://localhost:$2/health" >/dev/null 2>&1 && return 0
    [[ $(rsh "docker inspect -f '{{.State.Running}}' $1 2>/dev/null") == true ]] || { log "$1 is not running (load failed)"; return 1; }
    sleep 3
  done
  log "$1 not healthy in 300 s"; return 1
}
radiance_down() { [[ ${DRY:-0} == 1 ]] || [[ $(rsh "systemctl --user is-active radiance") != active ]]; }
drive() { # drive <label> <load.py args...>: the driver on endor, output to <label>.txt, json to <label>.json
  local l=$1; shift
  python3 "$H/load.py" "$@" --out "$OUT/$l.json" 2>&1 | tee "$OUT/$l.txt" | tee -a "$OUT/steps.log"
}
grab() { rsh "docker logs $1" > "$OUT/serve-$2-$1.log" 2>&1; }
visit() { # visit <label> <T|D|X>
  local L=$1 arm=$2
  radiance_down || { log "radiance is active again: abort"; exit 4; }
  log "== $L ($arm)"
  rsh "free -g | head -2" >> "$OUT/steps.log"
  vram_start "$OUT/$L-vram.tsv"
  if [[ $arm == D ]]; then
    srv dp-d0 $P0 0 8192 16 "" || true
    [[ ${DRY:-0} == 1 ]] || srv dp-d1 $P1 1 8192 16 "" || true
    [[ ${DRY:-0} == 1 ]] && { cat "$OUT/serve-start-dp-d0.txt" | head -80 > "$OUT/serve-help.txt"; vram_stop; return 0; }
    wait_health dp-d0 $P0 && wait_health dp-d1 $P1 || { grab dp-d0 $L; grab dp-d1 $L; vram_stop; rsh "docker rm -f dp-d0 dp-d1 >/dev/null 2>&1"; return 1; }
    local urls=$U0,$U1
  else
    local mb=16; [[ $arm == X ]] && mb=32
    srv dp-t $PT 0,1 8192 $mb "--tp 2 --devices 0,1" || true
    [[ ${DRY:-0} == 1 ]] && { vram_stop; return 0; }
    wait_health dp-t $PT || { grab dp-t $L; vram_stop; rsh "docker rm -f dp-t >/dev/null 2>&1"; return 1; }
    local urls=$UT
  fi
  drive $L-ident --urls $urls --ident
  if [[ $arm == D && $L == D1 ]]; then   # CPU starvation check: one server alone at 16 streams, then both at 32 with CPU sampled
    drive $L-solo16 --urls $U0 --streams 16 --bursts 3
  fi
  if [[ $arm == X ]]; then drive $L-load --urls $urls --streams 16,32; else drive $L-load --urls $urls --streams 1,8,16,32; fi
  if [[ $arm == D && $L == D1 ]]; then
    python3 "$H/load.py" --urls $U0,$U1 --streams 32 --bursts 1 --gen 512 --no-warm --out "$OUT/$L-cpucheck.json" > "$OUT/$L-cpucheck.txt" 2>&1 &
    local lp=$!; sleep 8
    for k in 1 2 3; do { echo "--- sample $k $(date +%T)"; rsh "docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}'; ps -eo pid,nlwp,pcpu,comm --sort=-pcpu | head -6; top -bn1 | sed -n 1,4p"; } >> "$OUT/$L-cpu.txt" 2>&1; sleep 1; done
    wait $lp; cat "$OUT/$L-cpucheck.txt" | tee -a "$OUT/steps.log"
  fi
  if [[ $arm == T && $L == T1 ]]; then   # calibration against 11-05 (localeval, tp2): 1 stream 163.0, 16 streams 476.1 tok/s
    python3 - "$OUT/$L-load.json" <<'PY' | tee -a "$OUT/steps.log"
import json, sys
rows = {r["streams"]: r for r in json.load(open(sys.argv[1]))["rows"]}
for n, ref in ((1, 163.0), (16, 476.05)):
    v = rows[n]["agg_sum_median"]
    print(f"CALIBRATION T1 N={n}: {v:.1f} tok/s against 11-05 {ref} ({(v / ref - 1) * 100:+.1f} %)" + ("" if abs(v / ref - 1) < 0.08 else "  <-- MORE THAN 8 % OFF: driver or box differs, do not trust ratios blindly"))
PY
  fi
  for n in dp-t dp-d0 dp-d1; do rsh "docker inspect -f '{{.State.Running}}' $n 2>/dev/null" | grep -q true && grab $n $L; done
  rsh "docker rm -f dp-t dp-d0 dp-d1 >/dev/null 2>&1"; vram_stop
}
START=$SECONDS
for v in ${VISITS:-T1 D1 T2 D2}; do visit $v "${v:0:1}" || log "$v failed"; done
[[ ${DRY:-0} == 1 ]] && { log "DRY done"; exit 0; }
ctx_try() { # ctx_try <max-seq> [max-batch=16] [mem-budget GiB=29]: one server on card 0 (the second card idle), a ~16k-token prompt
  local ms=$1 mb=${2:-16} bud=${3:-29} L=ctx-$1
  [[ $mb != 16 || $bud != 29 ]] && L=ctx-$1-mb$mb-b$bud
  radiance_down || { log "radiance is active again: abort"; exit 4; }
  log "== $L"
  vram_start "$OUT/$L-vram.tsv"
  MEMB=$bud srv dp-c $P0 0 $ms $mb "" || true
  local ok=False
  if wait_health dp-c $P0; then
    drive $L --urls $U0 --ctx-probe 16000
    ok=$(python3 -c "import json;d=json.load(open('$OUT/$L.json'));print(bool(d['ok'] and (d['prompt_tokens'] or 0) >= 15500))")
  fi
  grab dp-c $L; rsh "docker rm -f dp-c >/dev/null 2>&1"; vram_stop
  log "max-seq $ms max-batch $mb mem-budget $bud: server context $(grep -ao 'context [0-9]* tokens' "$OUT/serve-$L-dp-c.log" | head -1); loads and serves a ~16k prompt = $ok"
  [[ $ok == True ]]
}
if [[ -n ${CTX_CONFIGS:-} ]]; then   # CTX_CONFIGS="maxseq:maxbatch:budget ...": each tried once, nothing else runs (VISITS="" EXTRA=0)
  for c in $CTX_CONFIGS; do IFS=: read -r cm cb cg <<< "$c"; ctx_try "$cm" "$cb" "$cg" || true; done
elif [[ ${PROBE:-1} == 1 ]]; then
  if ctx_try 32768; then ctx_try 65536 || true
  else ctx_try 24576 || ctx_try 20480 || ctx_try 18432 || true; fi
fi
if [[ ${EXTRA:-1} == 1 ]] && (( SECONDS - START < CAP )); then visit X1 X || log "X1 failed"; else log "extra max-batch 32 T visit skipped (elapsed $((SECONDS - START)) s)"; fi
log "all done in $((SECONDS - START)) s"
