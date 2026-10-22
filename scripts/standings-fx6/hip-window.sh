#!/usr/bin/env bash
# fx6: R9700 (tatooine, ONE card = HIP/Vulkan device 0) interleaved same-session comparison, Qwen3.6-35B-A3B.
# Runs on the Mac (needs /opt/code/deepeval) as ONE fleetq command so the EXIT trap restores radiance:
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "fx6 R9700 pass 1" -- \
#     env OUT=<dir> ARMS="intarsia llama-vk hipfire llama-hip" PASS=p1 bash scripts/standings-fx6/hip-window.sh
# ARMS is the visit order (interleaving: pass 2 uses the reverse order). Arms:
#   intarsia   intarsia HIP, NVFP4, tree ~/intarsiaml-fx6 (origin/main cd617de2), volume intarsia-cargo-target-fx6
#   llama-vk   llama.cpp v0.5.0 (build llama.cpp-v0.5.0-vulkan, 7fe450e) Vulkan/RADV, MXFP4_MOE GGUF
#   llama-vk4  llama.cpp 0.4.0-dev (10864) Vulkan, the comparator of the 2026-10-21 standings (continuity)
#   llama-hip  llama.cpp v0.5.0 HIP build in image fx6-rocm-llama:7.2.4
#   intarsia-old  the 2026-10-21 standings binary (tree ~/intarsiaml-standings, before the J1 response_format merge), MMLU diagnosis control
#   hipfire    hipfire ad10b3d MQ4R (container hipfire:ad10b3d), sequential mode, via ssh tunnel
# Optional MMLU=1 adds, after the intarsia and llama-vk arms, a small MMLU latency diagnosis (finding 3: 8x per-call latency).
# Window cap WINDOW_S (<= 1140 s) of radiance downtime INCLUDING the restart poll; arms that no longer fit are skipped and logged.
set -uo pipefail
: "${OUT:?set OUT to the raw output dir}"
mkdir -p "$OUT"
T=tatooine
H=$(cd "$(dirname "$0")" && pwd)
WINDOW_S=${WINDOW_S:-1140}
ARMS=${ARMS:-"intarsia llama-vk hipfire llama-hip"}
PASS=${PASS:-p1}
[ "${MMLU:-0}" = 1 ] && ARMS="intarsia-old $ARMS"   # the pre-J1 binary of the 2026-10-21 standings (tree ~/intarsiaml-standings) is the 8x MMLU-latency control
SIZES=${SIZES:-128,1024,4096,16384}
LE=/opt/code/deepeval
LOG=$OUT/downtime.log
now() { date '+%F %T'; }
say() { echo "[window] $* $(now)" | tee -a "$LOG"; }
capped() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }

# ---- preflight: everything that can fail cheaply fails BEFORE radiance is stopped ----
: >"$OUT/preflight.out"
rsync -a "$H/remote/" "$T:fx6-remote/" >>"$OUT/preflight.out" 2>&1 || { echo "PREFLIGHT FAIL rsync"; exit 3; }
ssh $T 'docker run --rm -v intarsia-cargo-target-standings:/target:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /target/release/intarsia && echo old-bin-ok; docker run --rm -v intarsia-cargo-target-fx6:/target:ro intarsia-hip-dev:rocm7.2.4-rust1.90.0 test -x /target/release/intarsia && echo intarsia-bin-ok; test -x /home/josef/bench-r9700/llama.cpp-v0.5.0-vulkan/build/bin/llama-server && echo vk-ok; docker image inspect fx6-rocm-llama:7.2.4 hipfire:ad10b3d >/dev/null && echo images-ok; LD_LIBRARY_PATH=/home/josef/bench-r9700/llama.cpp-v0.5.0-vulkan/build/bin /home/josef/bench-r9700/llama.cpp-v0.5.0-vulkan/build/bin/llama-server --version 2>&1 | grep -i version; /home/josef/.local/opt/llama.cpp/llama-server --version 2>&1 | grep -i version; cat ~/intarsiaml-fx6/.fx6-commit; systemctl --user is-active radiance' </dev/null >>"$OUT/preflight.out" 2>&1
grep -q intarsia-bin-ok "$OUT/preflight.out" && grep -q images-ok "$OUT/preflight.out" || { echo "PREFLIGHT FAIL (see $OUT/preflight.out)"; exit 3; }
grep -q old-bin-ok "$OUT/preflight.out" || ARMS=$(echo " $ARMS " | sed 's/ intarsia-old / /')   # control binary missing: drop that arm
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { echo preflight ok; exit 0; }

t0=$(date +%s)
left() { echo $(( WINDOW_S - ($(date +%s) - t0) )); }
TUN=""
stop_servers() {
  [ -n "$TUN" ] && { kill $TUN 2>/dev/null; TUN=""; }
  ssh $T 'bash ~/fx6-remote/stop-all.sh' </dev/null >/dev/null 2>&1
}
restore() {
  stop_servers
  say "starting radiance"
  ssh $T 'systemctl --user start radiance' </dev/null
  local i
  for i in $(seq 1 36); do
    if ssh $T 'curl -s -m 3 http://localhost:8080/v1/models' </dev/null | grep -q '"data":\['; then
      say "radiance answers (after $((i * 5)) s); window total $(( $(date +%s) - t0 )) s"
      return 0
    fi
    sleep 5
  done
  say "RADIANCE DID NOT ANSWER within 180 s"
  return 1
}
wait_up() { # wait_up <curl url> <secs>
  local i
  for ((i = 0; i < $2; i += 3)); do
    curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0
    sleep 3
  done
  return 1
}
gpu() { ssh $T 'bash ~/fx6-remote/sample.sh' </dev/null 2>&1; }
save_result() { # <localeval out file> <dest name>
  local res; res=$(grep -o 'saved .*result.json' "$1" | tail -1 | sed 's/^saved //')
  [ -n "$res" ] && [ -f "$res" ] && cp "$res" "$OUT/$2.result.json" && cp "$(dirname "$res")/summary.md" "$OUT/$2.summary.md" 2>/dev/null
}

# visit <arm> <need-seconds>
visit() {
  local arm=$1 need=$2 url mid start name
  name="$arm-$PASS"
  if (( $(left) < need )); then say "SKIPPED $name: $(left) s left, needs $need"; return 0; fi
  case $arm in
    intarsia)  start="bash ~/fx6-remote/start-intarsia.sh"; url=http://$T:8090; mid=qwen36-35b-intarsia-hip; health=$url/health ;;
    intarsia-old) start="bash ~/fx6-remote/start-intarsia.sh old"; url=http://$T:8090; mid=qwen36-35b-intarsia-hip; health=$url/health ;;
    llama-vk)  start="bash ~/fx6-remote/start-llama-vk.sh v0.5.0"; url=http://$T:8091; mid=MXFP4_MOE; health=$url/health ;;
    llama-vk4) start="bash ~/fx6-remote/start-llama-vk.sh v0.4.0"; url=http://$T:8091; mid=MXFP4_MOE; health=$url/health ;;
    llama-hip) start="bash ~/fx6-remote/start-llama-hip.sh"; url=http://$T:8091; mid=MXFP4_MOE; health=$url/health ;;
    hipfire)   start="bash ~/fx6-remote/start-hipfire.sh"; url=http://$T:11435; mid=qwen3.6-35b-a3b.mq4r; health=$url/v1/models ;;
    *) say "unknown arm $arm"; return 0 ;;
  esac
  say "visit $name: starting server ($(left) s left); gpu: $(gpu)"
  stop_servers; sleep 2
  ssh $T "$start" </dev/null >"$OUT/$name-server-start.out" 2>&1
  if ! wait_up "$health" $([ $arm = hipfire ] && echo 450 || echo 240); then
    say "visit $name: server did not come up"
    ssh $T 'docker logs --tail 40 fx6-intarsia 2>&1; docker logs --tail 40 fx6-llamacpp-hip 2>&1; docker logs --tail 40 fx6-hipfire 2>&1; tail -40 /tmp/fx6-llama.log 2>&1' </dev/null >"$OUT/$name-server.err" 2>&1
    stop_servers; return 1
  fi
  say "visit $name: server up ($(left) s left)"
  local extra=(); [ $arm = hipfire ] && extra=(--ignore-busy)
  local cap=$(( $(left) - 45 )); (( cap > 420 )) && cap=420
  echo "gpu before: $(gpu)" >"$OUT/$name-gpu.txt"
  (cd $LE && capped "$cap" uv run localeval speed --url "$url/v1" --model "$mid" --sizes "$SIZES" --gen 128 --reps 3 --thinking off \
     ${extra[@]+"${extra[@]}"} --label "fx6-$name") >"$OUT/$name-localeval.out" 2>&1
  say "visit $name: localeval speed exit $? ($(left) s left)"
  echo "gpu after: $(gpu)" >>"$OUT/$name-gpu.txt"
  save_result "$OUT/$name-localeval.out" "$name"
  curl -s -m 3 "$url/metrics" 2>/dev/null | grep -iE "graph|eager" | head >"$OUT/$name-metrics.txt" 2>&1
  if [ "${SB:-0}" = 1 ]; then   # reconciliation: the 2026-09-26 stream_bench.py (same decode formula as localeval) on this server, single stream
    local port=${url##*:}
    ssh $T "timeout 300 python3 ~/bench-r9700/stream_bench.py --url http://127.0.0.1:$port/v1 --model $mid --streams 1 --output /tmp/fx6-sb-$name.jsonl" </dev/null >"$OUT/$name-streambench.out" 2>&1
    ssh $T "cat /tmp/fx6-sb-$name.jsonl; rm -f /tmp/fx6-sb-$name.jsonl" </dev/null >"$OUT/$name-streambench.jsonl" 2>&1
    say "visit $name: stream_bench done ($(left) s left)"
  fi
  [ -n "${PROBE:-}" ] && python3 "$H/probe-mmlu.py" "$url/v1" "$mid" "$OUT/$name-probe.jsonl" >"$OUT/$name-probe.out" 2>&1
  if [ "${MMLU:-0}" = 1 ] && { [ $arm = intarsia ] || [ $arm = intarsia-old ] || [ $arm = llama-vk ]; } && (( $(left) > 150 )); then
    local st
    for st in json_schema none; do
      (( $(left) > 90 )) || break
      (cd $LE && capped $(( $(left) > 150 ? 100 : $(left) - 45 )) uv run localeval quality --url "$url/v1" --model "$mid" --bench mmlu --limit 4 \
         --tasks high_school_physics,virology --thinking off --structured $st --judge none --no-unim --label "fx6-$name-mmlu-$st") \
         >"$OUT/$name-mmlu-$st.out" 2>&1
      say "visit $name: mmlu --structured $st exit $? ($(left) s left)"
      save_result "$OUT/$name-mmlu-$st.out" "$name-mmlu-$st"
    done
  fi
  stop_servers
  sleep 3
}

say "job ${FLEETQ_JOB:-none}; pass $PASS; arms: $ARMS; window cap ${WINDOW_S} s; stopping radiance"
trap restore EXIT
ssh $T 'systemctl --user stop radiance' </dev/null
sleep 2
say "radiance stopped; gpu: $(gpu)"
for a in $ARMS; do
  case $a in intarsia|intarsia-old) n=240 ;; hipfire) n=300 ;; *) n=210 ;; esac
  visit "$a" $n
done
say "arms done"
