#!/usr/bin/env bash
# G4X: one interleaved round of localeval speed arms on spark2 (runs on endor; localeval is the only benchmark driver allowed by the fleet rules).
#   le.sh <round> <arm:binary-name> ...      e.g. le.sh 1 base:base g1:g1 g2a:g2a
# Holds ONE fleetq slot (spark2, --quiet gpu,cpu, --mem 34) for the round; per arm a fresh `intarsia serve` (0.0.0.0:8091, model id g4x),
# `localeval speed` with FLEETQ_JOB set to the slot, then the server is stopped. Binaries: ~/intarsiaml-g4x/replay-out/bins/<name> on spark2.
# Env: SIZES (128,1024,4096,16384) EXPECT (minutes, 15 per arm).
set -uo pipefail
RND=$1; shift
SIZES=${SIZES:-128,1024,4096,16384}; N=$#; EXPECT=${EXPECT:-$((15 * N + 5))}
OUT=${LE_LOG:-/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/le}; mkdir -p $OUT
job=$(fleetq acquire --host spark2 --mem 34 --quiet gpu,cpu --expect $EXPECT --wait --label "G4X localeval speed round $RND: $*" 2>&1 | tee $OUT/acquire-r$RND.log | grep -oE '#[0-9]+' | head -1 | tr -d '#')
[ -n "$job" ] || { echo "no fleetq job"; exit 1; }
echo "slot #$job"
trap 'fleetq release $job >/dev/null 2>&1' EXIT
for spec in "$@"; do
  arm=${spec%%:*}; bin=${spec#*:}
  R=/home/nvidia/intarsiaml-g4x
  cmd="$R/replay-out/bins/$bin serve --backend cuda --model /home/nvidia/models/Gemma-4-26B-A4B-NVFP4 --port 8091 --host 0.0.0.0 --served-model-name g4x --max-seq 20480 --mem-budget 30 --profile-every 0"
  echo "== round $RND arm $arm $(date +%T)"
  ssh -n -o BatchMode=yes spark2 "gb=\$(( \$(awk '/MemAvailable/ {print \$2}' /proc/meminfo) / 1024 / 1024 )); [ \$gb -ge 45 ] || { echo ABORT memory \$gb; exit 9; }; nohup $cmd > $R/replay-out/le-serve-$arm-r$RND.log 2>&1 < /dev/null & echo \$! > $R/replay-out/le-serve.pid" || { echo "start failed"; break; }
  up=0; for i in $(seq 1 90); do curl -sf http://spark-fa00:8091/health >/dev/null && { up=1; break; }; sleep 2; done
  if [ $up = 1 ]; then
    [ -n "${SKIP_SPEED:-}" ] || (cd /opt/code/deepeval && FLEETQ_JOB=$job uv run localeval speed --url http://spark-fa00:8091/v1 --model g4x --thinking off --sizes $SIZES --gen 128 --reps ${REPS:-2} \
       --label "g4x-$arm-r$RND" --recipe "$cmd" > $OUT/$arm-r$RND.log 2>&1; echo "localeval exit $?"; grep -aE "run_id|Run id|record|unanimis|results/" $OUT/$arm-r$RND.log | head -4)
    if [[ " ${MMLU_ARMS:-} " == *" $arm "* ]]; then  # MMLU 5-shot, --limit 25 per subject = n 1425 (G4D's setting), thinking off
      (cd /opt/code/deepeval && FLEETQ_JOB=$job uv run localeval quality --url http://spark-fa00:8091/v1 --model g4x --thinking off --bench mmlu --limit 25 \
         --label "g4x-$arm-mmlu" --recipe "$cmd" > $OUT/$arm-mmlu.log 2>&1; echo "localeval mmlu exit $?"; grep -aE "mmlu|run_id|Run id|record" $OUT/$arm-mmlu.log | tail -4 | cut -c1-200)
    fi
  else echo "server did not come up"; fi
  ssh -n -o BatchMode=yes spark2 "kill \$(cat $R/replay-out/le-serve.pid) 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 \$(cat $R/replay-out/le-serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 \$(cat $R/replay-out/le-serve.pid) 2>/dev/null; true"
  sleep 3
done
for x in ${EXTRA:-}; do  # extra steps inside the same slot: g0 = CUDA plan show, prof16k = nsys decode profile of the base build at 16k
  case $x in
    g0) echo "== G0 $(date +%T)"; ssh -n spark2 "bash $R/replay-out/g0.sh ${G0_BINS:-base g2b}" ;;
    prof16k) echo "== profile $(date +%T)"; ssh -n spark2 "bash $R/replay-out/profile.sh base-16k bins/${PROF_BIN:-base} 64000" ;;
  esac
done
echo "== le round $RND done $(date +%T)"
