#!/usr/bin/env bash
# Lane c8k: decode stall (ITL of 4 streams) for 4k and 16k prompts at chunk 2048, 8192, and 8192 with `--busy-chunk 2048` (the opt-in knob:
# solo prefill at 8192, prefill beside decoding rows at 2048). Mac side, inside a fleetq slot on spark2 (no vLLM here; retro stays up and
# its running-request count is recorded before each visit). Visits interleaved x3 (round 2 reversed).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 60 --label "..." -- scripts/c8k/itl-window.sh <outname> <rounds> <tree>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H=spark2; HN=spark-fa00; PORT=8190
OUTN=$1; ROUNDS=$2
D=/opt/code/wt-dfB/scripts/c8k
OUT=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/results/$OUTN; mkdir -p "$OUT"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
rsh() { ssh -n -o BatchMode=yes "$H" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
SPID=""
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3" ; SPID=""; }; }
trap stop_server EXIT INT TERM
run_arm() { # run_arm <round> <name> <chunk> <extra serve args>
  local r=$1 name=$2 tree=$3; shift 3
  log "== round $r arm $name"
  log "retro running requests: $(curl -s -m 5 http://$HN:8000/metrics | grep -aE '^vllm:num_requests_running' | head -1)"
  SPID=$(rsh "cd ~/intarsia-decfuse; nohup ~/intarsia-decfuse/$tree/target/acc/release/intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 45 --profile-every 0 --max-batch 4 $* > ~/intarsia-decfuse/out/df-itl-$name-r$r.log 2>&1 < /dev/null & echo \$!")
  local i up=0
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" > /dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then log "server $name did not come up"; stop_server; return 1; fi
  rsh "grep -aE 'prefill chunk|continuous batching' ~/intarsia-decfuse/out/df-itl-$name-r$r.log | head -3" | cut -c1-200 | tee -a "$OUT/run.log"
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run python "$D/itl4.py" --url http://$HN:$PORT/v1 --model intarsia-35b --prompt-tokens 4096 --reps 2 --out "$OUT/itl4k-$name-r$r.json" ) > "$OUT/itl4k-$name-r$r.out" 2>&1
  grep -a '^\[itl4\]' "$OUT/itl4k-$name-r$r.out" | tail -1 | cut -c1-330 | tee -a "$OUT/run.log"
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run python "$D/itl4.py" --url http://$HN:$PORT/v1 --model intarsia-35b --prompt-tokens 16384 --reps 2 --stagger-s 4 --stagger-gens 768,384,384,384 --out "$OUT/itl16k-$name-r$r.json" ) > "$OUT/itl16k-$name-r$r.out" 2>&1
  grep -a '^\[itl4\]' "$OUT/itl16k-$name-r$r.out" | tail -1 | cut -c1-330 | tee -a "$OUT/run.log"
  stop_server
}
ARMS=("main2048:df-dev:" "c4096:df-dev:--prefill-chunk 4096" "b8192:df-dev:--prefill-chunk 8192" "a8192b2048:df-pfb:")
for r in $(seq 1 $ROUNDS); do
  idx=(0 1 2 3); (( r % 2 == 0 )) && idx=(3 2 1 0)
  for k in "${idx[@]}"; do IFS=: read -r n c x <<< "${ARMS[k]}"; run_arm $r "$n" "$c" $x; done
done
log "== done"
