#!/usr/bin/env bash
# Lane vr16: ONE spark2 window (retro taken down): vLLM --max-num-seqs 16 vs intarsia main, localeval concurrency 1/4/8/16.
#   (inside a fleetq slot, FLEETQ_JOB set)  scripts/vr16/window.sh <rounds>
# Order: stop spark2-vllm (container missioncntrl-guidance) -> run the SAME launcher with only --max-num-seqs 16 (container vr16-vllm)
# -> localeval x <rounds> -> stop it -> intarsia main x <rounds> (fresh server per round) -> restore the original container and check /v1/models.
# Settings are the scc ones (scripts/scc/le-window.sh): 1024-token prompts, gen 256, reps 3, --thinking off, intarsia --max-seq 8192 --mem-budget 40 --max-batch 16.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
ROUNDS=${1:-2}
HOST=spark2; HN=spark-fa00; PORT=8190
OUT=/opt/code/intarsiaml/.worktrees/vr16/docs/results/2026-10-27-vr16-vllm-seqs16/raw; mkdir -p $OUT
R='$HOME/intarsiaml-vr16'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SIZES=1024; GEN=256; CONC=1,4,8,16
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
SPID=""; RESTORED=0
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
models_up() { curl -sf -m 5 "http://$HN:8000/v1/models" >/dev/null; }
restore() {
  stop_server
  (( RESTORED )) && return
  echo "== restore $(date +%T)"
  rsh "docker rm -f vr16-vllm >/dev/null 2>&1; docker start missioncntrl-guidance"
  local i; for i in $(seq 1 150); do models_up && { RESTORED=1; echo "restored: /v1/models answers $(date +%T)"; curl -s "http://$HN:8000/v1/models" | head -c 150; echo; return; }; sleep 5; done
  echo "RESTORE FAILED: /v1/models does not answer"
}
trap 'restore' EXIT INT TERM
le() { # le <label> <url> <model> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "$2" --model "$3" --thinking off --sizes $SIZES --gen $GEN --reps 3 --concurrency $CONC \
      --label "$1" --no-unim --recipe "$4" ) > $OUT/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $OUT/$1.out | head -2 | tr '\n' ' ')" | tee -a $OUT/index.txt
  grep -aE 'streams: aggregate' $OUT/$1.out
}
# 2a. swap vLLM
echo "== stop original $(date +%T)"
rsh "docker inspect missioncntrl-guidance --format '{{json .Config.Cmd}}'" > $OUT/vllm-original-cmd.json
rsh "cat ~/missioncntrl-guidance-run.sh | sed -e 's/--max-num-seqs 2 /--max-num-seqs 16 /' -e 's/--restart unless-stopped/--restart no/'" > $OUT/vllm-seqs16-launcher.sh
rsh "docker stop missioncntrl-guidance >/dev/null"
ssh -o BatchMode=yes $HOST "NAME=vr16-vllm bash -s" < $OUT/vllm-seqs16-launcher.sh
echo "== waiting for vr16-vllm $(date +%T)"
up=0; for i in $(seq 1 240); do models_up && { up=1; break; }; sleep 5; done
if [[ $up != 1 ]]; then echo "vLLM seqs16 did not come up"; rsh "docker logs --tail 15 vr16-vllm 2>&1"; exit 1; fi
echo "== vLLM up $(date +%T)"
rsh "docker inspect vr16-vllm --format '{{json .Config.Cmd}}'" > $OUT/vllm-seqs16-cmd.json
VR="spark2 vLLM (missioncntrl/vllm-openai:65b7662d) with ONLY --max-num-seqs 16 changed (flags in raw/vllm-seqs16-cmd.json)"
for r in $(seq 1 $ROUNDS); do echo "== vllm16 round $r $(date +%T)"; le "scc-vllm16-r$r" "http://$HN:8000/v1" guidance "$VR"; done
# 2c. swap to intarsia
echo "== stop vLLM $(date +%T)"
rsh "docker rm -f vr16-vllm >/dev/null"; sleep 5
for r in $(seq 1 $ROUNDS); do
  echo "== main round $r $(date +%T)"
  SPID=$(rsh "mkdir -p $R/out; nohup $R/bin/main serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 > $R/out/serve-main-r$r.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  if [[ $up != 1 ]]; then echo "intarsia did not come up"; rsh "tail -5 $R/out/serve-main-r$r.log"; stop_server; continue; fi
  le "scc-main-r$r" "http://$HN:$PORT/v1" intarsia "intarsia serve --backend cuda --max-seq 8192 --mem-budget 40 bin=main --max-batch 16"
  stop_server
done
restore
echo "== done $(date +%T)"
