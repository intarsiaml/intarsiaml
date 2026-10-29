#!/usr/bin/env bash
# sb8: served-token identity, solo (no --max-batch) vs batched (--max-batch 16 via recipe cap 16), Qwen3-8B BF16 on spark2.
set -uo pipefail
: "${FLEETQ_JOB:?}"
H=spark2; HN=spark-fa00; PORT=8192
WT=/opt/code/intarsiaml/.worktrees/sb8
RAW=$WT/docs/results/2026-10-29-sb8-spark-8b-multistream/raw/ident; mkdir -p $RAW
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
BIN=/home/nvidia/intarsiaml-sb8/bin/intarsia
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
for arm in solo mb16; do
  if [[ $arm == solo ]]; then X=""; else X="--recipe /home/nvidia/intarsiaml-sb8/qwen3-mb16.toml --max-batch 16"; fi
  SPID=$(rsh "nohup $BIN serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 $X > ~/intarsiaml-sb8/out/ident-$arm.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server $arm did not come up"; stop_server; exit 1; }
  ( cd $WT && python3 scripts/htk/equal.py run --url http://$HN:$PORT/v1 --out $RAW/$arm.json ) > $RAW/$arm.run.log 2>&1
  tail -2 $RAW/$arm.run.log
  stop_server
done
( cd $WT && python3 scripts/htk/equal.py compare $RAW/solo.json $RAW/mb16.json ) | tee $RAW/compare.log | tail -5
