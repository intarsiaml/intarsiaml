#!/usr/bin/env bash
# Lane fp8kv, GPU window 3 (one fleetq slot on spark2, quiet on GPU and CPU), dev binary, one `intarsia serve` per arm
# (--max-seq 20480 --mem-budget 40, recipe defaults: MTP k=2 at one stream), after stepprof.sh (the B = 16 attention bucket):
#   bf16      : greedy.py (1k / 4k / 16k, 2 prompts each, 256 greedy tokens), localeval quality mmlu (10 per task) + gsm8k (100)
#   fp8       : the same on --kv-dtype fp8, greedy.py run twice (run-to-run determinism; the second run restores the prefixes)
#   fp8-nospec: the fp8 cache set by the RECIPE (a copy of recipes/qwen3_5_moe.toml with kv_dtype = "fp8_e4m3", `--recipe`) and
#               --spec off, greedy.py once (MTP drafted == serial on the fp8 cache; /metrics must say kv_dtype="fp8_e4m3")
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 45 --label "fp8kv: W3 accuracy bf16 vs fp8 KV" -- scripts/fp8kv/w3.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW=$H/../../docs/results/2026-10-29-fp8kv-opt-in/raw/w3; mkdir -p $RAW
HOST=spark2; HN=spark-fa00; PORT=8190
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
SPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap 'stop_server' EXIT INT TERM
SERVE="serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 20480 --mem-budget 40 --profile-every 0"
start() { # start <name> <extra args>
  SPID=$(rsh "mkdir -p ~/intarsiaml-fp8kv/out; nohup ~/intarsiaml-fp8kv/bin/dev $SERVE $2 > ~/intarsiaml-fp8kv/out/w3-$1.log 2>&1 < /dev/null & echo \$!")
  local i; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { curl -s "http://$HN:$PORT/metrics" | grep -a '^intarsia_info' | tee $RAW/info-$1.txt; return 0; }; sleep 2; done
  echo "server $1 did not come up"; rsh "tail -5 ~/intarsiaml-fp8kv/out/w3-$1.log"; return 1
}
quality() { # quality <name> <recipe text>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval quality --url "http://$HN:$PORT/v1" --model intarsia --thinking off \
      --bench mmlu --limit 10 --label "fp8kv-$1-mmlu" --recipe "$2" ) > $RAW/quality-$1-mmlu.out 2>&1
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval quality --url "http://$HN:$PORT/v1" --model intarsia --thinking off \
      --bench gsm8k --limit 100 --label "fp8kv-$1-gsm8k" --recipe "$2" ) > $RAW/quality-$1-gsm8k.out 2>&1
  grep -aE 'run_id|Run id|results/|mmlu|gsm8k' $RAW/quality-$1-*.out | tail -12
}
echo "commit $(git -C "$H" rev-parse --short HEAD) $(date +%T)"
"$H/stepprof.sh" bf16 fp8_e4m3   # the B = 16 decode-attention bucket under nsys, first (nothing else loaded)
rsh "sed 's/^kv_dtype = \"bf16\"/kv_dtype = \"fp8_e4m3\"/' ~/intarsiaml-fp8kv/src/recipes/qwen3_5_moe.toml > ~/intarsiaml-fp8kv/out/qwen3_5_moe-fp8.toml; grep -n '^kv_dtype' ~/intarsiaml-fp8kv/out/qwen3_5_moe-fp8.toml"
for arm in bf16 fp8 fp8-nospec; do
  case $arm in bf16) X="" ;; fp8) X="--kv-dtype fp8" ;; fp8-nospec) X="--recipe ~/intarsiaml-fp8kv/out/qwen3_5_moe-fp8.toml --spec off" ;; esac
  echo "== $arm $(date +%T)"
  start $arm "$X" || continue
  python3 "$H/greedy.py" run --url http://$HN:$PORT/v1 --out $RAW/greedy-$arm.json > $RAW/greedy-$arm.log 2>&1; tail -1 $RAW/greedy-$arm.log
  [[ $arm == fp8 ]] && { python3 "$H/greedy.py" run --url http://$HN:$PORT/v1 --out $RAW/greedy-fp8-again.json > $RAW/greedy-fp8-again.log 2>&1; tail -1 $RAW/greedy-fp8-again.log; }
  [[ $arm != fp8-nospec ]] && quality $arm "intarsia serve (lane fp8kv dev) --backend cuda --max-seq 20480 --mem-budget 40 $X"
  rsh "grep -aE 'graph|eager|spec:|row-table' ~/intarsiaml-fp8kv/out/w3-$arm.log | sort | uniq -c | sort -rn | head -8" > $RAW/serve-$arm-notes.txt
  stop_server
done
for p in "bf16 fp8" "fp8 fp8-again" "fp8 fp8-nospec"; do echo "== greedy compare $p"; python3 "$H/greedy.py" compare $RAW/greedy-${p% *}.json $RAW/greedy-${p#* }.json; done | tee $RAW/greedy-compare.txt
echo "== done $(date +%T)"
