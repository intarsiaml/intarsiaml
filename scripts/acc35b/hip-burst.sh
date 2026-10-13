#!/usr/bin/env bash
# One HIP burst on tatooine for docs/results/2026-10-14-35b-accuracy.md (the scripts/hip-decode/burst.sh pattern, 9-minute
# cap). Runs ON tatooine inside a lease the caller took:
#   fleetq acquire --host tatooine --takes-down radiance --mem 30 --label ... --wait
#   FLEETQ_JOB=<id> ssh tatooine 'FLEETQ_JOB=<id> bash ~/intarsiaml-acc35b/scripts/acc35b/hip-burst.sh'
#   fleetq release <id>   (only after "radiance answers")
# Stops radiance, runs the HIP accuracy dumps (35b-short both modes, 35b-long decode only), restarts radiance, polls
# /v1/models (<= 3 min). Log with stop / start / answered times in ~/hipburst-logs/.
set -uo pipefail
LOGDIR=$HOME/hipburst-logs; mkdir -p "$LOGDIR"
LOG=$LOGDIR/acc35b-$(date +%m%d-%H%M%S).log
now() { date '+%F %T'; }
restart() {
  docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0 | xargs -r docker rm -f >/dev/null 2>&1
  echo "[burst] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  local i
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[burst] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"; return 0
    fi
    sleep 5
  done
  echo "[burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"; return 1
}
trap restart EXIT
export INTARSIA_REPO=$HOME/intarsiaml-acc35b HIP_VISIBLE_DEVICES=0
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh; B=/target/release/intarsia; M=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
O=/work/acc-out; mkdir -p "$INTARSIA_REPO/acc-out"
echo "[burst] job ${FLEETQ_JOB:-none}; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance; sleep 2
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
hrun() { # hrun <tag> <golden> <flags>
  local tag=$1 g=$2; shift 2
  echo "--- $tag $(now)" | tee -a "$LOG"
  timeout 240 $D env INTARSIA_ACC_POS_DUMP=$O/$tag.pos.tsv $B accuracy --backend hip --model $M --golden $G/$g --mem-budget 29 \
    --allow-unstamped --allow-dirty --prefill-chunk 256 --write-baseline $O/$tag.baseline.json --write-self-golden $O/$tag.dump.safetensors "$@" \
    > "$INTARSIA_REPO/acc-out/$tag.log" 2>&1
  echo "exit $?" | tee -a "$LOG"
  grep -E "verdict|^prefill |^decode" "$INTARSIA_REPO/acc-out/$tag.log" | head -6 | tee -a "$LOG"
}
for s in ${STAGES:-short long}; do
  case $s in
    short) hrun hip qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 16 --max-seq 4096 ;;
    # the golden every CUDA number uses (sha256 b6b51e1b...); tatooine's file without the suffix is an older vLLM run
    shortm) sha256sum /data/models/intarsia/golden/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors | tee -a "$LOG"
            hrun hip-meta qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors --mode both --decode-positions 16 --max-seq 4096 ;;
    long)  hrun hip-long qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode decode --decode-positions 64 --max-seq 16384 ;;
    longp) hrun hip-longpre qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode prefill --max-seq 16384 ;;
  esac
done
echo "[burst] stages done $(now)" | tee -a "$LOG"
