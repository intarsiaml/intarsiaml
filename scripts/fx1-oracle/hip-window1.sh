#!/usr/bin/env bash
# ON tatooine, one radiance-down window (trap restores radiance on any exit). fx1 window 1: acc-short-args accuracy
# runs of the 35B (old gate verdict against the vLLM short golden) with top-64 dumps for the oracle comparison:
# tp=1 clean, tp=1 row-parallel scale mutants x1.01 / x1.005, tp=2 clean, tp=2 INTARSIA_TP_TEST_BUG=scale:1.01.
set -uo pipefail
export R=$HOME/intarsiaml-fx1
LOG=$HOME/hipburst-logs/fx1-$(date +%m%d-%H%M%S).log; mkdir -p $HOME/hipburst-logs $R/fx1out
now() { date '+%F %T'; }
restart() {
  docker rm -f intarsia-serve >/dev/null 2>&1
  echo "[burst] starting radiance $(now)" | tee -a "$LOG"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[burst] radiance answers $(now) (after $((i * 5)) s)" | tee -a "$LOG"; return 0
    fi
    sleep 5
  done
  echo "[burst] RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a "$LOG"; return 1
}
trap restart EXIT
echo "[burst] job ${FLEETQ_JOB:-none}; fx1 window 1; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
stages() {
  cd $R
  export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-fx1
  D=deploy/tatooine/dev.sh; B=/target/release/intarsia; M=/models/Qwen3.6-35B-A3B-NVFP4
  # the canonical short golden (sha256 b6b51e1b...), copied into the tree: /models/golden holds another vLLM sample
  G=/work/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; O=/work/fx1out
  A="accuracy --backend hip --golden $G --mode both --decode-positions 16 --max-seq 4096 --mem-budget 29 --allow-unstamped --allow-dirty"
  run() { # name gpus model [extra...]
    local n=$1 g=$2 m=$3; shift 3
    echo "[fx1] $n start $(date +%T)"
    HIP_VISIBLE_DEVICES=$g $D $B $A --model $m --write-self-golden $O/hip-$n.dump.safetensors --write-baseline $O/hip-$n.baseline.json "$@" < /dev/null > $R/fx1out/hip-$n.log 2>&1
    echo "[fx1] $n exit $? $(date +%T)"; grep -E "floor \[|golden gate" $R/fx1out/hip-$n.log | cut -c1-260
  }
  run clean 0 $M
  run mut1.01 0 /work/mutants/mut1.01
  run mut1.005 0 /work/mutants/mut1.005
  run tp2-clean 0,1 $M --tp 2 --devices 0,1
  INTARSIA_TP_TEST_BUG=scale:1.01 run tp2-bug1.01 0,1 $M --tp 2 --devices 0,1
}
timeout "${BURST_CAP:-1080}" bash -c "$(declare -f stages); stages" 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
