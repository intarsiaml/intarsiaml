#!/usr/bin/env bash
# ON tatooine, one radiance-down window (trap restores radiance on any exit). fx1 window 2, lane-fx1 HIP build:
#  A. the engine against the committed CPU oracle (qualify's acc-oracle flags) with dumps: tp=1 clean on GPU 0 twice
#     and on GPU 1 (determinism), the x1.01 row-parallel scale mutant, tp=2 clean, tp=2 INTARSIA_TP_TEST_BUG=scale:1.01;
#  B. qualify (tp=2 stage included) on clean weights without a HIP baseline -> MISSING-BASELINE; it records the
#     acc-short baseline, the envelope and the tp=1 / tp=2 oracle dumps;
#  C. the recorded files installed in golden/baselines/ of this tree;
#  D. qualify clean, tp=2 included -> PASS;  E. qualify on the x1.01 mutant -> FAIL;
#  F. qualify clean with INTARSIA_TP_TEST_BUG=scale:1.01 and --tp 2 -> FAIL (tp stage).
set -uo pipefail
export R=$HOME/intarsiaml-fx1
export WS=${FX1_WS:-w2}   # output subdirectory; FX1_SKIP_A=1 skips part A (window 3 reuses window 2's tp=2 dump)
export FX1_SKIP_A=${FX1_SKIP_A:-0}
LOG=$HOME/hipburst-logs/fx1$WS-$(date +%m%d-%H%M%S).log; mkdir -p $HOME/hipburst-logs $R/fx1out/$WS
now() { date '+%F %T'; }
restart() {
  docker rm -f intarsia-serve >/dev/null 2>&1
  for c in $(docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0); do
    docker inspect --format '{{.Args}}' $c | grep -q 'fx1out' && docker kill $c >/dev/null
  done
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
echo "[burst] job ${FLEETQ_JOB:-none}; fx1 window 2; stopping radiance $(now)" | tee "$LOG"
systemctl --user stop radiance
sleep 2
echo "[burst] radiance stopped $(now)" | tee -a "$LOG"
stages() {
  cd $R
  export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-target-fx1
  D=deploy/tatooine/dev.sh; B=/target/release/intarsia; M=/models/Qwen3.6-35B-A3B-NVFP4
  G=/work/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors   # canonical short golden (sha256 b6b51e1b...)
  OR=/work/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
  O=/work/fx1out/$WS; H=$R/fx1out/$WS
  OF="--gate-class bf16 --max-mean-kl 2e-2 --min-top1 0 --no-ref-noise --model-id Qwen3.6-35B-A3B-NVFP4"
  A="accuracy --backend hip --golden $OR --mode prefill --decode-positions 16 --max-seq 4096 --mem-budget 29 --allow-unstamped --allow-dirty $OF"
  run() { # name gpus model [extra...]
    local n=$1 g=$2 m=$3; shift 3
    echo "[fx1] $n start $(date +%T)"
    HIP_VISIBLE_DEVICES=$g $D $B $A --model $m --write-self-golden $O/hip-$n.dump.safetensors "$@" < /dev/null > $H/hip-$n.log 2>&1
    echo "[fx1] $n exit $? $(date +%T)"; grep -E "floor \[|golden gate" $H/hip-$n.log | cut -c1-240
  }
  if [ "$FX1_SKIP_A" != 1 ]; then
    run oclean 0 $M
    run oclean-again 0 $M
    run oclean-gpu1 1 $M
    run omut1.01 0 /work/mutants/mut1.01
    run otp2-clean 0,1 $M --tp 2 --devices 0,1
    INTARSIA_TP_TEST_BUG=scale:1.01 run otp2-bug1.01 0,1 $M --tp 2 --devices 0,1
  fi
  Q="qualify --backend hip --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 29 --bench-sizes 128 --reps 1 --gen 16"
  SK="--skip kernels --skip bench --skip acc-long --skip acc-4k --skip bench-long --skip ab"
  qual() { # name gpus model [extra...]
    local n=$1 g=$2 m=$3; shift 3
    echo "[fx1] qualify $n start $(date +%T)"
    HIP_VISIBLE_DEVICES=$g $D $B $Q --model $m --out $O/q-$n "$@" < /dev/null > $H/q-$n.log 2>&1
    echo "[fx1] qualify $n exit $? $(date +%T)"
    grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $H/q-$n.log | cut -c1-260
  }
  qual 0-nobaseline 0,1 $M --tp 2 --devices 0,1 $SK
  mkdir -p golden/baselines
  Bd=$H/q-0-nobaseline/baselines
  cp $Bd/acc-short.baseline.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.baseline.json
  cp $Bd/envelope.json golden/baselines/qwen36-35b-a3b-nvfp4-vllm-top20.hip.envelope.json
  cp $Bd/qwen36-35b-a3b-nvfp4-cpu-oracle.hip.selfgolden.safetensors $Bd/qwen36-35b-a3b-nvfp4-cpu-oracle.hip-tp2.selfgolden.safetensors golden/baselines/
  echo "[fx1] installed: $(ls golden/baselines | grep 35b | tr '\n' ' ')"
  qual 1-clean 0,1 $M --tp 2 --devices 0,1 $SK --skip envelope
  qual 2-mut1.01 0 /work/mutants/mut1.01 $SK --skip envelope --skip tp
  INTARSIA_TP_TEST_BUG=scale:1.01 qual 3-tp2bug1.01 0,1 $M --tp 2 --devices 0,1 $SK --skip envelope
}
timeout "${BURST_CAP:-1140}" bash -c "$(declare -f stages); stages" 2>&1 | tee -a "$LOG"
echo "[burst] stages done $(now) (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
