#!/usr/bin/env bash
# ON tatooine inside `fleetq run --host tatooine --exclusive --takes-down radiance`: the HIP no-op of T1. Stops radiance,
# runs greedy 256 tokens (short and 4k prompt, Qwen3-8B BF16, GPU 0) on base and dev, restarts radiance in an EXIT trap
# and waits for /v1/models. Hard cap 600 s on the GPU part. Logs stop / start / answer times to ~/t1/out/downtime.log.
set -uo pipefail
OUT=$HOME/t1/out; mkdir -p $OUT
LOG=$OUT/downtime.log
now() { date '+%F %T %Z'; }
restart() {
  echo "radiance start $(now)" | tee -a $LOG
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "radiance answers $(now) (after $((i * 5)) s)" | tee -a $LOG; return 0
    fi
    sleep 5
  done
  echo "RADIANCE DID NOT ANSWER within 180 s $(now)" | tee -a $LOG; return 1
}
trap restart EXIT
echo "job ${FLEETQ_JOB:-none} radiance stop $(now)" | tee -a $LOG
systemctl --user stop radiance
sleep 3
gpu() {
  for T in base dev; do
    for n in short 4k; do
      pf=docs/results/raw/2026-10-15-metal-decode-d1/prompt-short.txt
      [ $n = 4k ] && pf=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
      INTARSIA_REPO=$HOME/t1/$T INTARSIA_TARGET_VOL=intarsia-cargo-target-t1$T HIP_VISIBLE_DEVICES=0 \
        $HOME/t1/$T/deploy/tatooine/dev.sh bash -c "/target/release/intarsia generate -m /models/Qwen3-8B --backend hip --max-seq 8192 \
          --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --show-ids -p \"\$(cat $pf)\"" \
        > $OUT/hip-g256-$n-$T.out 2> $OUT/hip-g256-$n-$T.err
      echo "$T $n exit $? $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $OUT/hip-g256-$n-$T.err | tail -1)"
    done
  done
}
export -f gpu; export OUT
timeout 600 bash -c gpu
for n in short 4k; do cmp -s $OUT/hip-g256-$n-base.out $OUT/hip-g256-$n-dev.out && echo "HIP greedy 256 $n: byte-identical" || echo "HIP greedy 256 $n: DIFFER"; done
echo "gpu part done $(now)" | tee -a $LOG
