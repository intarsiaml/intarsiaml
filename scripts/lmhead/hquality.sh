#!/usr/bin/env bash
# Lane lmhead-fp8, ON tatooine inside a window (called by hdrive.sh `quality`, TP / BIN / DEC in the environment): the bf16 head's top-64 rows
# (self-golden) of the prompts-only golden ~/lmh-golden/prompts.safetensors (scripts/lmhead/mquality.sh), then the same binary with
# INTARSIA_LMHEAD_QUANT=fp8 / mxfp4 compared against it. Cards 0,1 at TP=2, card 0 at TP=1.
set -uo pipefail
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
b=lmhead-bin; [[ ${BIN:-head} == base ]] && b=lmhbase-bin
gpus=0; extra=""; [[ ${TP:-2} == 2 ]] && { gpus=0,1; extra="--tp 2 --devices 0,1"; }
run() { # run <env arg> <accuracy args...>
  local e=$1; shift; local nm=lmh-run-$RANDOM
  timeout -s KILL 1500 docker run --rm --name $nm --network host --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" \
    --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=$gpus $e -v "$HOME/$b":/b:ro -v /data/models:/models:ro -v "$HOME/lmh-golden":/g \
    intarsia-hip-dev:rocm7.2.4-rust1.90.0 /b/intarsia accuracy --backend hip --model $M27 --mem-budget 29 --max-seq 20480 $extra --mxfp4-act ${ACT:-none} \
    --mode both --decode-positions ${DEC:-128} --self-topk 64 --allow-same-code --allow-dirty --allow-unstamped "$@" < /dev/null
  local rc=$?; [[ $rc -eq 137 ]] && docker kill $nm >/dev/null 2>&1; return $rc
}
[[ ${SKIPWRITE:-0} == 1 ]] || { echo "QUALITY write bf16 head tp=${TP:-2} $(date +%T)"
run "" --golden /g/prompts.safetensors --write-self-golden /g/self-tp${TP:-2}.safetensors > ~/lmh-golden/write-tp${TP:-2}.out 2>&1; echo "QUALITY write exit $?"; }
for arm in ${ARMS:-fp8 mxfp4 off}; do
  # arm = Q or Q_chunkN (a perturbation-free reference: the bf16 head at another prefill chunk, to size the legitimate reordering noise)
  q=${arm%%_*}; xa=""; [[ $arm == *_chunk* ]] && xa="--prefill-chunk ${arm##*_chunk} --allow-config-mismatch"
  echo "QUALITY compare $arm tp=${TP:-2} $(date +%T)"
  sleep 20   # let the previous container give its cards back
  run "-e INTARSIA_LMHEAD_QUANT=$q" $xa --self-golden /g/self-tp${TP:-2}.safetensors --self-max-kl 1 --self-max-pos-kl 100 --self-max-flip-pct 100 --self-max-tail 1 --self-prompt-factor 1000 \
    > ~/lmh-golden/cmp-$arm-tp${TP:-2}.out 2>&1; echo "QUALITY compare $arm exit $?"
done
tail -n +1 ~/lmh-golden/write-tp${TP:-2}.out ~/lmh-golden/cmp-*-tp${TP:-2}.out
