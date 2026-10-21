#!/usr/bin/env bash
# Lane G2b window 4 (ON tatooine, radiance down, called by window.sh):
#  1. HIP tp=1 no-op of the REBASED feat/35b-tp2 (tree dev2) against origin/main 76635aa (tree base2): stages-w1.sh with
#     NOOP_ONLY=1 into <out>/noop (plan, accuracy dumps + stdout, 256 ids, 8B and 35B).
#  2. Gate follow-ups on the window-1/2 dev tree (04306ee): self-golden tp=2 vs tp=1 with the provenance flags (window 2
#     lacked --pre-change), the 1 % bug on rank 1, tp=2 run-to-run (repeat vs window 1's tp=2 short dump).
#     (The tp=1 fusion-off reference, stages-nf.sh, ran in window 3.)
set -uo pipefail
O=${1:?out dir}
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
mkdir -p "$O/noop"
BASE_TREE=base2 DEV_TREE=dev2 NOOP_ONLY=1 bash $HOME/g2b/stages-w1.sh "$O/noop"
log "no-op stages exit $?"
# the rebased head at tp=2: 256 ids against window 1's tp=2 ids of 04306ee (does the rebase onto #42 move tp=2?)
for k in short 4k; do
  p='Write a long story about a robot who learns to paint.'; [[ $k == 4k ]] && p="$(cat $HOME/intarsiaml-g2b-dev2/scripts/hip-decode/prompt4k.txt)"
  INTARSIA_REPO=$HOME/intarsiaml-g2b-dev2 timeout 600 $HOME/intarsiaml-g2b-dev2/deploy/tatooine/dev.sh /work/g2b-intarsia generate \
    -m /models/Qwen3.6-35B-A3B-NVFP4 --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think --profile-every 0 --ignore-eos \
    --show-ids -p "$p" --max-tokens 256 --tp 2 --devices 0,1 > "$O/gen-dev2-35b-tp2-$k.out" 2> "$O/gen-dev2-35b-tp2-$k.err"
  grep -a '^ids:' "$O/gen-dev2-35b-tp2-$k.err" > "$O/gen-dev2-35b-tp2-$k.ids"
  cmp -s "$O/gen-dev2-35b-tp2-$k.ids" $HOME/g2b/out/w1/gen-dev-35b-tp2-$k.ids && log "IDS tp2 $k rebased == 04306ee" || log "IDS tp2 $k rebased != 04306ee"
done

R=$HOME/intarsiaml-g2b-dev
M35=/models/Qwen3.6-35B-A3B-NVFP4; G=/models/golden
SG=$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; LG=$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors
TP2="--tp 2 --devices 0,1"
PV="--pre-change 04306ee --allow-same-code --allow-config-mismatch"
BS="--self-max-kl 1.8e-2 --self-max-pos-kl 2.3 --self-max-flip-pct 3.76 --self-max-tail 2.8e-3 --self-prompt-factor 6.5"
BL="--self-max-kl 5.7e-2 --self-max-pos-kl 54.6 --self-max-flip-pct 8.7 --self-max-tail 7.1e-3 --self-prompt-factor 5.9"
# NF=1: INTARSIA_NO_FUSION=1 inside the container (dev.sh forwards only INTARSIA_HIP_* / INTARSIA_TP_*)
run() { # run <gpus> <timeout> <out-prefix> <intarsia args...>
  local gpus=$1 to=$2 pfx=$3; shift 3
  INTARSIA_REPO=$R HIP_VISIBLE_DEVICES=$gpus timeout "$to" "$R/deploy/tatooine/dev.sh" env INTARSIA_NO_FUSION=${NF:-0} /work/g2b-intarsia "$@" > "$O/$pfx.out" 2> "$O/$pfx.err"
  local rc=$?; log "$pfx exit $rc: $(grep -ahE '^golden gate|self-golden gate|SELF-STATS|^Error' "$O/$pfx.out" "$O/$pfx.err" | cut -c1-220 | tr '\n' '|')"
  return $rc
}
A="accuracy --backend hip --model $M35 --mem-budget 29 --max-seq 17408"
cp $HOME/g2b/out/w1/acc-dev-35b-tp2-short.safetensors $R/g2b-out/T2-short.safetensors
run 0,1 600 self-tp2-short $A --self-golden /work/g2b-out/S1-short.safetensors --mode both $PV $TP2 $BS
run 0,1 900 self-tp2-long $A --self-golden /work/g2b-out/S1-long.safetensors --mode both $PV $TP2 $BL
INTARSIA_TP_TEST_BUG=scale:1.01 run 0,1 600 self-tp2bug-short $A --self-golden /work/g2b-out/S1-short.safetensors --mode both $PV $TP2 $BS
INTARSIA_TP_TEST_BUG=scale:1.01 run 0,1 900 self-tp2bug-long $A --self-golden /work/g2b-out/S1-long.safetensors --mode both $PV $TP2 $BL
run 0,1 600 self-tp2rep-short $A --self-golden /work/g2b-out/T2-short.safetensors --mode both $PV $TP2
# localeval of the rebased head at tp=2 (window 3's sidecar did not run): serve, READY, wait for the Mac's DONE (<= 420 s)
R2=$HOME/intarsiaml-g2b-dev2
docker rm -f g2b-serve >/dev/null 2>&1
docker run -d --name g2b-serve --network host --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
  -e HIP_VISIBLE_DEVICES=0,1 -v $R2:/work -v /data/models/intarsia:/models:ro -w /work \
  intarsia-hip-dev:rocm7.2.4-rust1.90.0 /work/g2b-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 \
  --profile-every 0 --mem-budget 29 --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-g2b --max-seq 17408 \
  --tp 2 --devices 0,1 > /dev/null
up=0; for i in $(seq 1 80); do curl -sf -m 2 http://localhost:8090/health >/dev/null 2>&1 && { up=1; break; }; sleep 3; done
if [[ $up == 1 ]]; then
  log "tp2 server (rebased head) up for localeval"; touch "$O/READY"
  for i in $(seq 1 140); do [[ -f "$O/DONE" ]] && break; sleep 3; done
  log "localeval sidecar $( [[ -f $O/DONE ]] && echo done || echo 'timed out')"
else
  log "tp2 server did not come up"
fi
docker logs g2b-serve > "$O/serve-localeval.log" 2>&1
docker rm -f g2b-serve >/dev/null 2>&1
log "done"
