#!/usr/bin/env bash
# Lane G2b window 5 (ON tatooine, radiance down, called by window.sh): localeval of the rebased head (005ecf7) at tp=2,
# for the cross-engine comparison (windows 3 and 4 lost their localeval run to Mac-side sidecar bugs). Serves tp=2, writes
# READY, waits for the Mac sidecar's DONE (<= 420 s).
set -uo pipefail
O=${1:?out dir}
log() { echo "$(date +%T) $*" | tee -a "$O/steps.log"; }
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
