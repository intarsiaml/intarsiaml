#!/usr/bin/env bash
# Lane H27P shared helpers, ON tatooine (sourced). Every GPU command runs in the dev image through deploy/tatooine/dev.sh with both cards.
R=$HOME/intarsiaml-h27p; V=intarsia-cargo-h27p; D=$R/deploy/tatooine/dev.sh
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
H=/data/models/radiance/hf-cache/hub
now() { date '+%F %T'; }
restore() {
  systemctl --user start radiance
  for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
  echo "[h27p] radiance $(systemctl --user is-active radiance) after $((i * 5)) s; models: $(curl -s -m 3 http://localhost:8080/v1/models | head -c 100)"
}
# gpu <timeout s> <cmd...>: run in the dev image, both cards, the lane's repo + target volume (the release CLI is /target/release/intarsia)
gpu() { local t=$1; shift
  INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V INTARSIA_MODELS=$H HIP_VISIBLE_DEVICES=${G:-0,1} timeout -s KILL "$t" $D "$@" < /dev/null
}
# gpup: same in the profiler image (rocprofv3)
gpup() { local t=$1; shift
  INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V INTARSIA_MODELS=$H HIP_VISIBLE_DEVICES=${G:-0,1} timeout -s KILL "$t" $D "$@" < /dev/null
}
# cfgargs <name>: the chunk arms of the 4k prompt
cfgargs() { case $1 in
  c256) echo "" ;;
  c512) echo "--prefill-chunk 512" ;;
  c1024) echo "--prefill-chunk 1024" ;;
  c2048) echo "--prefill-chunk 2048" ;;
  c4096) echo "--prefill-chunk 4096" ;;
  *) echo "unknown cfg $1" >&2; return 1 ;;
esac; }
