#!/usr/bin/env bash
# Lane W4A8, ON tatooine inside a fleetq job that holds both cards and has taken radiance down (final window): gate 5, "G0 for the other
# models unchanged on HIP". scripts/hg/noop.sh run on the 8B (Qwen3-8B) and the 35B (Qwen3.6-35B-A3B-NVFP4): `plan show` (text, json)
# and 256 greedy ids after a short and a ~3.5k-token prompt, for the origin/main binary (card 0) and this branch's binary (card 1) in
# parallel; then noop.sh cmp: every artifact must be identical. Binaries (copied into the repo copy by build-g0.sh):
#   ~/intarsiaml-w4a8/base-intarsia (origin/main)   ~/intarsiaml-w4a8/w4a8-intarsia (this branch)
# The trap brings radiance back and waits until it answers (cat-window.sh stops it again for the next part).
set -uo pipefail
R=$HOME/intarsiaml-w4a8; V=intarsia-cargo-w4a8; D=$R/deploy/tatooine/dev.sh
O=${O:-$HOME/intarsiaml-w4a8-logs/g0}; mkdir -p "$O"
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
restore() {
  systemctl --user start radiance
  for i in $(seq 1 48); do curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\[' && break; sleep 5; done
  echo "[g0] radiance $(systemctl --user is-active radiance) after $((i * 5)) s"
}
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[g0] radiance $(systemctl --user is-active radiance) $(date +%T)"
run() { # run <tag> <binary> <gpu>
  RUN="env INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=$3 $D $2" \
    bash $R/scripts/hg/noop.sh run "$O" "$1" hip $M8 $M35 $R/scripts/hg/prompts < /dev/null
}
run base /work/base-intarsia 0 > "$O/noop-base.log" 2>&1 &
run dev /work/w4a8-intarsia 1 > "$O/noop-dev.log" 2>&1 &
wait
cat "$O/noop-base.log" "$O/noop-dev.log"
bash $R/scripts/hg/noop.sh cmp "$O" base dev
echo "[g0] done $(date +%T)"
