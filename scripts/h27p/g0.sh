#!/usr/bin/env bash
# Lane H27P G0, ON tatooine INSIDE a GPU window (win.sh g0.sh): "plan show for the 35B and the 8B is unchanged". scripts/hg/noop.sh on the 8B
# (Qwen3-8B) and the 35B (Qwen3.6-35B-A3B-NVFP4): `plan show` (text, json) and 256 greedy ids after a short and a ~3.5k-token prompt, for the
# origin/main binary (card 0) and this branch's binary (card 1) in parallel; then noop.sh cmp: every artifact must be identical.
set -uo pipefail
R=$HOME/intarsiaml-h27p; V=intarsia-cargo-h27p; D=$R/deploy/tatooine/dev.sh
O=${O:-$HOME/intarsiaml-h27p-logs/g0}; mkdir -p "$O"
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
run() { # run <tag> <binary> <gpu>
  RUN="env INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V HIP_VISIBLE_DEVICES=$3 $D $2" \
    bash $R/scripts/hg/noop.sh run "$O" "$1" hip $M8 $M35 $R/scripts/hg/prompts < /dev/null
}
run base /work/base-intarsia 0 > "$O/noop-base.log" 2>&1 &
run dev /work/h27p-intarsia 1 > "$O/noop-dev.log" 2>&1 &
wait
cat "$O/noop-base.log" "$O/noop-dev.log"
bash $R/scripts/hg/noop.sh cmp "$O" base dev
