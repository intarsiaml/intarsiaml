#!/usr/bin/env bash
# Card 1 of window S8 (lane ink-hip-overlap2): the Qwen-on-HIP no-op gate, part 2: Qwen3.6-35B-A3B-NVFP4 `--max-batch 4` at 4 at once
# with INTARSIA_OVERLAP=1, o6 and main6 (vs S7's o6 solo server); then Inkling-2 G2 short / long on o6, and the hip tiny suites
# inkling2_batch / inkling2_packed on the o6 tree.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s8; mkdir -p "$L"
M=$Q
for arm in "o6;q-b4o-o6" "main6;q-b4o-main6"; do
  IFS=';' read -r b tag <<< "$arm"
  (( $(left) < 40 )) && { echo "[$tag] SKIP ($(left) s left)"; continue; }
  SENV="INTARSIA_OVERLAP=1" WAITS=60 srv 1 $tag $b --max-batch 4 --max-seq 8192 --mem-budget 29
  [ "$UP" != 0 ] && { bw $tag conc:4:0 conc:4:1 conc:4:2 conc:4:3; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total ' | sed "s/^/[$tag] /"; }
  grep -aE "overlap|twin|decode lane" "$L/$tag.log" | head -3 | sed "s/^/[$tag log] /"
  stop 1
done
M=/models/Inkling-2-Mini
for g in short long; do g2 1 o6 $g; done
stop 1
tiny 1 o6 inkling2_batch inkling2_packed
echo "[bg] done $(now)"
