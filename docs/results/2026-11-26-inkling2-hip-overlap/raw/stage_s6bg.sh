#!/usr/bin/env bash
# Card 1 of window S6 (lane ink-hip-overlap2): the Qwen-on-HIP no-op gate, Qwen3.6-35B-A3B-NVFP4 in intarsiaml's own container.
# (1) greedy md5 (`generate --spec off`, 256 tokens, short / 4k): main6 and o6, each with INTARSIA_OVERLAP unset and =1;
# (2) serve: o6's solo server (16 prompts one at a time), then `--max-batch 4` at 4 at once (4 waves): o6 unset, o6 =1, main6 =1;
# every batched reply is compared with o6's solo reply afterwards (serve_bitwise.py compare, on the Mac).
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s6; mkdir -p "$L"
for p in short 4k; do
  q35 1 main6 "" $p 256; q35 1 o6 "" $p 256; q35 1 o6 INTARSIA_OVERLAP=1 $p 256; q35 1 main6 INTARSIA_OVERLAP=1 $p 256
done
M=$Q
WAITS=90 srv 1 q-solo-o6 o6 --max-seq 8192 --mem-budget 29
[ "$UP" != 0 ] && bw q-solo-o6 solo
stop 1
for arm in "o6;;q-b4-o6" "o6;INTARSIA_OVERLAP=1;q-b4o-o6" "main6;INTARSIA_OVERLAP=1;q-b4o-main6"; do
  IFS=';' read -r b env tag <<< "$arm"
  (( $(left) < 40 )) && { echo "[$tag] SKIP ($(left) s left)"; continue; }
  SENV="$env" WAITS=90 srv 1 $tag $b --max-batch 4 --max-seq 8192 --mem-budget 29
  [ "$UP" != 0 ] && { bw $tag conc:4:0 conc:4:1 conc:4:2 conc:4:3; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total ' | sed "s/^/[$tag] /"; }
  grep -aE "overlap|twin|decode lane" "$L/$tag.log" | head -3 | sed "s/^/[$tag log] /"
  stop 1
done
M=/models/Inkling-2-Mini
echo "[bg] done $(now)"
