#!/usr/bin/env bash
# Card 1 of window S7 (lane ink-hip-overlap2): the Qwen-on-HIP no-op gate, part 1, Qwen3.6-35B-A3B-NVFP4 in intarsiaml's own container:
# greedy md5 (`generate --spec off`, 256 tokens, short / 4k) of main6 and o6, each with INTARSIA_OVERLAP unset and =1; o6's solo
# server (the 16 prompts one at a time); o6 `--max-batch 4` with INTARSIA_OVERLAP unset at 4 at once (4 waves). Compared afterwards.
set -uo pipefail
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib.sh"
. "$HOME/intarsiaml-ink-hip-bpf/.scratch/stage_lib2.sh"
L=$REPO/logs/s7; mkdir -p "$L"
for p in short 4k; do
  q35 1 main6 "" $p 256; q35 1 o6 "" $p 256; q35 1 o6 INTARSIA_OVERLAP=1 $p 256; q35 1 main6 INTARSIA_OVERLAP=1 $p 256
done
M=$Q
WAITS=90 srv 1 q-solo-o6 o6 --max-seq 8192 --mem-budget 29
[ "$UP" != 0 ] && bw q-solo-o6 solo
stop 1
if (( $(left) > 45 )); then
  WAITS=60 srv 1 q-b4-o6 o6 --max-batch 4 --max-seq 8192 --mem-budget 29
  [ "$UP" != 0 ] && { bw q-b4-o6 conc:4:0 conc:4:1 conc:4:2 conc:4:3; curl -sf -m 5 "http://$IP:8133/metrics" | grep -E '^intarsia_batch_overlap_steps_total ' | sed 's/^/[q-b4-o6] /'; }
  stop 1
fi
M=/models/Inkling-2-Mini
echo "[bg] done $(now)"
