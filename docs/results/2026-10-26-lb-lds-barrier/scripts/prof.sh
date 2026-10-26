#!/usr/bin/env bash
# Lane lb, ON tatooine inside a fleetq slot (--quiet gpu,cpu --takes-down radiance-rank0, card 0): rocprofv3 kernel traces of the 35B on
# the base and the dev binary, for the per-kernel before/after table (kagg.py). One server-less `generate` per (tree, mode):
#   pf4k  : prefill of the 4096-token hp prompt (--max-tokens 1)      pf16k : the 16384-token prompt (--max-tokens 1)
#   dec   : short prompt, 192 tokens (decode)                          dec4k : 4k prompt, 128 tokens (decode at 4k context)
# usage: prof.sh "<trees: base dev base dev>" "<modes>"   output: ~/intarsiaml-lb/logs/prof/<tree>-<mode>-<n>.csv (one per run)
set -uo pipefail
LB=$HOME/intarsiaml-lb
O=$LB/logs/prof; mkdir -p "$O"
trees=${1:-"base dev"}; modes=${2:-"pf4k pf16k dec dec4k"}
M35=/models/Qwen3.6-35B-A3B-NVFP4
SHORT='Write a long story about a robot who learns to paint.'
trap '$LB/dev/scripts/lb/win.sh restore' EXIT
$LB/dev/scripts/lb/win.sh stop
n=0
for t in $trees; do
  n=$((n + 1))
  for m in $modes; do
    case $m in
      pf4k)  p=$(cat $HOME/hp-prompts/prompt4096.txt); mt=1 ;;
      pf16k) p=$(cat $HOME/hp-prompts/prompt16384.txt); mt=1 ;;
      dec)   p=$SHORT; mt=192 ;;
      dec4k) p=$(cat $LB/dev/scripts/hip-decode/prompt4k.txt); mt=128 ;;
    esac
    d=/work/logs-prof/$t-$m-$n; rm -rf "$LB/$t/logs-prof/$t-$m-$n"; mkdir -p "$LB/$t/logs-prof"
    INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 INTARSIA_REPO=$LB/$t INTARSIA_TARGET_VOL=intarsia-cargo-lb-$t HIP_VISIBLE_DEVICES=0 \
      INTARSIA_REGISTRY_VOL=intarsia-cargo-lb-registry timeout -s KILL 400 $LB/dev/deploy/tatooine/dev.sh \
      rocprofv3 --kernel-trace --output-format csv -d $d -o run -- /target/release/intarsia generate -m $M35 --backend hip --mem-budget 29 \
      --max-seq 18432 --profile-every 0 --ignore-eos -p "$p" --max-tokens $mt > "$O/$t-$m-$n.log" 2>&1 < /dev/null
    csv=$(find "$LB/$t/logs-prof/$t-$m-$n" -name '*kernel_trace.csv' | head -1)
    if [ -n "$csv" ]; then cp "$csv" "$O/$t-$m-$n.csv"; rm -rf "$LB/$t/logs-prof/$t-$m-$n"; echo "[prof] $t $m #$n ok $(wc -l < "$O/$t-$m-$n.csv") rows"; else echo "[prof] $t $m #$n NO TRACE"; fi
  done
done
