#!/usr/bin/env bash
# usage: clk.sh <outfile> : every 2 s append "epoch sm_mhz power_w util_pct temp_c" (clock guard sampler, runs on the Spark; stopped by PID)
f=$1
while true; do
  echo "$(date +%s) $(nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | tr -d ',')" >> "$f"
  sleep 2
done
