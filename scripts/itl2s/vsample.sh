#!/usr/bin/env bash
# 1 Hz: epoch second, vLLM requests running, waiting (the neighbour's load on this box). usage: vsample.sh <out file>
while true; do
  echo "$(date +%s) $(curl -s --max-time 2 localhost:8000/metrics | awk '/^vllm:num_requests_(running|waiting)/ {printf "%s ", $NF}')" >> "$1"
  sleep 1
done
