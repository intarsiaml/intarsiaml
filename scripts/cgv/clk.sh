#!/usr/bin/env bash
# Lane cgv: ON spark1 inside a fleetq slot. Samples SM clock / power / throttle reasons every 300 ms while a 192-token MTP-off decode runs (4k prompt).
cd ~/intarsiaml-cgv
(nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 300 > out/clk.csv &)
sleep 1
P=$(head -c 14035 trees/main/tools/prompts/corpus_docs.txt)
trees/main/target/release/intarsia generate -m models/Qwen3.8-27B-MXFP4 --backend cuda --mem-budget 45 --max-seq 8192 --chat --no-think --temperature 0 --ignore-eos --profile-every 0 "$@" -p "$P" --max-tokens 192 < /dev/null 2>&1 | grep finish | cut -c1-200
pkill -f "nvidia-smi --query"
awk -F, '{print $2,$3,$4,$5}' out/clk.csv | sort | uniq -c | sort -rn | head -20
