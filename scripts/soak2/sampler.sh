#!/usr/bin/env bash
# soak2 copy of scripts/soak/sampler.sh plus the spec (MTP) counters.
# usage: sampler.sh <pid> <port> <outfile> [interval-s=60]: one TSV row per interval while <pid> lives (runs on the serving box)
pid=$1; port=$2; f=$3; iv=${4:-60}
KEYS=(
  intarsia_requests_total intarsia_requests_running intarsia_requests_waiting intarsia_requests_rejected_total
  intarsia_engine_failed intarsia_engine_no_progress_seconds
  'intarsia_requests_finished_total{reason="length"}' 'intarsia_requests_finished_total{reason="stop"}'
  intarsia_tokens_prompt_total intarsia_tokens_cached_total intarsia_tokens_generated_total
  intarsia_kv_bytes_used intarsia_device_memory_free_bytes
  intarsia_prefix_store_lookups_total intarsia_prefix_store_hits_total intarsia_prefix_store_misses_total
  intarsia_prefix_store_tokens_saved_total intarsia_prefix_store_inserts_total intarsia_prefix_store_rejected_total
  intarsia_prefix_store_entries
  'intarsia_prefix_store_tier_bytes{tier="host"}' 'intarsia_prefix_store_tier_budget_bytes{tier="host"}'
  'intarsia_prefix_store_tier_evictions_total{tier="host"}' 'intarsia_prefix_store_tier_objects{tier="host"}'
  intarsia_spec_steps_total intarsia_spec_proposed_tokens_total intarsia_spec_accepted_tokens_total
  intarsia_batch_itl_seconds_count intarsia_batch_paused_steps_total
  'intarsia_batch_itl_seconds_bucket{le="0.15"}' 'intarsia_batch_itl_seconds_bucket{le="0.25"}' 'intarsia_batch_itl_seconds_bucket{le="0.5"}'
)
{ printf 'time\tmem_avail_mb\tmem_used_mb\tswap_used_mb\trss_mb'; for k in "${KEYS[@]}"; do printf '\t%s' "$k"; done; echo; } > "$f"
while kill -0 "$pid" 2>/dev/null; do
  m=$(curl -s --max-time 5 "http://127.0.0.1:$port/metrics")
  row="$(date +%T)\t$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)\t$(free -m | awk '/Mem:/ {print $3}')\t$(free -m | awk '/Swap:/ {print $3}')\t$(( $(ps -o rss= -p "$pid" 2>/dev/null || echo 0) / 1024 ))"
  for k in "${KEYS[@]}"; do
    v=$(printf '%s\n' "$m" | awk -v k="$k" '$1==k {print $2; f=1} END {if (!f) print "NA"}')
    row="$row\t$v"
  done
  printf '%b\n' "$row" >> "$f"
  sleep "$iv"
done
