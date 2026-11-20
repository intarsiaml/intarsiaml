#!/usr/bin/env bash
# memgrow sampler (runs on the Spark). usage: sampler.sh <pid> <port> <outfile.csv> [interval-s=30]
# One CSV row per interval while <pid> lives: /proc/meminfo (MB), the process's status + smaps_rollup (MB), the vLLM
# container's cgroup (MB) and main / EngineCore RSS, and the store / engine counters from the intarsia server's /metrics.
# <pid> is the loop pid (a keepalive); the process to inspect is read from the file named by env IPIDFILE on every tick (empty / missing: NA).
# Env VLLM_CG = the vLLM container's cgroup dir (default: found from the missioncntrl-guidance container's main pid).
pid=$1; port=$2; f=$3; iv=${4:-30}
CG=${VLLM_CG:-/sys/fs/cgroup$(cut -d: -f3 /proc/$(docker inspect -f '{{.State.Pid}}' missioncntrl-guidance)/cgroup)}
MI=(MemFree MemAvailable Buffers Cached 'Active(anon)' 'Inactive(anon)' 'Active(file)' 'Inactive(file)' Unevictable Mlocked AnonPages Mapped Shmem Dirty Writeback Slab SReclaimable SUnreclaim KernelStack PageTables Percpu VmallocUsed AnonHugePages SwapFree)
ST=(VmRSS RssAnon RssFile RssShmem VmSwap VmData Threads)
SM=(Pss Pss_Anon Pss_File Pss_Shmem Private_Dirty Locked AnonHugePages)
CGK=(anon file shmem file_mapped file_dirty slab_reclaimable slab_unreclaimable kernel)
KEYS=(
  intarsia_requests_total intarsia_requests_running intarsia_engine_failed
  intarsia_tokens_generated_total intarsia_kv_bytes_used intarsia_device_memory_free_bytes intarsia_weights_bytes
  intarsia_prefix_store_entries intarsia_prefix_store_inserts_total intarsia_prefix_store_insert_bytes_total intarsia_prefix_store_bytes_restored_total
  'intarsia_prefix_store_tier_bytes{tier="device"}' 'intarsia_prefix_store_tier_bytes{tier="host"}' 'intarsia_prefix_store_tier_bytes{tier="disk"}'
  'intarsia_prefix_store_tier_objects{tier="device"}' 'intarsia_prefix_store_tier_objects{tier="host"}' 'intarsia_prefix_store_tier_objects{tier="disk"}'
  'intarsia_prefix_store_tier_evictions_total{tier="device"}' 'intarsia_prefix_store_tier_evictions_total{tier="host"}' 'intarsia_prefix_store_tier_evictions_total{tier="disk"}'
  'intarsia_prefix_store_tier_budget_bytes{tier="host"}' intarsia_prefix_store_pinned_entries
)
{ printf 'epoch'
  for k in "${MI[@]}"; do k=${k//\(/_}; printf ',mi_%s' "${k//\)/}"; done
  for k in "${ST[@]}"; do printf ',st_%s' "$k"; done
  for k in "${SM[@]}"; do printf ',sm_%s' "$k"; done
  printf ',ipid,fds,vllm_cg_current'; for k in "${CGK[@]}"; do printf ',vllm_cg_%s' "$k"; done
  printf ',vllm_rss_main,vllm_rss_engine,gpu_mb_intarsia,gpu_mb_vllm,gpu_mb_all'
  for k in "${KEYS[@]}"; do printf ',%s' "$(printf '%s' "$k" | sed 's/^intarsia_//; s/[{}"=]/_/g')"; done; echo; } > "$f"
# one awk helper: pick named fields of a file as ",v1,v2,..." (value / divisor, NA when missing)
pick() { # pick <file> <divisor-kB|B> key...   (keys as in the file's first column, with the trailing colon for /proc files)
  local file=$1 div=$2; shift 2
  if [ ! -r "$file" ]; then local n; n=$(awk -v keys="$*" 'BEGIN{print split(keys,K,"|")}'); local o=""; for ((i=0;i<n;i++)); do o="$o,NA"; done; printf '%s' "$o"; return; fi
  awk -v keys="$*" -v div="$div" 'BEGIN{n=split(keys,K,"|"); for(i=1;i<=n;i++) w[K[i]]=i} {if($1 in w) v[w[$1]]=($1=="Threads:"?$2:int($2/div))} END{for(i=1;i<=n;i++) printf ",%s", (i in v?v[i]:"NA")}' "$file" 2>/dev/null
}
join_keys() { local IFS='|'; echo "$*"; }
miK=$(join_keys "${MI[@]/%/:}"); stK=$(join_keys "${ST[@]/%/:}"); smK=$(join_keys "${SM[@]/%/:}"); cgK=$(join_keys "${CGK[@]}")
while kill -0 "$pid" 2>/dev/null; do
  ep=$(date +%s)
  mi=$(pick /proc/meminfo 1024 "$miK")
  sup=$(cat "${IPIDFILE:-/nonexistent}" 2>/dev/null); ip=0   # the file holds the supervisor's pid; the engine is its child (the supervisor's own RSS is tiny: ab35d's 162 MB column)
  if [ -n "$sup" ] && [ -d /proc/$sup ]; then ip=$(pgrep -P "$sup" | head -1); [ -n "$ip" ] || ip=$sup; fi
  st=$(pick /proc/$ip/status 1024 "$stK")
  sm=$(pick /proc/$ip/smaps_rollup 1024 "$smK")
  fds=$(ls /proc/$ip/fd 2>/dev/null | wc -l)
  cur=$(( $(cat "$CG/memory.current" 2>/dev/null || echo 0) / 1048576 ))
  cg=$(pick "$CG/memory.stat" 1048576 "$cgK")
  vr=$(ps -eo rss,cmd 2>/dev/null | awk '/local\/bin\/vllm/ && !/awk/ {m=int($1/1024)} /VLLM::EngineCore/ && !/awk/ {e=int($1/1024)} END{printf ",%s,%s", m+0, e+0}')
  ga=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  vep=$(pgrep -f VLLM::EngineCore | head -1)
  gp=$(printf '%s\n' "$ga" | awk -F, -v ip="$ip" -v vp="${vep:-0}" '{a+=$2} $1==ip {i+=$2} $1==vp {v+=$2} END{printf ",%d,%d,%d", i, v, a}')
  m=$(curl -s --max-time 5 "http://127.0.0.1:$port/metrics")
  mv=""; for k in "${KEYS[@]}"; do mv="$mv,$(printf '%s\n' "$m" | awk -v k="$k" '$1==k {print $2; f=1} END {if (!f) print "NA"}')"; done
  printf '%s%s%s%s,%s,%s,%s%s%s%s%s\n' "$ep" "$mi" "$st" "$sm" "$ip" "$fds" "$cur" "$cg" "$vr" "$gp" "$mv" >> "$f"
  sleep "$iv"
done
