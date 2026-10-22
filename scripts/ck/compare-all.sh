#!/usr/bin/env bash
# Lane ck: run chunkcmp.py on every dump directory under a backend's raw dir and save the text next to it.
#   compare-all.sh docs/results/2026-10-22-ck-prefill-chunk/raw/<backend>
set -uo pipefail
R=${1:?raw dir}; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$R/compare"
for f in $(find "$R" -path '*/acc/*' -name 'prefill-c2048.safetensors' -o -path '*/acc/*' -name 'decode-c2048.safetensors' | sort); do
  d=$(dirname "$f"); mode=$(basename "$f" | sed 's/-c2048.*//')
  tag=$(echo "$d" | sed -E 's#.*/(g[01]|[a-z0-9]+)/acc/#\1_#; s#.*/acc/##; s#/#_#g')
  case "$d" in */g0/*) pre=g0_;; */g1/*) pre=g1_;; *) pre="";; esac
  name="$(basename "$d")"; out="$R/compare/${mode}-${name}.txt"
  uv run --no-project --script "$HERE/chunkcmp.py" "$d" --mode "$mode" > "$out" 2>&1
  echo "$out: $(tail -1 "$out")"
done
