#!/usr/bin/env bash
# Lane H27P window 2, ON tatooine inside one fleetq slot taking radiance down: P2P bandwidth microbench + chunk 512/1024 timing.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
O=${O:-$HOME/intarsiaml-h27p-logs/w2}; mkdir -p "$O"
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[h27p] radiance $(systemctl --user is-active radiance) $(now)"
gpu 200 sh -c 'cd /work/scripts/h27p && hipcc -O2 --offload-arch=gfx1201 p2pbw.hip -o /tmp/p2pbw && /tmp/p2pbw' 2>&1 | tee "$O/p2pbw.txt"
P=$(cat $R/scripts/cmtp/ids-prompts/4k-0.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --spec off --ignore-eos --max-tokens 1 --profile-every 0"
for c in c512 c1024; do for rep in 1 2 3; do
  gpu 230 /target/release/intarsia $GEN -p "$P" $(cfgargs $c) > "$O/t-$c-$rep.out" 2> "$O/t-$c-$rep.err"
  echo "T $c rep$rep: $(grep -a 'prefill ' "$O/t-$c-$rep.err" | tail -1 | cut -c1-140)"
done; done
echo "[h27p] w2 done $(now)"
