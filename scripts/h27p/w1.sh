#!/usr/bin/env bash
# Lane H27P window 1, ON tatooine inside ONE fleetq slot taking radiance down (both cards): the tp2 4k prefill profile and the chunk A/B.
#   usage: w1.sh [sections: T (timing, 3 reps per arm) P (rocprofv3 kernel trace per arm) O (engine op table, profile-every 1)]  default T P O
#   env: CFGS="c256 c2048 c4096"  ACT=fp8|none  PROMPT=4k-0  O=<out dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SECTIONS=${*:-T P O}; want() { [[ " $SECTIONS " == *" $1 "* ]]; }
CFGS=${CFGS:-"c256 c2048 c4096"}; ACT=${ACT:-fp8}; PROMPT=${PROMPT:-4k-0}
O=${O:-$HOME/intarsiaml-h27p-logs/w1}; mkdir -p "$O"
trap restore EXIT
systemctl --user stop radiance; sleep 3; echo "[h27p] radiance $(systemctl --user is-active radiance) $(now)"
P=$(cat $R/scripts/cmtp/ids-prompts/$PROMPT.txt)
GEN="generate -m $M27 --backend hip --tp 2 --devices 0,1 --mem-budget 29 --max-seq 8192 --mxfp4-act $ACT --spec off --ignore-eos --max-tokens 1"
if want T; then for c in $CFGS; do for rep in 1 2 3; do
  gpu 230 /target/release/intarsia $GEN --profile-every 0 -p "$P" $(cfgargs $c) > "$O/t-$c-$rep.out" 2> "$O/t-$c-$rep.err"
  echo "T $c rep$rep: $(grep -a 'prefill ' "$O/t-$c-$rep.err" | tail -1 | cut -c1-140)"
done; done; fi
if want O; then for c in ${OCFGS:-c256 c2048}; do
  gpu 230 /target/release/intarsia $GEN --profile-every 1 --op-table -p "$P" $(cfgargs $c) > "$O/o-$c.out" 2> "$O/o-$c.err"
  echo "O $c: $(grep -a 'prefill ' "$O/o-$c.err" | tail -1 | cut -c1-140)"
done; fi
if want P; then for c in $CFGS; do
  d=/work/h27p-prof/$c; rm -rf "$R/h27p-prof/$c"; mkdir -p "$R/h27p-prof/$c"
  gpup 235 rocprofv3 --kernel-trace --output-format csv -d $d -o run -- /target/release/intarsia $GEN --profile-every 0 -p "$P" $(cfgargs $c) > "$O/p-$c.out" 2> "$O/p-$c.err"
  csv=$(find "$R/h27p-prof/$c" -name '*kernel_trace.csv' | head -1)
  if [[ -n $csv ]]; then gzip -c "$csv" > "$O/p-$c.csv.gz"; rm -rf "$R/h27p-prof/$c"; fi
  echo "P $c: $(grep -a 'prefill ' "$O/p-$c.err" | tail -1 | cut -c1-140) csv=$([[ -s $O/p-$c.csv.gz ]] && echo yes || echo NO)"
done; fi
echo "[h27p] w1 done $(now)"
