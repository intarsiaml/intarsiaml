#!/usr/bin/env bash
# run inside a covering fleetq slot on spark2: compare SASS of ~/rp/base (origin/main merge base) and ~/rp/A/one (dev), identifier lines dropped
export PATH=/usr/local/cuda/bin:$PATH
norm() { cuobjdump -sass "$1" 2>/dev/null | sed -E 's/_GLOBAL__N__[0-9a-f]{8}/_GLOBAL__N__X/g' | grep -v '^identifier = '; }
nb=0; ns=0; nd=0; lines=0
for f in $(ls ~/rp/A/one/target/release/build/intarsia-backend-cuda-*/out/*.o); do
  b=$(ls ~/rp/base/target/release/build/intarsia-backend-cuda-*/out/$(basename $f) | head -1)
  nb=$((nb+1))
  if cmp -s <(norm $f) <(norm $b); then ns=$((ns+1)); else nd=$((nd+1)); echo "SASS DIFFERS: $(basename $f)"; fi
  lines=$((lines + $(norm $f | wc -l)))
done
echo "SASS compared $nb objects ($lines normalised lines in the dev objects): $ns identical, $nd differ (identifier lines and the GLOBAL__N hash normalised)"
