#!/bin/zsh
B=$(ls -t /opt/code/intarsiaml-spec/target/release/deps/zz_psotime-* | grep -v '\.d$' | head -1)
for i in 1 2 3 4 5; do
  echo "--- run $i with extras"; $B --nocapture 2>&1 | grep PSOTIME
  echo "--- run $i without extras"; PSO_SKIP_EXTRAS=1 $B --nocapture 2>&1 | grep PSOTIME
done
