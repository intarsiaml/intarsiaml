#!/usr/bin/env bash
# Lane l5r7 gate window g1 (card 1): device bitwise/accuracy tests, no-op ids + plan show, self-golden logits identity, G-C. Each stage script is in the snapshot.
S=$HOME/intarsiaml-l5-$NAME/scripts/l5r7
for st in gc noop sg; do
  echo "######## stage $st $(date '+%F %T')"
  bash $S/stages-$st.sh
done
echo "######## g1 done $(date '+%F %T')"
