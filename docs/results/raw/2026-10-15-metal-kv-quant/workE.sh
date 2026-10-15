#!/bin/zsh
# one slot, three parts in priority order: F (final gates of the default path), E (speed, sizing, split sweep), C (16k+ proxy, localeval)
D=/opt/code/intarsiaml-kvq/docs/results/raw/2026-10-15-metal-kv-quant
echo "######## part F: final gates $(date)"; zsh $D/workF.sh
echo "######## part E: speed $(date)"; zsh $D/workE_body.sh
echo "######## part C: 16k proxy + localeval $(date)"; zsh $D/workC.sh
echo "######## all parts done $(date)"
