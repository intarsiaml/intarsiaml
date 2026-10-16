#!/bin/zsh
# pilot, one slot: 8 reuse chains (script gateA.sh) + 17 spec cases at B 2 3 4 (script gateB.sh)
HERE=${0:A:h}
O=$1
CHAINS=8 $HERE/gateA.sh $O/A
IDS=$(python3 -c "print(','.join('s%03d'%i for i in range(17)))") BS="3 8" SCALE_HIGH=0.3 $HERE/gateB.sh $O/B
