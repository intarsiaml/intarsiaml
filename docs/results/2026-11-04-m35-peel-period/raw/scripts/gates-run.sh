#!/bin/zsh
# gates 1, 2 (ids), 3, 4 (plan show, kernels), 5 (MTP) in one slot
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel-period
S=$PWD/.scratch; M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
LENS=2050,2052,2055,2056,2057,6146,6153,258,260,265,516,772,1028,1540,4356,2058,2064,2300,6154,48,233,266,4099,4101,4105,5770
echo "#### ids"; rm -rf $S/logs/ids35
BG_MAX=1200 BG_SLEEP=1 zsh $S/ids-run.sh $S/logs/ids35 $S/logs/gates/prompts-all.json $LENS $M35 18481 main:$S/intarsia.main peel:$S/intarsia.peel
python3 $S/gates-ids.py $S/logs/ids35 $S/logs/ids35/ids.json | tee $S/logs/ids35/gates-1-2.txt
echo "#### plan show + kernels"; zsh $S/gates-a.sh
echo "#### mtp"; zsh $S/gates-mtp.sh "2050 2057 6146"
