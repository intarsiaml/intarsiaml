#!/bin/bash
# stage.sh ARM KIND   (KIND: lmmlu lgsm mmmlu mgsm); needs FLEETQ_JOB=7899 in env
cd /opt/code/intarsiaml/.worktrees/hrr
ARM=$1; KIND=$2; Q=.scratch/qa
$Q/serve_arm.sh $ARM $KIND > $Q/up-$ARM-$KIND.log 2>&1 &
for i in $(seq 1 80); do grep -q "^up" $Q/up-$ARM-$KIND.log 2>/dev/null && break; sleep 3; done
read S P PORT < $Q/arm-$ARM-$KIND.pids
REC="intarsia serve --backend metal --max-seq 20000 --spec off (head e64c9ddfb, INTARSIA_PREFIX_REUSE=$([ $ARM = B ] && echo exact || echo unset)), chunk 4096"
case $KIND in
 lmmlu) (cd /opt/code/deepeval && uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --thinking off --structured none --bench mmlu --limit 25 --label hrr-$ARM-mmlu --recipe "$REC") > $Q/out-$ARM-$KIND.txt 2>&1 ;;
 lgsm) (cd /opt/code/deepeval && uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --thinking off --structured none --bench gsm8k --limit 200 --label hrr-$ARM-gsm8k --recipe "$REC") > $Q/out-$ARM-$KIND.txt 2>&1 ;;
 mmmlu) (cd /opt/code/deepeval && .venv/bin/python /opt/code/intarsiaml/.worktrees/hrr/$Q/mt.py $PORT /opt/code/intarsiaml/.worktrees/hrr/$Q/mt-$ARM-mmlu.json mmlu) > $Q/out-$ARM-$KIND.txt 2>&1 ;;
 mgsm) (cd /opt/code/deepeval && .venv/bin/python /opt/code/intarsiaml/.worktrees/hrr/$Q/mt.py $PORT /opt/code/intarsiaml/.worktrees/hrr/$Q/mt-$ARM-gsm8k.json gsm8k) > $Q/out-$ARM-$KIND.txt 2>&1 ;;
esac
kill $S $P 2>/dev/null; sleep 3
echo "stage $ARM $KIND done" >> $Q/chain.log
