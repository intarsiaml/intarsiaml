#!/bin/bash
# chunk4k window 1: plan show (all models), greedy-ids gate (4096 vs 2048, MTP, repeat), re-record Metal 35B baselines (gc2 recipe), verdict.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w1; mkdir -p $O
B=$W/.lane/intarsia-metal; M=$HOME/Models/Qwen3.6-35B-A3B-NVFP4; IP=$W/scripts/mpf/ids-prompts
export INTARSIA_REPO=$W
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/job.log; }
log "start $(git rev-parse --short HEAD)"
for m in Qwen3.6-35B-A3B-NVFP4 Qwen3-8B Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct Qwen3.8-27B-Quark-AWQ-MXFP4 MiniCPM-V-4.7-35B-A3B-MLX-4bit Kolibri-1-MLX-4bit; do
  $B plan show --backend metal --model $HOME/Models/$m --max-seq 8192 --json </dev/null > $O/plan-$m.json 2> $O/plan-$m.err
  log "plan $m exit $?: $(python3 -c "import json;d=json.load(open('$O/plan-$m.json'));print(d.get('prefill_chunk'),d.get('prefill_chunk_source'),d.get('scratch_bytes'))" 2>&1 | tail -1)"
done
for n in short-0 4k-0 4k-1 16k-0; do
  case $n in short*) MS=4096;; 4k*) MS=8192;; 16k*) MS=20480;; esac
  for arm in c4096 c2048 mtp; do
    case $arm in c4096) A="--spec off --repeat 2";; c2048) A="--spec off --prefill-chunk 2048";; mtp) A="--spec mtp --spec-k 2";; esac
    /usr/bin/time -l $B generate -m $M --backend metal --max-seq $MS --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
      -p "$(cat $IP/$n.txt)" --max-tokens 256 $A > $O/ids-$n-$arm.out 2> $O/ids-$n-$arm.err < /dev/null
    rc=$?
    grep -a '^ids:' $O/ids-$n-$arm.err | sort -u > $O/ids-$n-$arm.uniq
    log "ids $n $arm exit $rc: $(grep -a -E 'prefill chunk' $O/ids-$n-$arm.err | head -1 | cut -c1-80) | $(grep -a -E 'peak memory footprint|maximum resident' $O/ids-$n-$arm.err | tr -s ' ' | tr '\n' ';') | uniq-ids-lines $(wc -l < $O/ids-$n-$arm.uniq) md5 $(md5 -q $O/ids-$n-$arm.uniq)"
  done
  for a in c2048 mtp; do cmp -s $O/ids-$n-c4096.uniq $O/ids-$n-$a.uniq && log "  $n: c4096 == $a" || log "  $n: c4096 DIFFERS from $a"; done
done
# re-record (gc2 recipe): old files aside, q-0 records at the recipe chunk, install, q-1 verdict
G=$W/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; OR=$W/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
BD=golden/baselines; S35=qwen36-35b-a3b-nvfp4
Q="qualify --backend metal --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 40 --stage-timeout-min 25 --model $M"
qual() { local n=$1; shift; local t0=$(date +%s)
  $B $Q --out $O/q-$n "$@" < /dev/null > $O/q-$n.log 2>&1
  log "qualify $n exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$n.log | cut -c1-400 | tee -a $O/job.log; }
mkdir -p $O/old-baselines
for f in $BD/$S35-vllm-top20.metal.baseline.json $BD/$S35-vllm-top20.metal.envelope.json $BD/$S35-vllm-4k-top20.metal.baseline.json $BD/$S35-cpu-oracle.metal.selfgolden.safetensors; do mv $f $O/old-baselines/; done
qual 0-record --skip kernels --skip gate --skip bench --skip bench-long --skip ab --skip tp
B0=$O/q-0-record/baselines; ls -l $B0 | tee -a $O/job.log
cp $B0/acc-short.baseline.json $BD/$S35-vllm-top20.metal.baseline.json
cp $B0/envelope.json $BD/$S35-vllm-top20.metal.envelope.json
cp $B0/acc-4k.baseline.json $BD/$S35-vllm-4k-top20.metal.baseline.json
cp $B0/*.selfgolden.safetensors $BD/
git status --short $BD | tee -a $O/job.log
qual 1-verdict --skip envelope --skip bench-long --skip ab --skip tp
grep -a -hE "judged|skipped" $O/q-1-verdict/logs/acc-4k*.out 2>/dev/null | cut -c1-300 | head -10 | tee -a $O/job.log
log done
