#!/bin/zsh
# All GPU work of the long-context profile inside ONE exclusive fleetq slot (slot.sh sets FLEETQ_JOB). Essential runs first;
# the sweeps at the end are skipped after the deadline so the doc can be written by 15:30 EDT.
source /opt/code/intarsiaml-longctx/docs/results/raw/2026-10-15-metal-longctx-profile/common.sh
zmodload zsh/datetime
mkdir -p $R/plain $R/cb $R/kernel $R/clock $R/micro $R/localeval
P=$R/prompts
EX=$W/target/release/examples
CTXS=(4096 16384 32764 40000)
MM=$R/clock/macmon.jsonl
DEADLINE_SWEEP=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-15 14:15" +%s)
macmon_start $MM
trap macmon_stop EXIT
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git -C $W rev-parse HEAD)"; echo "binary: $(ls -l $BIN)"; shasum -a 256 $BIN
  echo "DEVELOPER_DIR=$DEVELOPER_DIR"; sw_vers | tr '\n' ' '; echo; /Users/josef/bin/macmon --version; pmset -g therm; pgrep -fl zoom.us; fleetq status --host endor | grep -E "RUNNING|WAIT"; } > $R/clock/versions.txt 2>&1
grep -q "RUNNING.*$FLEETQ_JOB\|#$FLEETQ_JOB.*running\|running" $R/clock/versions.txt || echo "WARNING: slot not seen RUNNING in fleetq status"
echo "idle start: $(idle3)" | tee $R/clock/idle.txt
printf "start\tend\tlabel\n" > $R/runs.tsv
sleep 2
stamp() { # <label> <command...>: run a command, record epoch start/end
  local lab=$1; shift
  local s=$EPOCHREALTIME
  "$@"
  printf "%.3f\t%.3f\t%s\n" $s $EPOCHREALTIME "$lab" >> $R/runs.tsv
}
run() { # <outstem> <ctx> <chunk> <max-tokens> [env assignment...]
  local stem=$1 ctx=$2 chunk=$3 mt=$4; shift 4
  stamp $(basename $stem) env "$@" $BIN generate --model $M --backend metal --max-seq 40960 --temperature 0 --max-tokens $mt --ignore-eos \
    --profile-every 0 --prefill-chunk $chunk --prompt "$(cat $P/prompt-$ctx.txt)" > $stem.out 2> $stem.err
  echo "$(basename $stem) [$(date +%H:%M:%S)]: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s) | decode [0-9]* ms ([0-9.]* tok/s)' $stem.err | tail -1) T=$(gpu_temp $MM)C"
}
before_deadline() { [ $(date +%s) -lt $DEADLINE_SWEEP ]; }

echo "=== warmup (page-in)"; run $R/plain/warm 4096 2048 8 NOENV=1

echo "=== 1. kernel mode, chunk 2048: prefill + 64 decode steps"
for c in $CTXS; do
  cooldown $MM "kernel-$c"
  run $R/kernel/ctx$c-c2048 $c 2048 65 INTARSIA_METAL_PROFILE=kernel:$R/kernel/ctx$c-c2048.jsonl
done

echo "=== 2. cb mode (GPU busy and idle), 129 tokens"
for c in $CTXS; do
  cooldown $MM "cb-$c"
  run $R/cb/ctx$c-c2048 $c 2048 129 INTARSIA_METAL_PROFILE=cb:$R/cb/ctx$c-c2048.jsonl
done

echo "=== 3. plain: 3 reps per context, chunk 2048, 129 forced tokens (128 decode steps)"
for r in 1 2 3; do
  for c in $CTXS; do
    cooldown $MM "plain-$c-r$r"
    run $R/plain/ctx$c-c2048-r$r $c 2048 129 NOENV=1
  done
done

echo "=== 4. localeval speed through intarsia serve (chunk 2048 recipe default), max-seq 40960"
stamp localeval-serve zsh -c '
  '$BIN' serve --model '$M' --backend metal --device 0 --host 127.0.0.1 --port 8791 --served-model-name intarsia-metal --max-seq 40960 --profile-every 0 --no-supervisor --no-ui > '$R'/localeval/server.out 2> '$R'/localeval/server.err &
  SPID=$!
  trap "kill $SPID 2>/dev/null; sleep 2" EXIT
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8791/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || { echo server died; exit 1; }; sleep 2; done
  echo server up
  cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8791/v1 --model intarsia-metal --sizes 128,4096,16384,32768 --gen 256 --reps 3 --thinking off --label metal-longctx > '$R'/localeval/localeval.out 2> '$R'/localeval/localeval.err
  echo localeval exit $?
'
tail -5 $R/localeval/localeval.out
sleep 3

echo "=== 5. microbenchmarks, default settings"
echo "--- flash_bench: nax prefill attention at chunk 2048 (36 layers, own KV cache each) and whole-prompt totals"
cooldown $MM "flash c2048"
stamp flash-c2048 $EX/flash_bench --chunk 2048 --reps 8 nax 0 2048 8192 16384 30720 38912 63488 2>&1 | tee -a $R/micro/flash-nax.txt
for tot in 4096 16384 32768 40000 65536; do
  echo "## total $tot chunk 2048" | tee -a $R/micro/flash-total.txt
  stamp flash-total-$tot $EX/flash_bench --chunk 2048 --total $tot --reps 4 nax 2>&1 | tee -a $R/micro/flash-total.txt
done
echo "--- attn_bw: decode split-KV, default policy"
for ctx in 128 1024 4096 8192 16384 32768 40000 65536; do
  cooldown $MM "attn_bw $ctx"
  echo "## ctx $ctx" | tee -a $R/micro/attn-bw-default.txt
  stamp attn-bw-$ctx $EX/attn_bw --reps 20 single splitkv $ctx 2>&1 | tee -a $R/micro/attn-bw-default.txt
done

echo "=== 6. kernel mode at other chunks (prefill only)"
for c in 4096 16384; do
  for ch in 512 1024 4096; do
    run $R/kernel/ctx$c-c$ch $c $ch 1 INTARSIA_METAL_PROFILE=kernel:$R/kernel/ctx$c-c$ch.jsonl
  done
done
cooldown $MM "kernel-32764-c1024"
run $R/kernel/ctx32764-c1024 32764 1024 1 INTARSIA_METAL_PROFILE=kernel:$R/kernel/ctx32764-c1024.jsonl
run $R/kernel/ctx32764-c4096 32764 4096 1 INTARSIA_METAL_PROFILE=kernel:$R/kernel/ctx32764-c4096.jsonl

echo "=== 7. sweeps (skipped after 14:15 EDT)"
if before_deadline; then
  echo "--- flash_bench chunk x pos0"
  for ch in 512 1024 4096; do
    cooldown $MM "flash c$ch"
    echo "## chunk $ch" | tee -a $R/micro/flash-nax.txt
    stamp flash-c$ch $EX/flash_bench --chunk $ch --reps 8 nax 0 2048 8192 16384 30720 38912 63488 2>&1 | tee -a $R/micro/flash-nax.txt
  done
fi
if before_deadline; then
  echo "--- prefill vs chunk (plain, 1 token), 2 rounds"
  for r in 1 2; do
    for c in 4096 16384 32764; do
      for ch in 256 512 1024 2048 4096; do
        [ $c = 32764 -a $ch = 256 ] && continue
        before_deadline || continue
        cooldown $MM "chunk-$c-$ch-r$r"
        run $R/plain/ctx$c-c$ch-pf-r$r $c $ch 1 NOENV=1
      done
    done
  done
fi
if before_deadline; then
  echo "--- attn_bw split policy sweep INTARSIA_METAL_SPLITKV=min_keys,target_tgs"
  for ctx in 4096 16384 32768 40000 65536; do
    for pol in 128,60 128,120 128,240 128,360 128,512 256,120 512,120 1024,120; do
      before_deadline || continue
      echo "## ctx $ctx policy $pol" | tee -a $R/micro/attn-bw-sweep.txt
      stamp attn-bw-sweep-$ctx-$pol env INTARSIA_METAL_SPLITKV=$pol $EX/attn_bw --reps 20 splitkv $ctx 2>&1 | tee -a $R/micro/attn-bw-sweep.txt
    done
  done
fi
echo "idle end: $(idle3)" | tee -a $R/clock/idle.txt
gzip -f $R/kernel/*.jsonl $R/cb/*.jsonl
echo "slotwork done $(date)"
