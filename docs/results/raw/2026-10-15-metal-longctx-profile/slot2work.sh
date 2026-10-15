#!/bin/zsh
# Second slot: the 32k rows the first slot missed (wrong prompt file name), plain runs with a clock-gated retry, plain-vs-cb A/B,
# and the NAX flash NSG experiment (patched binaries: nax-flash-s16-s32-experiment.patch, built in the scratchpad).
source /opt/code/intarsiaml-longctx/docs/results/raw/2026-10-15-metal-longctx-profile/common.sh
zmodload zsh/datetime
export COOL=70 COOLN=9
SC=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/lc
EXPBIN=$SC/target-exp/release/intarsia
EXPEX=$SC/target-exp/release/examples
D2=$R/s2
mkdir -p $D2/plain $D2/cb $D2/kernel $D2/exp $D2/clock
P=$R/prompts
MM=$D2/clock/macmon.jsonl
macmon_start $MM
trap macmon_stop EXIT
RT=$D2/runs.tsv
printf "start\tend\tlabel\n" > $RT
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "head: $(git -C $W rev-parse HEAD)"; echo "binary: $(ls -l $BIN)"; shasum -a 256 $BIN $EXPBIN
  echo "DEVELOPER_DIR=$DEVELOPER_DIR"; /Users/josef/bin/macmon --version; pmset -g therm; pmset -g | grep -i powermode; } > $D2/clock/versions.txt 2>&1
echo "idle start: $(idle3)" | tee $D2/clock/idle.txt
sleep 2
envlog() { { echo "== $1 $(date +%T)"; uptime | sed 's/.*load/load/'; ps -axo pcpu,comm | sort -nr | head -5 | awk '{printf "%s %s | ", $1, substr($0, index($0,$2), 60)}'; echo; ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; } >> $D2/envlog.txt; }
run() { # <bin> <outstem> <ctx> <chunk> <max-tokens> [env assignment...]
  local bin=$1 stem=$2 ctx=$3 chunk=$4 mt=$5; shift 5
  envlog $(basename $stem)
  local s=$EPOCHREALTIME
  env "$@" $bin generate --model $M --backend metal --max-seq 40960 --temperature 0 --max-tokens $mt --ignore-eos \
    --profile-every 0 --prefill-chunk $chunk --prompt "$(cat $P/prompt-$ctx.txt)" > $stem.out 2> $stem.err
  local e=$EPOCHREALTIME
  printf "%.3f\t%.3f\t%s\n" $s $e "$(basename $stem)" >> $RT
  local ck; ck=$(python3 $R/pclock_check.py $MM $s $e $stem.err); local ok=$?
  echo "$(basename $stem) [$(date +%H:%M:%S)]: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s) | decode [0-9]* ms ([0-9.]* tok/s)' $stem.err | tail -1) clock(med p10 %<1450 W C)=[$ck] clockok=$ok"
  return $ok
}
plain_gated() { # <bin> <stemprefix> <ctx> <chunk> <max-tokens> [env...]: up to 3 attempts until the prefill clock is >= 1450 MHz
  local bin=$1 pre=$2 ctx=$3 ch=$4 mt=$5; shift 5
  for a in 1 2 3; do
    cooldown $MM "$pre-a$a"
    run $bin $pre-a$a $ctx $ch $mt "$@" && return 0
  done
  return 1
}

echo "=== warmup"; run $BIN $D2/plain/warm 4096 2048 8 NOENV=1
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-15 14:20" +%s)
before_deadline() { [ $(date +%s) -lt $DEADLINE ]; }

micro() { # <outfile> <label> <cmd...>: short GPU microbenchmark, cooled first, clock-gated, up to 3 attempts; every attempt is kept
  local out=$1 lab=$2; shift 2
  for a in 1 2 3; do
    cooldown $MM "$lab-a$a"
    local tmp=$D2/clock/tmp.$$; local s=$EPOCHREALTIME
    "$@" > $tmp 2>&1
    local e=$EPOCHREALTIME
    printf "%.3f\t%.3f\t%s-a%s\n" $s $e "$lab" $a >> $RT
    local ck; ck=$(python3 $R/wclock.py $MM $s $e); local ok=$?
    { echo "## $lab attempt $a clock(med p10 %<1450 W C n)=[$ck] clockok=$ok"; cat $tmp; } | tee -a $out
    [ $ok = 0 ] && return 0
  done
  return 1
}
FB=$W/target/release/examples/flash_bench
AB=$W/target/release/examples/attn_bw

echo "=== A. 32k kernel + cb traces (missed in slot 1)"
for a in 1 2; do
  cooldown $MM kernel-32768
  run $BIN $D2/kernel/ctx32768-c2048-a$a 32768 2048 65 INTARSIA_METAL_PROFILE=kernel:$D2/kernel/ctx32768-c2048-a$a.jsonl && break
done
for a in 1 2; do
  cooldown $MM cb-32768
  run $BIN $D2/cb/ctx32768-c2048-a$a 32768 2048 129 INTARSIA_METAL_PROFILE=cb:$D2/cb/ctx32768-c2048-a$a.jsonl && break
done

echo "=== B. decode split-KV attention bandwidth, default policy (gated)"
for ctx in 128 1024 4096 8192 16384 32768 40000 65536; do
  micro $D2/exp/attn-bw-default.txt attn_bw-$ctx $AB --reps 12 single splitkv $ctx
done

echo "=== C. plain runs, clock-gated: 3 per context, chunk 2048, 129 forced tokens"
for r in 1 2 3; do
  for c in 4096 16384 32768 40000; do
    plain_gated $BIN $D2/plain/ctx$c-c2048-r$r $c 2048 129 NOENV=1
  done
done

echo "=== D. prefill attention (NAX flash) per position and chunk, default s4, one gated invocation per point"
for ch in 2048 512 1024 4096; do
  for p0 in 0 2048 8192 16384 30720 38912 63488; do
    before_deadline || break
    micro $D2/exp/flash-s4.txt flash-c$ch-p$p0 $FB --chunk $ch --reps 5 nax $p0
  done
done

echo "=== E. NAX flash NSG experiment (patched build): s4 s8 s16 s32, chunk 2048, interleaved by position"
for p0 in 8192 30720 63488 0 16384 38912 2048; do
  for v in attn_nax_d128_s4 attn_nax_d128_s8 attn_nax_d128_s16 attn_nax_d128_s32 attn_nax_d128_s32 attn_nax_d128_s16 attn_nax_d128_s8 attn_nax_d128_s4; do
    before_deadline || break
    micro $D2/exp/flash-variants.txt "fv-$v-p$p0" $EXPEX/flash_bench --chunk 2048 --reps 5 $v $p0
  done
done

echo "=== F. decode split policy sweep (gated), min_keys,target_tgs"
for ctx in 16384 32768 40000 65536 4096; do
  for pol in 128,60 128,120 128,240 128,360 128,512 256,120 512,120 1024,120 128,120; do
    before_deadline || break
    micro $D2/exp/attn-bw-sweep.txt "sweep-$ctx-$pol" env INTARSIA_METAL_SPLITKV=$pol $AB --reps 12 splitkv $ctx
  done
done

echo "=== G. NSG variants in the engine: plain prefill (1 token), order s4 s8 s16 s32 s32 s16 s8 s4, 16k and 40k"
for c in 16384 40000; do
  for v in s4 s8 s16 s32 s32 s16 s8 s4; do
    before_deadline || break
    cooldown $MM "g-$c-$v"
    run $EXPBIN $D2/exp/ctx$c-$v-$RANDOM $c 2048 1 INTARSIA_METAL_FLASH_NAX=attn_nax_d128_$v
  done
done
echo "--- greedy ids at 4k, 48 tokens, per variant"
for v in s4 s8 s16 s32; do
  INTARSIA_METAL_FLASH_NAX=attn_nax_d128_$v $EXPBIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 48 --ignore-eos --show-ids --profile-every 0 --prompt "$(cat $P/prompt-4096.txt)" > $D2/exp/ids-$v.out 2> $D2/exp/ids-$v.err
done

echo "=== H. kernel mode at other chunks (prefill only), 16k"
for ch in 512 1024 4096; do
  before_deadline || break
  cooldown $MM "kernel-16384-c$ch"
  run $BIN $D2/kernel/ctx16384-c$ch 16384 $ch 1 INTARSIA_METAL_PROFILE=kernel:$D2/kernel/ctx16384-c$ch.jsonl
done
echo "idle end: $(idle3)" | tee -a $D2/clock/idle.txt
gzip -f $D2/kernel/*.jsonl $D2/cb/*.jsonl 2>/dev/null
echo "slot2work done $(date)"
