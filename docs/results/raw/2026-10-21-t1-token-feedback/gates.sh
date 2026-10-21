#!/bin/zsh
# T1 Metal gates + speed, one exclusive endor slot (--mem 24). base = origin/main build, dev = this branch;
# "fb" = dev with INTARSIA_TOKEN_FEEDBACK=1, "off" = dev with the default (feedback off on Metal until decided).
#  1 Metal tests              2 greedy 256 ids (short, 4k; profile-every 0 and 16)   3 stop conditions on the real model
#  4 cb traces (step gap)     5 goldens                                              6 plain-generate 4k A/B x3 + macmon
#  7 served bench A/B x3      8 short EOS-terminated chat requests back to back      9 qualify (fb)
# usage (inside slot): zsh gates.sh <out-dir> [steps...]   (steps default: all)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; shift; mkdir -p $O
STEPS=${*:-"1 2 3 4 5 6 7 8 9"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/t1
W=/opt/code/intarsiaml/.claude/worktrees/agent-a06622c5d21b6b82d
RD=$W/docs/results/raw/2026-10-21-t1-token-feedback
P=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
PS=$W/docs/results/raw/2026-10-15-metal-decode-d1/prompt-short.txt
M=/Users/josef/models/Qwen3-8B; G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
B=$S/intarsia-base; D=$S/intarsia-dev
[ -x $B ] && [ -x $D ] || { echo "binaries missing (run build.sh first)" >&2; exit 3; }
MPID=""; SPID=""
trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null' EXIT
echo "base $(md5 -q $B) dev $(md5 -q $D) steps [$STEPS] $(date -Iseconds)" | tee -a $O/info.txt
# arm -> (binary, env)
bin() { [[ $1 == base* ]] && echo $B || echo $D; }
fbenv() { [[ $1 == fb* ]] && echo 1 || echo 0; }
gen() { # arm out-name extra-args...
  local arm=$1 name=$2; shift 2
  env INTARSIA_TOKEN_FEEDBACK=$(fbenv $arm) $(bin $arm) generate --model $M --backend metal --max-seq 20000 --temperature 0 --show-ids "$@" \
    > $O/$name.out 2> $O/$name.err
  echo "$name exit $? $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $O/$name.err | tail -1)"
}

if has 1; then
  echo "== 1 Metal tests $(date -Iseconds)"
  ( cd $W && INTARSIA_TEST_REAL_MODEL=1 cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $O/cargo-test-metal.txt 2>&1 ); echo "metal crate exit $?"
  ( cd $W && cargo test --workspace --features intarsia-cli/metal > $O/cargo-test-workspace-metal.txt 2>&1 ); echo "workspace+metal exit $?"
  for f in $O/cargo-test-metal.txt $O/cargo-test-workspace-metal.txt; do
    grep -h "test result" $f | awk -v f=${f:t} '{p+=$4; x+=$6} END {print f, "passed", p, "failed", x}'; done
fi

if has 2; then
  echo "== 2 greedy 256 $(date -Iseconds)"
  for f in short:$PS 4k:$P/prompt-4096.txt; do
    n=${f%%:*}; pf=${f#*:}
    for arm in base off fb; do gen $arm g256-$n-$arm --profile-every 0 --max-tokens 256 --ignore-eos --prompt "$(cat $pf)"; done
    for arm in off fb; do cmp -s $O/g256-$n-base.out $O/g256-$n-$arm.out && echo "greedy 256 $n $arm: byte-identical" || echo "greedy 256 $n $arm: DIFFER"; done
  done
  # profiled steps (every 16th, the generate default) interleave with feedback steps
  for arm in base fb; do gen $arm g256p16-short-$arm --profile-every 16 --max-tokens 256 --ignore-eos --prompt "$(cat $PS)"; done
  cmp -s $O/g256p16-short-base.out $O/g256p16-short-fb.out && echo "greedy 256 short profile-every 16 fb: byte-identical" || echo "profile-every 16: DIFFER"
fi

if has 3; then
  echo "== 3 stop conditions $(date -Iseconds)"
  Q="List three primary colors, comma separated, nothing else."
  for arm in base fb; do
    gen $arm stop-eos-$arm --profile-every 0 --chat --no-think --max-tokens 200 --prompt "$Q"
    gen $arm stop-max-$arm --profile-every 0 --chat --no-think --max-tokens 7 --prompt "$Q"
    gen $arm stop-str-$arm --profile-every 0 --chat --no-think --max-tokens 200 --stop "," --prompt "$Q"
    gen $arm stop-min-$arm --profile-every 0 --chat --no-think --max-tokens 200 --min-tokens 12 --prompt "$Q"
    gen $arm stop-long-$arm --profile-every 0 --chat --no-think --max-tokens 400 --prompt "Write two sentences about the sea."
  done
  for k in eos max str min long; do
    cmp -s $O/stop-$k-base.out $O/stop-$k-fb.out && echo "stop $k: byte-identical ($(wc -w < $O/stop-$k-fb.out) words)" || echo "stop $k: DIFFER"
    grep -ho 'finish[^,]*' $O/stop-$k-fb.err | tail -1
  done
fi

if has 4; then
  echo "== 4 cb traces, 4k + short, 129 tokens, interleaved $(date -Iseconds)"
  for r in 1 2 3; do
    for arm in base fb fb base; do
      for f in 4k:$P/prompt-4096.txt short:$PS; do
        n=${f%%:*}; pf=${f#*:}; t=cb-$n-$arm-$r-$RANDOM
        INTARSIA_METAL_PROFILE=cb:$O/$t.jsonl gen $arm $t --profile-every 0 --max-tokens 129 --ignore-eos --prompt "$(cat $pf)" > /dev/null
        python3 $RD/cbgap.py $O/$t.jsonl 128 | sed "s|^|$t |"; gzip -f $O/$t.jsonl
      done
    done
  done
fi

if has 5; then
  echo "== 5 goldens $(date -Iseconds)"
  for arm in base fb; do
    env INTARSIA_TOKEN_FEEDBACK=$(fbenv $arm) $(bin $arm) accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > $O/acc-short-$arm.log 2>&1; echo "acc short $arm exit $?"
    env INTARSIA_TOKEN_FEEDBACK=$(fbenv $arm) $(bin $arm) accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 16 --max-seq 8192 > $O/acc-long-$arm.log 2>&1; echo "acc long $arm exit $?"
  done
  for k in short long; do cmp -s $O/acc-$k-base.log $O/acc-$k-fb.log && echo "acc $k: logs byte-identical" || { echo "acc $k: DIFFER"; diff $O/acc-$k-base.log $O/acc-$k-fb.log | head; }; done
fi

if has 6; then
  echo "== 6 plain-generate 4k A/B x3 (256 tokens), macmon 100 ms $(date -Iseconds)"
  AB=$O/ab-gen${AB_TAG:-}; mkdir -p $AB
  for arm in ${=AB_ARMS:-base-1 fb-1 fb-2 base-2 base-3 fb-3}; do
    /Users/josef/bin/macmon pipe -i 100 > $AB/$arm-macmon.jsonl 2>/dev/null & MPID=$!
    sleep 1
    O0=$O; O=$AB; gen $arm $arm --profile-every 0 --max-tokens 256 --ignore-eos --prompt "$(cat $P/prompt-4096.txt)"; O=$O0
    echo "$arm $(python3 -c 'import time; print(f"{time.time():.3f}")')" >> $AB/times.txt
    sleep 1; kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""; gzip -f $AB/$arm-macmon.jsonl
    sleep ${AB_SLEEP:-10}
  done
  python3 $RD/gate.py $AB
fi

if has 7; then
  echo "== 7 served bench A/B x3 (128 / 4k, gen 256, 3 reps, --at-most) $(date -Iseconds)"
  mkdir -p $O/ab-serve
  serve_run() { # arm
    local arm=$1
    env INTARSIA_TOKEN_FEEDBACK=$(fbenv $arm) $(bin $arm) serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8093 \
      --served-model-name intarsia-metal --max-seq 20000 --profile-every 0 --no-supervisor --no-ui > $O/ab-serve/$arm-server.out 2> $O/ab-serve/$arm-server.err &
    SPID=$!
    for i in {1..240}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || break; sleep 1; done
    /Users/josef/bin/macmon pipe -i 250 > $O/ab-serve/$arm-macmon.jsonl 2>/dev/null & MPID=$!
    $B bench speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,4096 --gen 256 --reps 3 --thinking off --at-most \
      --gb-per-token 15.1368 --bandwidth-gbs 614 --prefill-chunk 2048 --label $arm --out-dir $O/ab-serve > $O/ab-serve/$arm-bench.txt 2>&1
    echo "$arm bench exit $?"; tail -4 $O/ab-serve/$arm-bench.txt
    if [[ ${2:-} == eos ]]; then python3 $RD/eosreq.py http://127.0.0.1:8093/v1 intarsia-metal 20 > $O/ab-serve/$arm-eos.txt 2>&1; tail -2 $O/ab-serve/$arm-eos.txt; fi
    kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""; gzip -f $O/ab-serve/$arm-macmon.jsonl
    kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
    kill -0 $SPID 2>/dev/null && kill -9 $SPID; wait $SPID 2>/dev/null; SPID=""
    for i in {1..30}; do lsof -iTCP:8093 -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
    sleep 5
  }
  EOSARG=""; has 8 && EOSARG=eos
  for arm in base-1 fb-1 fb-2 base-2 base-3 fb-3; do serve_run $arm $EOSARG; done
fi

if has 9; then
  echo "== 9 qualify (fb) $(date -Iseconds)"
  ( cd $W && INTARSIA_TOKEN_FEEDBACK=1 $D qualify --model $M --backend metal --golden-dir $G --out $O/qualify > $O/qualify.txt 2>&1 ); echo "qualify exit $?"; tail -25 $O/qualify.txt
fi
echo GATES_DONE $(date -Iseconds)
