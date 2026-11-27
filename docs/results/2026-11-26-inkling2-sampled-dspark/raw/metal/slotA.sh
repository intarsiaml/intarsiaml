#!/bin/bash
# endor EXCLUSIVE fleetq slot, lane ink-sampled: Metal correctness. (1) the tiny Inkling-2 suites + the lane's tests on the Metal backend,
# (2) greedy md5 main vs branch (spec off and the recipe's default, short + 4k, 256 tokens; 64 = the first 64 ids), (3) the real-model
# law / seed-equality harness, arm off (INTARSIA_SPEC_SAMPLED_DFLASH=0) then arm on (=1). Time-boxed by STOP_AT (epoch s).
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'cleanup; exit 130' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-sampled; S=$W/.scratch; O=$S/metal/A-${1:-$(date +%m%d-%H%M)}; mkdir -p $O
export CARGO_TARGET_DIR=$W/target
M=$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BB=$S/bin/intarsia-${BR:-br1}; BM=$S/bin/intarsia-main
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate)' | head -3)
[ -n "$others" ] && { echo "GUARD: another intarsia run on the box: $others"; exit 1; }
echo "start $(date +%T) out $O branch md5 $(md5 -q $BB) main md5 $(md5 -q $BM) git $(git -C $W rev-parse --short HEAD)"
cd $W
if [ -z "${SKIP_TINY:-}" ]; then
  echo "== tiny suites on metal $(date +%T)"
  env INTARSIA_INKLING2_BACKEND=metal cargo test --release --no-fail-fast -p intarsia-models --test inkling2 --test inkling2_dspark --test inkling2_dspark_cut --test inkling2_verify_softcap --test inkling2_dspark_sampled --test inkling2_dspark_sampled_gumbel 2>&1 | grep -aE '^test |test result|Running|panicked|random draft|scripted' > $O/tiny.txt
  cat $O/tiny.txt | grep -aE 'test result|FAILED|panicked|Running|draft'
fi
if [ -z "${SKIP_MD5:-}" ]; then
  echo "== greedy md5 $(date +%T)"
  G="generate -m $M --backend metal --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids $S/prompts/md5.json"
  for b in main br; do
    B=$BM; [ $b = br ] && B=$BB
    for sp in off def; do
      if [ $sp = off ]; then $B $G --spec off > /dev/null 2> $O/md5-$b-$sp.err; else $B $G > /dev/null 2> $O/md5-$b-$sp.err; fi
      echo "  $b $sp rc $? $(date +%T) | $(grep -aE '^decode overall' $O/md5-$b-$sp.err) | $(grep -aE '^spec: verify' $O/md5-$b-$sp.err)"
    done
  done
fi
if [ -z "${SKIP_LAW:-}" ]; then
  N=${LAW_N:-1000}; NS=${LAW_SHARED:-200}
  T=$(ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1)
  echo "== real law harness $T N $N shared $NS $(date +%T)"
  for arm in 0 1; do
    seeds="0..$N"; [ $arm = 1 ] && seeds="1000000..$((1000000 + N)),0..$NS"
    rm -f $O/law-$arm.txt
    env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=metal INTARSIA_SPEC_SAMPLED_DFLASH=$arm \
      INTARSIA_SS_SEEDS=$seeds INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=6 INTARSIA_SS_OUT=$O/law-$arm.txt \
      $T --nocapture > $O/law-$arm.log 2>&1
    echo "  arm $arm rc $? $(date +%T) lines $(wc -l < $O/law-$arm.txt) | $(grep -aE 'requests in|loaded in|sampled requests' $O/law-$arm.log | tr '\n' ' ')"
  done
fi
echo "end $(date +%T)"
