#!/bin/sh
# W (--quiet gpu,cpu; the lane's wrap-up, coordinator rule after job K: slope < 1.5 us -> option a).
# Arms: A = after = the tip 81bb8c4b8 (snapshot .scratch/snap12) with its defaults;
#       B = before = the same binary with this lane's four switches off (INTARSIA_INK_ROUTE_FUSED=0 INTARSIA_INK_MSG_PRE=0
#           INTARSIA_INK_MIX_HOIST=0 INTARSIA_INK_BLEND_NORM_C=0);
#       C = the brief's base a88db4620 (snapshot .scratch/snapC), defaults (WITH_C=1; 0 skips it).
#   1. builds; 2. Metal tests at the tip (backend-metal Inkling tests, models inkling2 = G1 tiny, inkling2_batch,
#      inkling2_mix_decode_cpu); 3. greedy md5 A vs B; G2 short / long (decode mode), G3 short / long at the tip, self-goldens
#      byte-compared with job G's; plan show Qwen3-8B / Llama-3.1-8B against the base binary;
#   4. traces: normal step (cb, token feedback) and per-dispatch (job E's method: COMMIT_EVERY=1, feedback off, disp.py) for
#      A, B, C: dispatches per token and controller ms per token;
#   5. localeval speed, sizes 128 and 5000, gen 128, 3 reps, serve --max-batch 1 --max-seq 8192, order A B C C B A.
# No INTARSIA_* variable is inherited: every run sets its own. The body runs in its own process group; TERM kills the tree.
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
WANT=81bb8c4b8; WANTC=a88db4620
WITH_C=${WITH_C:-1}
L=$S/W; mkdir -p $L
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
OFF4="INTARSIA_INK_ROUTE_FUSED=0 INTARSIA_INK_MSG_PRE=0 INTARSIA_INK_MIX_HOIST=0 INTARSIA_INK_BLEND_NORM_C=0"
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
set -m
guard() { # dir want
  [ "$(git -C $1 rev-parse --short=9 HEAD)" = $2 ] || { echo "ABORT: $1 at $(git -C $1 rev-parse --short=9 HEAD), not $2"; exit 1; }
  [ -z "$(git -C $1 status --porcelain --untracked-files=no)" ] || { echo "ABORT: $1 has uncommitted changes"; exit 1; }
}
body() {
  guard $S/snap12 $WANT
  [ $WITH_C = 1 ] && guard $S/snapC $WANTC
  for d in snap12 snapC; do
    [ $d = snapC ] && [ $WITH_C != 1 ] && continue
    [ -d $S/$d/target ] || cp -c -R $S/snap11/target $S/$d/target
    (cd $S/$d && git ls-files -z crates kernels | xargs -0 touch)
  done
  date > $L/times.txt
  echo "STEP build"
  (cd $S/snap12 && cargo build --release --bin intarsia --features metal) > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
  if [ $WITH_C = 1 ]; then
    (cd $S/snapC && cargo build --release --bin intarsia --features metal) > $L/build_c.log 2>&1 || { echo "BUILD C FAILED (arm C dropped)"; tail -20 $L/build_c.log; WITH_C=0; }
  fi
  BA=$S/snap12/target/release/intarsia; BC=$S/snapC/target/release/intarsia
  echo "build $(date)" >> $L/times.txt
  cd $S/snap12
  echo "STEP Metal tests at the tip"
  t() { tag=$1; shift; env "$@" > $L/t_$tag.log 2>&1; rc=$?
    echo "EXIT $tag $rc: $(grep -h 'test result' $L/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
    grep -h "panicked\|FAILED" $L/t_$tag.log | head -8; }
  {
  t metal_ink INTARSIA_X=0 cargo test --release -p intarsia-backend-metal --test inkmoe --test inkling2_attn --test inkctrl --test inkctrl_nometal4 --test inkmix --test inkload
  t models_metal INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch --test inkling2_mix_decode_cpu -- --test-threads 1
  } 2>&1 | tee $L/tests.txt
  echo "tests $(date)" >> $L/times.txt
  echo "STEP md5 A vs B"
  for n in 64 256; do
    $S/tmo.sh 300 $BA generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens $n --temperature 0 --profile-every 0 > $L/g${n}_A.out 2> $L/g${n}_A.err
    env $OFF4 $S/tmo.sh 300 $BA generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens $n --temperature 0 --profile-every 0 > $L/g${n}_B.out 2> $L/g${n}_B.err
    a=$(md5 -q $L/g${n}_A.out); b=$(md5 -q $L/g${n}_B.out)
    echo "MD5 $n tokens: A $a, B $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER ) (job G off: 64 65cde8f35a1802e3e2fadb44d94d080d, 256 61fd59fecb5e0d72ffd08f2c6d07855a)"
  done | tee $L/md5.txt
  echo "STEP G2 / G3 at the tip"
  for g in short long; do
    kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
    $BA accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-self.safetensors > $L/g2-$g.log 2>&1; echo EXIT $? >> $L/g2-$g.log
    kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
    $BA accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-self.safetensors > $L/g3-$g.log 2>&1; echo EXIT $? >> $L/g3-$g.log
    for k in g2 g3; do
      echo "== $k-$g: $(grep -h '^OVERALL' $L/$k-$g.log | tr -s ' ' | cut -c1-110 | tr '\n' '|') $(tail -1 $L/$k-$g.log)"
      echo "   self-golden vs job G (fcd05b172, off): $(python3 -I $S/stcmp.py $S/G/$k-$g-off-self.safetensors $L/$k-$g-self.safetensors 2>&1)"
    done
  done | tee $L/gates.txt
  echo "STEP plan show"
  for m in Qwen3-8B Llama-3.1-8B-Instruct; do
    $S/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_${m}_base.out 2> $L/plan_${m}_base.err
    $BA plan show --model ~/models/$m --backend metal > $L/plan_${m}_tip.out 2> $L/plan_${m}_tip.err
    if cmp -s $L/plan_${m}_base.out $L/plan_${m}_tip.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_tip.err; then v=byte-identical; else v=DIFFERS; fi
    echo "PLAN $m, base f55bfe8b7 vs $WANT: $v"
  done | tee $L/plan.txt
  echo "gates $(date)" >> $L/times.txt
  echo "STEP traces"
  arm_env() { case $1 in B) echo "$OFF4";; *) echo "INTARSIA_X=0";; esac; }
  arm_bin() { case $1 in C) echo $BC;; *) echo $BA;; esac; }
  ARMS="A B"; [ $WITH_C = 1 ] && ARMS="A B C"
  for a in $ARMS; do
    for k in short 5k; do
      if [ $k = 5k ]; then PR="$(cat $S/prompt4k.txt)"; else PR="$P"; fi
      env $(arm_env $a) INTARSIA_METAL_PROFILE=cb:$L/cb_${a}_$k.jsonl $S/tmo.sh 300 $(arm_bin $a) generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$PR" --max-tokens 257 > $L/cb_${a}_$k.out 2> $L/cb_${a}_$k.err
      echo "normal $a $k: $(python3 $S/cbstat.py $L/cb_${a}_$k.jsonl fb 2>&1 | grep -E 'steps|span' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_${a}_$k.err) md5 $(md5 -q $L/cb_${a}_$k.out)"
    done
    env $(arm_env $a) INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_$a.jsonl $S/tmo.sh 300 $(arm_bin $a) generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_$a.out 2> $L/disp_$a.err
    python3 $S/disp.py $L/disp_$a.jsonl > $L/disp_$a.txt 2>&1
    echo "per-dispatch $a: $(head -1 $L/disp_$a.txt); $(tail -1 $L/disp_$a.txt)"
  done | tee $L/traces.txt
  echo "traces $(date)" >> $L/times.txt
  echo "STEP localeval (order A B C C B A)"
  . $S/serve_lib.sh
  ORDER="A B A B"; [ $WITH_C = 1 ] && ORDER="A B C C B A"
  i=0
  for a in $ORDER; do
    i=$((i + 1)); port=$((8110 + i)); SRV_PORT=$port
    case $a in A) rev=$WANT; lab=after; e="";; B) rev=$WANT; lab=before-switches-off; e="$OFF4";; C) rev=$WANTC; lab=brief-base;; esac
    [ $a = C ] && e=""
    start_group $L/serve_${i}_$a.log $L/serve_${i}_$a.err env $e INTARSIA_X=0 $(arm_bin $a) serve --model $CK --backend metal --host 127.0.0.1 --port $port --max-batch 1 --max-seq 8192 --served-model-name inkling-2-mini
    n=0; until curl -sf http://127.0.0.1:$port/v1/models > /dev/null; do sleep 1; n=$((n + 1)); [ $n -gt 240 ] && break; kill -0 $SRV_PGID 2>/dev/null || break; done
    if curl -sf http://127.0.0.1:$port/v1/models > /dev/null; then
      (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model inkling-2-mini --thinking default --sizes 128,5000 --gen 128 --reps 3 \
         --label endor-intarsia-metal-ink-dfuse-$lab-$rev \
         --recipe "endor: ${e:+$e }intarsia serve --model Inkling-2-Mini --backend metal --max-batch 1 --max-seq 8192 (intarsiaml ink/dfuse arm $a, $lab, $rev)" > $L/le_${i}_$a.log 2>&1)
      RC=$?; echo "localeval $i arm $a ($lab $rev) exit $RC $( [ $RC = 0 ] || echo '(NOT COUNTED)' )"; grep -iE "decode|prefill|tok/s|run id|results" $L/le_${i}_$a.log | tail -8
    else
      echo "localeval $i arm $a: server did not come up (NOT COUNTED)"; tail -5 $L/serve_${i}_$a.err
    fi
    stop_group
  done 2>&1 | tee $L/localeval.txt
  echo "done $(date)" >> $L/times.txt
  cat $L/times.txt
}
body &
BP=$!
cut() {
  echo "CUT ($1): killing process group $BP"
  kill -TERM -- -$BP 2>/dev/null; sleep 5; kill -KILL -- -$BP 2>/dev/null; sleep 1
  left=$(pgrep -g $BP 2>/dev/null | tr '\n' ' '); [ -n "$left" ] && echo "LEFTOVER pids in group $BP: $left" || echo "no descendants left"
  exit 143
}
trap 'cut TERM' TERM; trap 'cut INT' INT; trap 'cut HUP' HUP
wait $BP; rc=$?
echo "body exit $rc"; exit $rc
