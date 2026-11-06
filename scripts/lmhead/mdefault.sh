#!/bin/zsh
# Lane lmhead-fp8-default, Metal gates, ONE slot (not exclusive: no timing):
#   fleetq run --host endor --quiet gpu --mem 40 --expect 70 --label "..." -- zsh scripts/lmhead/mdefault.sh <out dir>
# Binaries: $HEAD_BIN (this worktree, target/release/intarsia) and $BASE_BIN (main af6aa5226, .scratch/base/target/release/intarsia).
# 1. `plan show` (text and JSON) of Qwen3-8B, Llama-3.1-8B, Qwen3.6-35B-A3B NVFP4 and the Qwen3.8-27B, base vs head: all but the 27B byte-identical;
#    the 27B differs by the head lines only (the diff is kept); 27B head+off == base; 27B head == base+fp8.
# 2. `kernels test --backend metal` base vs head: 0 FAIL.
# 3. 256 greedy ids (short and ~3.5k prompts, default flags = MTP k2 where the recipe says so) of the 27B for base, base+fp8 (the opt-in lane's
#    behaviour), head, head+off; and of the 35B for base and head.
# 4. serve smoke of the head binary with the 27B defaults: MTP + fp8 head in the log, 4 greedy prompts solo vs concurrent (--max-batch 4), one sampled request.
set -u
R=${0:A:h:h:h}; O=${1:?out dir}; mkdir -p $O
HEAD_BIN=$R/target/release/intarsia; BASE_BIN=$R/.scratch/base/target/release/intarsia
PD=$R/scripts/tp2-hip/noop/prompts; H=$HOME/models
M8=$H/Qwen3-8B; ML=$H/Llama-3.1-8B-Instruct; M35=$H/Qwen3.6-35B-A3B-NVFP4; M27=$H/Qwen3.8-27B-Quark-AWQ-MXFP4
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
[[ -z "${INTARSIA_LMHEAD_QUANT:-}" ]] || { echo "INTARSIA_LMHEAD_QUANT is set in the environment"; exit 2; }
echo "head: $(ls -la $HEAD_BIN | awk '{print $6,$7,$8}')  base: $(ls -la $BASE_BIN | awk '{print $6,$7,$8}')  $(git -C $R rev-parse --short=9 HEAD)" | tee $O/binaries.txt
# arm name -> binary and env
typeset -A BIN ENVS
BIN=(base $BASE_BIN base-fp8 $BASE_BIN head $HEAD_BIN head-off $HEAD_BIN); ENVS=(base "" base-fp8 "INTARSIA_LMHEAD_QUANT=fp8" head "" head-off "INTARSIA_LMHEAD_QUANT=off")
run() { local arm=$1; shift; if [[ -n ${ENVS[$arm]} ]]; then env ${ENVS[$arm]} ${BIN[$arm]} "$@"; else ${BIN[$arm]} "$@"; fi }
plan() { # plan <arm> <name> <model>
  run $1 plan show --backend metal --model $3 --max-seq 8192 --json < /dev/null > $O/plan-$2-$1.json 2> $O/plan-$2-$1.json.err
  run $1 plan show --backend metal --model $3 --max-seq 8192 < /dev/null > $O/plan-$2-$1.txt 2> $O/plan-$2-$1.txt.err
}
for a in base head; do plan $a 8 $M8; plan $a llama $ML; plan $a 35 $M35; done
for a in base base-fp8 head head-off; do plan $a 27 $M27; done
echo "== plan show"
for n in 8 llama 35; do for k in json txt; do
  cmp -s $O/plan-$n-base.$k $O/plan-$n-head.$k && [[ -s $O/plan-$n-head.$k ]] && echo "plan $n $k: byte-identical ($(wc -c < $O/plan-$n-head.$k | tr -d ' ') B)" || echo "plan $n $k: DIFFER or empty"
done; done
for k in json txt; do
  cmp -s $O/plan-27-head-off.$k $O/plan-27-base.$k && [[ -s $O/plan-27-base.$k ]] && echo "plan 27 $k: head+off == base (byte-identical)" || echo "plan 27 $k: head+off vs base DIFFER"
  cmp -s $O/plan-27-head.$k $O/plan-27-base-fp8.$k && [[ -s $O/plan-27-head.$k ]] && echo "plan 27 $k: head == base+fp8 (byte-identical)" || echo "plan 27 $k: head vs base+fp8 DIFFER"
  diff $O/plan-27-base.$k $O/plan-27-head.$k > $O/plan-27-base-vs-head.$k.diff; echo "plan 27 $k: base vs head diff: $(grep -c '^[<>]' $O/plan-27-base-vs-head.$k.diff) changed lines (raw/plan-27-base-vs-head.$k.diff)"
done
echo "== kernels test"
for a in base head; do run $a kernels test --backend metal < /dev/null > $O/kernels-$a.out 2>&1; echo "kernels $a exit $?: $(grep -aE '^[0-9]+ kernels' $O/kernels-$a.out) FAIL lines: $(grep -acE '(^|[^A-Za-z])FAIL' $O/kernels-$a.out)"; done
diff <(grep -avE '^[0-9]+ kernels' $O/kernels-base.out) <(grep -avE '^[0-9]+ kernels' $O/kernels-head.out) > $O/kernels.diff && echo "kernels test: identical rows" || echo "kernels test: rows DIFFER, see kernels.diff"
echo "== ids"
ids() { # ids <arm> <name> <model> <prompt>
  run $1 generate -m $3 --backend metal --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids -p "$(cat $PD/$4.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/gen-$2-$4-$1.err
  grep -a '^ids:' $O/gen-$2-$4-$1.err > $O/gen-$2-$4-$1.ids
}
for p in short 4k; do for a in base base-fp8 head head-off; do ids $a 27 $M27 $p; done; for a in base head; do ids $a 35 $M35 $p; done; done
grep -ah "lm_head quantised\|speculative decoding" $O/gen-27-short-head.err | head -3 | cut -c1-200
for p in short 4k; do
  chk() { cmp -s $O/gen-$1-$p-$2.ids $O/gen-$1-$p-$3.ids && [[ -s $O/gen-$1-$p-$2.ids ]] && echo "ids $1 $p: $2 == $3 ($(tr ',' '\n' < $O/gen-$1-$p-$2.ids | wc -l | tr -d ' ') ids, sha $(shasum $O/gen-$1-$p-$2.ids | cut -c1-8))" || echo "ids $1 $p: $2 vs $3 DIFFER or empty"; }
  chk 27 head base-fp8; chk 27 head-off base; chk 27 head base; chk 35 head base
done
echo "== serve smoke"
PORT=8299
for mb in 1 4; do
  $HEAD_BIN serve --backend metal --model $M27 --served-model-name q27 --max-seq 8192 --profile-every 0 --no-supervisor --max-batch $mb --host 127.0.0.1 --port $PORT > $O/serve-mb$mb.out 2> $O/serve-mb$mb.err &
  SP=$!; up=0; for i in {1..240}; do curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  if [[ $up == 1 ]]; then
    grep -ah "lm_head quantised\|speculative decoding\|--spec" $O/serve-mb$mb.err | head -3 | cut -c1-200
    python3 $R/scripts/lmhead/msmoke.py http://127.0.0.1:$PORT q27 $O mb$mb
  else echo "SMOKE mb$mb: server did not come up"; tail -5 $O/serve-mb$mb.err | cut -c1-200; fi
  kill $SP 2>/dev/null; for i in {1..30}; do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
done
echo "== done"
