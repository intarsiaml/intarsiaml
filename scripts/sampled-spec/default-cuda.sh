#!/usr/bin/env bash
# Gates of the default-on change on CUDA (spark1), inside one fleetq slot (FLEETQ_JOB set):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 60 --label "ss: CUDA default-on gates" -- scripts/sampled-spec/default-cuda.sh
# Builds main (d40d93896) and the branch head in ~/intarsiaml-ss, then with INTARSIA_SPEC_SAMPLED unset: greedy ids 35B tp=1
# --spec mtp --spec-k 2 short and 4K (main vs branch), plan show json, and the default check (startup line, verify passes of a
# T 0.7 request); removes ~/intarsiaml-ss at the end.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
W="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$W/.scratch/default-cuda; mkdir -p $O; : > $O/summary.txt
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='~/intarsiaml-ss'; R="$L/src"; B=$L/bin
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; unset INTARSIA_SPEC_SAMPLED;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
build() {
  rsh "rm -rf $L/stage && mkdir -p $L/stage $R $B"
  git -C "$W" archive "$1" | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target $L/stage/ $R/ && rm -rf $L/stage && cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "rm -f $R/target/release/intarsia; cd $R && CARGO_BUILD_JOBS=4 MAX_JOBS=4 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3; cp $R/target/release/intarsia $B/$2"
}
echo "== build $(date +%T)"; build d40d93896 main; build "$(git -C "$W" rev-parse HEAD)" br
printf '%s' "$(cat /opt/code/deepeval/tests/data/prompts/natural-v1/n4096.txt)" | ssh -o BatchMode=yes spark1 "cat > $L/p4k.txt"
echo 'Write a short paragraph about the history of the printing press.' | ssh -o BatchMode=yes spark1 "cat > $L/pshort.txt"
G="--backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --spec mtp --spec-k 2"
for b in main br; do
  rsh "$B/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json" > $O/plan-$b.json 2> $O/plan-$b.err
  for p in short 4k; do
    rsh "$B/$b generate $G --temperature 0 --max-tokens 96 --ignore-eos --show-ids --prompt \"\$(cat $L/p$p.txt)\"" > $O/ids-$b-$p.out 2> $O/ids-$b-$p.err
  done
  rsh "$B/$b generate $G --temperature 0.7 --seed 3 --max-tokens 96 --prompt \"\$(cat $L/pshort.txt)\"" > $O/sampled-$b.out 2> $O/sampled-$b.err
done
cmp -s $O/plan-main.json $O/plan-br.json && echo "plan show json identical" >> $O/summary.txt || echo "plan show json DIFF" >> $O/summary.txt
for p in short 4k; do [ -n "$(grep '^ids:' $O/ids-main-$p.err)" ] && cmp -s <(grep '^ids:' $O/ids-main-$p.err) <(grep '^ids:' $O/ids-br-$p.err) && echo "ids $p identical" || echo "ids $p DIFF"; done >> $O/summary.txt
for b in main br; do echo "sampled $b: $(grep -h 'sampled requests' $O/sampled-$b.err) | $(grep -h '^spec: verify' $O/sampled-$b.err)"; done >> $O/summary.txt
rsh "rm -rf $L && echo removed $L" >> $O/summary.txt
cat $O/summary.txt
