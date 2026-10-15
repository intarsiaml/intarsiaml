#!/bin/zsh
# Metal final gates, ONE fleetq slot (acquired here; released by trap). CAND = metal-candidate2 (SAFE a2102bf + metal-p3-chunk,
# prefill chunk 2048 on Metal), SAFE = spike/metal a2102bf (chunk 256). Both binaries built with
# `cargo build --release -p intarsia-cli --features metal`, each in its own worktree, copied to $B.
B=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/final
CAND=$B/intarsia-cand
SAFE=$B/intarsia-safe
W=/opt/code/intarsiaml-final-cand
R=$W/docs/results/raw/2026-10-15-metal-final/gates
RAW=$W/docs/results/raw
PB=$RAW/2026-10-14-metal-phase-b          # prompt-short.txt, prompt-4k.txt
PC=$RAW/2026-10-14-metal-phase-c          # prompt-1k.txt
P16=$RAW/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
FD=$RAW/2026-10-14-metal-prefill-p1/full  # first_divergence.py
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/identity $R/accuracy $R/plain
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 60 --label "metal final: gates (tests, kernels, accuracy, ids, plain prefill)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
{ echo "date: $(date -Iseconds)"; echo "cand head: $(git rev-parse HEAD)"; echo "safe head: $(git -C /opt/code/intarsiaml-final-safe rev-parse HEAD)"
  echo "cand binary: $(ls -l $CAND)"; echo "safe binary: $(ls -l $SAFE)"; sw_vers | tr '\n' ' '; echo; } > $R/versions.txt 2>&1
pgrep -fl zoom.us > $R/zoom.txt 2>&1
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
echo "idle start: $(idle3)" | tee $R/idle-start.txt

echo "== 1. unit tests and admission (CAND)"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
grep -E "^test result|FAILED|panicked" $R/cargo-test-metal.txt | sort | uniq -c | head -20
$CAND kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
echo "ADMITTED lines: $(grep -c ADMITTED $R/kernels-test.txt)"; grep -E "FAIL|REJECT" $R/kernels-test.txt | head; grep -E "kernels," $R/kernels-test.txt

echo "== 2. accuracy against the fp32 goldens (bf16 gate class), CAND then SAFE"
for B in cand safe; do
  eval BIN=\$${(U)B}
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  for F in short long; do
    echo "$B $F: $(grep -E 'floor \[bf16\]' $R/accuracy/$B-$F.out | grep -oE '(prefill|decode[^ ]*) floor|top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|=> (PASS|FAIL)' | tr '\n' ' ')"
  done
  grep -h "prefill chunk" $R/accuracy/$B-short.err $R/accuracy/$B-long.err | sort -u
done

echo "== 3. greedy ids CAND vs SAFE (48 and 256 tokens, short and 4k prompts)"
for N in 48 256; do
  for P in short 4k; do
    for B in safe cand; do
      eval BIN=\$${(U)B}
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
        --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $FD/first_divergence.py $R/identity/safe-$P-$N.err $R/identity/cand-$P-$N.err "$P $N tokens safe vs cand" | tee -a $R/identity/summary.txt
  done
done

echo "== 4. plain prefill (generate --max-tokens 1) ABBA: A=SAFE B=CAND, each binary at its own default chunk"
promptfile() { case $1 in 1k) echo $PC/prompt-1k.txt;; 4k) echo $PC/prompt-4k.txt;; 16k) echo $P16;; esac; }
prun() { # label bin prompt idx
  local f=$R/plain/$1-$3-$4 pf=$(promptfile $3)
  $2 generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 1 --profile-every 0 --prompt "$(cat $pf)" > $f.out 2> $f.err
  echo "$1 $3 #$4: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $f.err | tail -1) $(grep -o 'prefill chunk [0-9]*' $f.err | head -1)"
}
prun warm $SAFE 4k 0
for P in 1k 4k 16k; do
  echo "idle $P: $(idle3)" >> $R/idle-start.txt
  for r in 1 2 3; do
    prun safe $SAFE $P ${r}a
    prun cand $CAND $P ${r}b
    prun cand $CAND $P ${r}c
    prun safe $SAFE $P ${r}d
  done
done
echo "idle end: $(idle3)" | tee -a $R/idle-start.txt
echo GATES_DONE
