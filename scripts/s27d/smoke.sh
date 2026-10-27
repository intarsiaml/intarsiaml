#!/usr/bin/env bash
# Lane s27d: one-shot generate on spark1 (runs ON spark1): bash -s < scripts/s27d/smoke.sh ; env TREE (default base), SPEC ("--spec off" | "--spec mtp --spec-k 2" | ""), NTOK, EXTRA
export PATH=/usr/local/cuda/bin:$PATH
T=$HOME/intarsiaml-s27d/trees/${TREE:-base}; M=$HOME/intarsiaml-s27d/models/Qwen3.8-27B-MXFP4
$T/target/release/intarsia generate -m $M --backend cuda --mem-budget 45 --max-seq 4096 --chat --no-think --temperature 0 --ignore-eos --show-ids \
  -p "Write a detailed explanation of how a hash map works, with a Rust example." --max-tokens ${NTOK:-128} ${SPEC-} ${EXTRA:-} < /dev/null 2>&1 | tail -${TAIL:-25}
