#!/usr/bin/env bash
# Lane cgv: ON spark1, inside a fleetq quiet slot. nsys profile of `generate` on the 27B: a ~4k-token prompt (CHARS of tools/prompts/corpus_docs.txt)
# then NTOK decoded tokens. Kernel trace with graph nodes, exported to csv in $O. Usage: bash -s -- <tree> <outname> [serve/generate args...] < prof-decode.sh
export PATH=/usr/local/cuda/bin:$PATH
TREE=$1; OUTN=$2; shift 2
P=$HOME/intarsiaml-cgv/trees/$TREE; O=$HOME/intarsiaml-cgv/out/$OUTN; rm -rf $O; mkdir -p $O
M=$HOME/intarsiaml-cgv/models/Qwen3.8-27B-MXFP4
head -c ${CHARS:-14035} $P/tools/prompts/corpus_docs.txt > $O/prompt.txt
nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $O/nsys $P/target/release/intarsia generate -m $M --backend cuda --mem-budget 45 \
  --max-seq 8192 --chat --no-think --temperature 0 --ignore-eos --profile-every 0 "$@" -p "$(cat $O/prompt.txt)" --max-tokens ${NTOK:-192} < /dev/null > $O/gen.out 2> $O/gen.err
tail -4 $O/gen.err | cut -c1-300
nsys stats -r cuda_gpu_trace --format csv --output $O/trace $O/nsys.nsys-rep > /dev/null 2>&1
ls $O
