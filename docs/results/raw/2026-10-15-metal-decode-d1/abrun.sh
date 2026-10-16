#!/bin/zsh
# usage: abrun.sh <ctx short|4k> <pairs> <outdir-name> <bin> <variant>...
# Interleaved A/B (exp.sh) of named variants of ONE new binary plus the base binary, inside the caller's fleetq slot.
# Variants: base (the 766c723 binary), off (new binary, every D1 kernel unregistered = base kernel set), glu (MatmulGlu only),
# gluN (+ RmsNormAddMatmulGlu), all (+ RmsNormAddMatmulQkv), qkv (RmsNormAddMatmulQkv only), new (the binary as is).
CTX=$1; PAIRS=$2; OUT=$3; BIN=$4; shift 4
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
BASE=/Users/josef/.local/state/claude-watch/d1bin/intarsia-base-766c723
G=rmsnorm_add_matmul_glu_bf16_gemv
Q=rmsnorm_add_matmul_qkv_bf16_gemv
specs=()
for v in "$@"; do
  case $v in
    base) specs+=("base=$BASE");;
    off) specs+=("off=$BIN:INTARSIA_METAL_D1_OFF=all:364");;
    glu) specs+=("glu=$BIN:INTARSIA_METAL_D1_OFF=$G+$Q:328");;
    gluN) specs+=("gluN=$BIN:INTARSIA_METAL_D1_OFF=$Q:292");;
    glu256) specs+=("glu256=$BIN:INTARSIA_METAL_D1_OFF=$G+$Q,INTARSIA_METAL_GLU256=1:328");;
    qkv) specs+=("qkv=$BIN:INTARSIA_METAL_D1_OFF=$G+matmul_glu_bf16_gemv:294");;
    all|new) specs+=("$v=$BIN:B5_DUMMY=1:222");;
    *) echo "unknown variant $v"; exit 2;;
  esac
done
$D/exp.sh $CTX $PAIRS $D/$OUT $specs
