#!/usr/bin/env bash
# Lane fp8kv (from attn16; on spark2, under fleetq --load cpu): SASS of every kernel in kernels/cuda/attention_splitkv.cu, base (main) vs dev,
# function by function (anonymous-namespace hash stripped). Prints IDENTICAL / DIFFERS / only-in per kernel.
#   ssh spark2 'bash -s' < scripts/fp8kv/sass.sh   (with ~/intarsiaml-fp8kv/kchk/kernels/cuda/attention_splitkv{,_base}.cu staged)
set -euo pipefail
cd ~/intarsiaml-fp8kv/kchk/kernels/cuda
for v in base dev; do
  f=attention_splitkv.cu; [ $v = base ] && f=attention_splitkv_base.cu
  /usr/local/cuda/bin/nvcc -cubin -O3 --std=c++17 -arch=sm_121 --expt-relaxed-constexpr -I. -o ../../$v.cubin $f
  /usr/local/cuda/bin/cuobjdump -sass ../../$v.cubin | sed -E 's/_ZN[0-9]+_GLOBAL__N__[0-9a-f]+_[0-9]+_attention_splitkv(_base)?_cu_[0-9a-f]+/_ZN_/g; s/(ombine_kernelILi[0-9]+ELb[01]E)Lb0E/\1/g' > ../../$v.sass
  mkdir -p ../../fn-$v; rm -f ../../fn-$v/*
  awk -v d=../../fn-$v '/Function : /{n=$3; gsub(/[^A-Za-z0-9_]/,"",n); f=d "/" n} f{print > f}' ../../$v.sass
done
cd ../..
for f in fn-base/*; do n=$(basename $f); if [ -f fn-dev/$n ]; then cmp -s $f fn-dev/$n && echo "IDENTICAL $n" || echo "DIFFERS $n"; else echo "only-in-base $n"; fi; done | sed -E 's/(splitkv[0-9a-z_]*kernel|combine_kernel)I/\1 I/' | awk '{print $1, $2, substr($3,1,60)}' | sort | uniq -c | sort -rn | head -80
for f in fn-dev/*; do n=$(basename $f); [ -f fn-base/$n ] || echo "only-in-dev $(echo $n | cut -c1-90)"; done
