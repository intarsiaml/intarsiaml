#!/bin/zsh
# compact view of gemv_b3 logs
grep -E "^RESULT|^==|idle" $1 | sed -E 's/RESULT case=([a-z]+) cand=gemv_(pair_)?(bf16_)?/\1 /; s/pct_peak=[0-9.]+ //; s/median=/m=/; s/p10=/p10=/; s/us_per_dispatch=/us=/; s/vs_first=/d=/; s/diff_elems=/nd=/; s/max_abs_diff=/mx=/'
