# Metal quantised KV cache: fp8 (e4m3) and int8 (block-scaled), opt-in (`--kv-dtype fp8|int8`)

Date 2026-10-15. Machine endor (MacBook Pro, Apple M5 Max 40-core GPU, 128 GB). Branch `metal-kv-quant`, worktree
`/opt/code/intarsiaml-kvq`, off `spike/metal` `56ec2d5`. Nothing merged or pushed. Raw data: `raw/2026-10-15-metal-kv-quant/`.

## 0. Result in one paragraph

Both quantised caches work end to end (append, split-KV decode, k-row decode for `--spec ngram`, NAX flash prefill,
plumbing, sizing, prefix reuse, rollback), the default bf16 KV path is bitwise unchanged, and spec stays exact against spec-off
within each KV dtype. **Neither mode meets the accuracy bar.** fp8 e4m3 is 6 to 10 times over the KL bar (5.9e-3 to 9.8e-3 against
1e-3) and fails top-1 everywhere. int8 with one bf16 scale per 32 elements is much better but still misses on the long golden (KL
4.1e-3 prefill) and on decode KL on all three goldens (1.1e-3 to 1.9e-3); it passes prefill on the short (9.3e-4) and 4k (9.8e-4)
goldens only barely. A fake-quant probe (section 5) shows the error is almost entirely in **K**: quantising V alone to int8 at
block 32 costs nothing (long KL 4.6e-4 against 3.7e-4 bf16, all gates pass), quantising K alone costs 4.5e-3 at block 32 and still
1.3e-3 at block 8. A mode that meets the bar therefore keeps K in bf16 and stores V as int8 (23 percent fewer KV bytes, not 50);
that is not built, see section 9. The speed gain is also well below the profile doc's estimate (section 6), because the quantised
decode kernels reach only 60 to 67 percent of the memory peak, so a halved KV read becomes a 25 percent shorter attention kernel
and about +2 percent engine-side decode at 16k.

PENDING-FILL

