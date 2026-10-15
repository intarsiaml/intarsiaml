# Metal decode D1: fusing the small per-layer ops into GEMV dispatches (2026-10-15)

**Outcome: no fusion gained; nothing to merge.** Three fused ops were built (bitwise the unfused Metal sequence, all gates
green) and none beat the unfused stream in an interleaved A/B. Decode dispatches per step did drop 364 -> 222, but the step
period did not: the best variant is neutral (-0.15 / -0.02 ms short / 4k median, inside the noise), the others are 0.2 to 2.1 ms
slower. The stop rule (two consecutive attempts each < 0.2 ms/token) was met at 00:10 EDT, so the run stopped there. The tree
under `crates/` and `kernels/` on `metal-d1-fuse` is byte-identical to `spike/metal` (766c723); the fusion code is kept in
history only (a18b8e1 .. 0bc548e) and reverted in 01157c0. **Merge-ready: not applicable. There is no code change to merge: vs spike/metal the branch is docs and raw logs only (net-zero experiment history), so a merge is harmless and the CUDA/HIP no-op check has nothing to check. The fusion-on binary passed every guardrail (ids identical, KL identical to base, kernels ADMITTED); it was dropped for speed only. Whether to merge the docs branch is your call.**

Host endor (Apple M5 Max, 614 GB/s), `/Users/josef/models/Qwen3-8B` BF16, base binary = spike/metal 766c723.
Raw logs: `docs/results/raw/2026-10-15-metal-decode-d1/`. fleetq: #1921 (op traces, unit tests, 48-token identity), #1923 (A/B 1), #1925 (kernel traces), #1927 (A/B 2), #1929 (A/B 3), #1930 (gates),
#1931 (base accuracy), #1932 (cargo test metal). #1917 and #1918 were cancelled before running. #1919 is void: both binaries lacked `--features metal`,
so its "identity identical" compared two identical error messages; #1921 is the valid rerun. (#1922, #1924, #1926, #1928 belong to another agent.)

## 1. Decode op list (kernel-mode GPU trace, `trace/base-kernel.jsonl.gz`, `trace_sum.py`)

364 dispatches per step (Qwen3-8B, 36 layers). Per layer: `matmul_bf16_gemv` (q), `matmul_pair` (k, v), `qk_norm_rope_kv`,
`attn_decode_splitkv`, `matmul_bf16_gemv` (o), `rmsnorm_add`, `matmul_pair` (gate, up), `silu_glu`, `matmul_bf16_gemv` (down), and
`rmsnorm_add` (next layer's input norm, shared with the residual add). Per step: embedding, rmsnorm, final `rmsnorm_add`, logits, argmax.

| op | calls/step | mean us | ms/step |
|---|---|---|---|
| matmul_pair (gate/up 36, k/v 36) | 72 | 187 | 13.47 |
| matmul (q, o, down) | 108 | 97 | 10.50 |
| logits | 1 | 2093 | 2.09 |
| attn_decode_splitkv | 36 | 21 | 0.76 |
| rmsnorm_add | 72 | 5.0 | 0.36 |
| qk_norm_rope_kv | 36 | 8.1 | 0.29 |
| silu_glu | 36 | 2.6 | 0.09 |

Small ops that were left: 72 `rmsnorm_add` + 36 `silu_glu` = 0.45 ms of kernel time and 108 dispatches, so the ceiling
for this whole phase was about 0.5 to 0.7 ms/token (B4 got 0.46 to 0.66 by removing 180 dispatches).

## 2. Fusions built (Metal-only core ops, CPU reference = composition of existing ops, early-return fallback)

| op | replaces | dispatches saved | kernel |
|---|---|---|---|
| `MatmulGlu` | gate/up pair + `silu_glu` | 1/layer (36) | `gemv_glu_v16/v8/v2`: one threadgroup per output row of BOTH matrices, GLU as epilogue; 512-thread variant `gemv_glu2_*` (gate row on simdgroups 0..7, up on 8..15) |
| `RmsNormAddMatmulGlu` | FFN `rmsnorm_add` + pair + `silu_glu` | 2/layer (72) | same, with the residual add + norm recomputed per threadgroup in a prologue (normed row in threadgroup memory; `sum` stored by tg 0) |
| `RmsNormAddMatmulQkv` | attention `rmsnorm_add` + q GEMV + k/v pair | 2/layer (70) | `gemv_qkv_norm_*`, same prologue, one tg per row of [q;k;v] |

Reduction order of every dot product is that of `gemv_bf16_rt<2,8>` (and `_a8<4,8>` for 8-byte aligned shards); the prologue is the
`rmsnorm_add_bf16_ex16` expression sequence, so all three are bitwise the unfused Metal sequence (2-byte-load fallback variant for odd
alignment is tolerance-only). `INTARSIA_METAL_D1_OFF=<kernel names or all>` unregistered kernels for same-binary A/B.

## 3. A/B (interleaved, cb-mode GPU trace of 129-token decode, step period from `period.py`; delta = new - base, positive = slower)

Median of per-pair deltas vs the spike/metal binary, ms/token. `ab*.log`, summary by `absum.py`.

| run | variant | dispatches | short (128) | 4k |
|---|---|---|---|---|
| A/B 1 (3 pairs, first prologue) | MatmulGlu | 328 | +0.19 | |
| | + RmsNormAddMatmulGlu | 292 | +0.19 | |
| | + RmsNormAddMatmulQkv (all) | 222 | +0.87 | |
| | new binary, all D1 off (control) | 364 | +0.09 (one +1.19 outlier) | |
| A/B 2 (4 pairs, prologue rewritten to hold less in registers) | MatmulGlu | 328 | +0.12 | +0.18 |
| | + RmsNormAddMatmulGlu | 292 | +1.10 | +0.63 |
| | all | 222 | +2.11 | +2.14 |
| A/B 3 (5 / 4 pairs, 512-thread MatmulGlu) | MatmulGlu 512 threads | 328 | **-0.15** | **-0.02** |
| | MatmulGlu 256 threads | 328 | +0.21 | +0.29 |

Reading: removing a ~3 us `silu_glu` dispatch is worth nothing once the FFN pair GEMV is already at bandwidth; the norm prologue
costs more than the dispatches it removes (every one of the 12288 / 6144 threadgroups redoes the 4096-element add + norm and
two barriers before its weight stream can fully ramp, and kernel-mode times of the fused kernels, 361 us vs 371 + 5 + 3 us unfused,
do not show it, hypothesis: kernel mode serializes the ramps the real stream overlaps; not measured. The step-period A/Bs are the evidence). The lower-register prologue rewrite (A/B 2) made
it worse, not better. A/B 3 shows the only variant that is not slower, and it is flat.

Stop rule: MatmulGlu (+0.19, then -0.15 / -0.02) and RmsNormAddMatmulGlu (+0.19, +1.10 / +0.63) are two consecutive attempts below
0.2 ms/token each, the qkv fusion was clearly negative. Stopped.

## 4. Gates (run on the last experiment binary `intarsia-d1d`, all three fusions on; they show the fusions were correct, not that anything is mergeable)

| gate | result |
|---|---|
| `tests/glu.rs` (10 tests: bitwise vs unfused at Qwen3-8B shapes, K tails, 8-byte aligned weights, tolerance for 2-byte/odd K, M=5 not accepted) | 10/10 pass (`tests/glu-d1d.log`) |
| `intarsia kernels test --backend metal` | 25 kernels, 280 cases, all ADMITTED incl. `matmul_glu` / `rmsnorm_add_matmul_glu` (max err 0) / `rmsnorm_add_matmul_qkv` (`gates/kernels-test.out`) |
| greedy ids 48 / 256, prompt-short and prompt-4k vs the 766c723 binary | byte-identical stdout and ids, all four (`gates/identity/`) |
| accuracy, base vs d1d | identical: prefill short KL 2.686e-4 top-1 99.77%, long KL 4.342e-4 top-1 99.65%; decode short 3.739e-4, long 3.731e-4 (`gates/accuracy/`) |
| `cargo test --workspace --release --exclude intarsia-backend-metal` | 269 pass, 0 fail (after updating `intarsia-models/tests/plan.rs` for the new fused call counts) |
| `cargo test -p intarsia-backend-metal --release` | 78 pass, 1 fail: `fingerprint_names_device_and_metallib` (my `d1off:` suffix in the fingerprint string); moot after the revert |
| `cargo check --workspace`, `--all-targets` | pass on the macOS host (Linux/CUDA/HIP check not run, as instructed) |

Note: the decode KL of the 766c723 binary measured here is 3.74e-4 / 3.73e-4, not the 4.04e-4 / 4.07e-4 in the task text; base and new are identical either way.

Not run: `intarsia qualify`, localeval pairs (128 / 1k / 4k, % of floor), and a decode-speed/prefill check of a final tree, because
no fusion is kept and the final tree is spike/metal's own code (`git diff 766c723 HEAD -- crates kernels` is empty). A/B step-period numbers above are the speed evidence.

## 5. intarsia-core files changed

Final tree: none. In the reverted experiment commits: `crates/intarsia-core/src/{ops.rs, model.rs, admission.rs}` (new ops, `ffn_norm_glu` /
`attn_norm_qkv` fallbacks, admission sweeps), plus `intarsia-backend-cpu/src/{fast,fused_ref,reference}.rs`, `intarsia-backend-{cuda,hip}/src/kernels.rs`
(new ops listed as unimplemented), `intarsia-backend-metal/{build.rs, src/{dense,glu,kernels,lib}.rs, tests/glu.rs}`, `kernels/metal/gemv_glu.metal`,
`intarsia-models/tests/plan.rs`.

## 6. Dead ends and what is left

- GLU epilogue alone: bitwise, flat. The silu dispatch is not worth a kernel.
- Norm prologue per threadgroup: every row redoes the norm; on 12288-row GEMVs this loses more than the dispatch saves. A smaller-register rewrite was worse.
- Not tried (each needs a design that breaks bitwise or needs cross-threadgroup sync): norm as an epilogue of the o / down GEMV via a last-threadgroup-done atomic counter; a precomputed per-row rsqrt from a partial sum-of-squares accumulated in the previous GEMV (changes reduction order, so tolerance-only); fusing `rmsnorm_add` into the *previous* GEMV's tail with a single extra 4096-element threadgroup.
- Remaining floor gap is not in small ops: 27.1 ms period vs 24.65 ms floor, mostly GEMV ramps and the 2.1 ms logits GEMV.
