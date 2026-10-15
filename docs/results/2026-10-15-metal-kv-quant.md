# Metal quantised KV cache: fp8 (e4m3) and int8 (block-scaled), opt-in (`--kv-dtype fp8|int8`)

Date 2026-10-15. Machine endor (MacBook Pro, Apple M5 Max 40-core GPU, 128 GB, 614 GB/s). Branch `metal-kv-quant`, worktree
`/opt/code/intarsiaml-kvq`, off `spike/metal` `56ec2d5`. Nothing merged or pushed (`git merge-tree` against the current
`spike/metal` `de8fe5b` is clean). Raw data: `raw/2026-10-15-metal-kv-quant/`.


**Decision 2026-10-15 18:50 EDT (user): shelved.** Only this doc and its raw data are merged into `spike/metal`; the code stays
on branch `metal-kv-quant` at `b244ff1` (worktree `/opt/code/intarsiaml-kvq`). Revisit with a K-bf16 / V-int8 mode (passes the
gates in the fake-quant probe, saves 23% of KV bytes) if KV memory ever limits batching.

## 0. Result in one paragraph

Both quantised caches are built and work end to end: append, split-KV decode, the k-row decode kernel of `--spec ngram`, NAX flash
prefill, plumbing, KV sizing, prefix reuse, truncation / rollback. The default bf16 KV path is bitwise unchanged, and spec stays
bit-exact against spec-off within each KV dtype. **Neither mode meets the accuracy bar, so neither should become a default, and
neither is recommended for general use yet.**

* fp8 e4m3: KL 5.9e-3 to 9.8e-3 on all three goldens (bar 1e-3), top-1 97 to 98 percent (bar 99.5), fails the 4k prefix-reuse gate.
* int8, one bf16 scale per 32 elements: much closer (KL 9.3e-4 / 4.1e-3 / 9.8e-4 prefill on short / long / 4k, decode 1.1e-3 to 1.9e-3).
  It passes short and 4k prefill KL only barely and misses on the long golden (4.1e-3 prefill, top-1 99.16 percent).
* A fake-quant probe (section 5) shows **the error is almost all in K**. V alone at int8 / block 32 costs nothing (long golden KL
  4.6e-4 against 3.7e-4 bf16, all gates pass). K alone at int8 / block 32 costs 4.5e-3, and even a block of 8 costs 1.3e-3. The mode that
  would meet the bar is "K bf16, V int8", which saves 23 percent of the KV bytes, not 50. Not built (section 9).
* The speed gain is far below the profile doc's +7.5 / +14 / +15.6 percent estimate: the quantised decode kernels reach only 60 to
  67 percent of peak bandwidth, so halving the bytes makes the attention kernel 25 percent shorter (not 50), which is +2.4 to +2.8 percent
  engine-side decode at 16k (the one context size with a clean GPU clock). Section 6.

## 1. Design

Metal (MSL 3.1 and 4.0, toolchain `metalfe-32023.921`) has **no fp8 type**: `fp8`, `float8_e4m3`, `e4m3` and friends are all unknown
type names, and `float8` is a vector name. There is no native conversion, so every e4m3 decode is integer code. Decode costs, per 4 bytes:
about 10 integer ops and 4 half-to-float converts (`unpack4_e4m3_s8`, kernels/metal/common.h): the magnitude bits are shifted under the
half exponent field (a half whose value is the e4m3 value times 2^-8, exact, subnormals included) and converted to f32. Two findings
shaped this:

* The obvious decode, an f32 multiply by 2^120, is **wrong on this GPU**: the f32 denormal operand is flushed (found by the exhaustive
  256-code test, which failed on the first run, section 4). The half route avoids it.
* The 2^-8 is folded into q (exact power-of-two scaling) and removed from the accumulator at the end, so the decode needs no per-element
  multiply and the result is bit for bit that of the true e4m3 values.

int8 with a bf16 scale per 32 consecutive elements of a head row (one simdgroup of the append kernels, so the absmax is a 5-step shuffle
and costs nothing measurable): q = clamp(rint(x / s), -127, 127), s = bf16(absmax / 127), the rounded scale is the one used on both
sides. In the attention kernels a lane's 8 elements lie inside one block, so the score is `(sum q_i * k_i) * s_k` and the V term is
`(p * s_v) * v_i`: one extra scale load per key per lane and one multiply, in the same order in every kernel. Scales live after the
data of each plane, `[Nkv, cap, D/32]` bf16 (core `state::kv_plane_bytes`), so one buffer per K / V as before and 6.25 percent over a
plain int8 plane, 53.1 percent of the bf16 bytes per position.

Why both: fp8 is the simplest (no scales, no extra bytes, 50.0 percent) but has 3 mantissa bits (about 3 to 6 percent relative error
per element, on K and on V); int8 per block has 7 bits against the block maximum, which is far better unless a block contains an
outlier. The accuracy runs decide, and the answer is that K needs something better than either (section 9). Per-head (one scale per
head) was not built: it needs a running absmax over the appended rows and requantisation, which is incompatible with the
quantise-on-append, fused-prologue design; per-block-of-32 gets the same effect locally and is stateless.

Quantisation is fused into `qk_norm_rope_kv` (decode and prefill forms) and `KvAppend`; the value stored is the quantisation of the
**bf16-rounded** post-rope K / V, the value a bf16 cache would hold. K, V and q of the default path are untouched.

## 2. Kernels (all new, nothing existing modified except as listed in section 8)

| kernel | what |
|---|---|
| `kv_append_f8`, `kv_append_i8` | `KvAppend` into a quantised cache |
| `qk_norm_rope_kv_f8[_pf]`, `qk_norm_rope_kv_i8[_pf]` | the fused prologue with the quantising store (decode form, and the D=128 prefill form) |
| `attn_decode_f8_splitkv_g{1,2,4,8}`, `attn_decode_i8_splitkv_g*` | split-KV decode stage 1 (combine pass is the bf16 one; it never reads the cache) |
| `attn_rows_f8_splitkv_g*`, `attn_rows_i8_splitkv_g*` (`attn_prefill_{f8,i8}_splitkv_rows`) | the k-row kernel for `--spec ngram` |
| `attn_prefill_{f8,i8}_flash_nax`, `attn_prefill_{f8,i8}_flash` | prefill attention: `kv_dequant_{f8,i8}` decodes the cache once into a bf16 staging buffer (exact), then the unchanged bf16 flash kernels run |
| `fp8_test_decode`, `fp8_test_encode` | test shaders for the exhaustive conversion checks |

The prefill attention does not dequantise inside the NAX flash kernel (the task asked for that). The kernel loads K / V fragments with
`cooperative_tensor::load` from a bf16 tensor; lane-wise byte loads there would cost the 3.3x the F1b doc found for non-matching
loads, and prefill attention is compute bound (46 to 51 TFLOPS), so halving its read bytes gains nothing anyway. The staging decode
covers only the `pos0 + T` visible rows (about 1 percent of a 40k prefill). Bit-exact by construction and tested (section 4).
Staging buffer: K + V planes of `[Nkv, cap, D]` bf16, i.e. the bf16 cache size, allocated on first fp8 / int8 prefill, grown on demand,
never allocated for bf16. **That is a memory cost the KV sizing does not count**: one buffer shared by all layers (they run in stream order), 2 x Nkv x cap x D x 2 B,
168 MB for Qwen3-8B at `--max-seq 40960`.

`attn_decode_f8` uses 4 units in flight per step (the bf16 kernel uses 2: half the bytes per key, twice the keys). A microbench of 2 / 4 / 8
(`INTARSIA_METAL_F8_U`, `raw/.../slot1c/attn_bw.txt`): 4 is best (16k: 92 us; 2: 96 us; 8: 127 us), the default.

## 3. Plumbing

* `--kv-dtype bf16|fp8|fp8_e4m3|int8` (existing flag, `common.rs`; recipes and `serve` take the same flag through the shared args;
  `recipe` kv_dtype key already existed). Anything else is refused with a message that names the supported set. Backend support is
  checked twice: in the CLI before the model loads, and in `Model::load` (so a recipe default or an API caller is refused too):
  **cuda / hip: bf16 only; metal: bf16, fp8, int8; cpu oracle: bf16, fp8, f16, f32**. Verified: `--backend cpu --kv-dtype int8` and
  `--backend metal --kv-dtype f16` are refused cleanly.
* KV memory sizing: `state::kv_plane_bytes(n_kv_heads, positions, head_dim, dtype)` is now the one function used by
  `SequenceState::new`, `used_bytes`, `plan.rs` phantom state and the per-token KV budget in `Model::load`. For every dtype but int8 it is
  the old expression. Measured (`raw/.../sizing/`, 8B, `--mem-budget`, recipe cap 40960): 20 GiB: bf16 32 683 tokens, fp8 40 960 (cap),
  int8 40 960 (cap); **18 GiB: bf16 18 120, fp8 36 240 (2.00x), int8 34 108 (1.88x)**. The bf16 staging buffer of section 2 is not in
  these numbers.
* Prefix reuse and truncation / rollback: nothing in them is dtype specific (they move the length, the rows stay), and the
  prefix-reuse gate of `accuracy` checked it for each dtype (section 4): the **restore** (truncate to the split and re-prefill) and
  **aligned** cases are 32/32 for bf16, fp8 and int8. The **continue** case (turn 2 re-chunked on top of turn 1) is exact-or-within-rounding
  for bf16 (24/32 past the rounding tolerance, ok) and int8 (32/32), and **fails for fp8** (diverges at token 4). That case compares a
  warm and a cold run whose chunk boundaries differ, so it amplifies any cache noise; fp8's noise is large enough to flip a greedy token.
  This is accuracy, not a reuse bug: restore / aligned are the paths that move the cache length and they are exact.
* `kernels test --backend metal`: the fp8 kernels are registered (`KvAppend` has oracle cases for an fp8 cache since before;
  `QkNormRopeKv` gained a bf16-activation / fp8-cache case in core admission, so the new fp8 prologue is admitted against the f64 oracle).
  The oracle has no block-scaled cache, so the int8 kernels cannot be admitted by it; they are **listed only when an int8 cache is
  requested** (`INTARSIA_METAL_KV_INT8`, set by the CLI for `--kv-dtype int8`), otherwise `kernels test` would report "NO CASES
  (untested)" and fail admission (it did, before this was added: the first qualify run failed exactly there, section 7). They are
  covered by `tests/kv_i8.rs` instead.

## 4. Tests and exactness proofs (`cargo test -p intarsia-backend-metal --release`, 117 tests pass)

`tests/kv_fp8.rs` (8 tests) and `tests/kv_i8.rs` (7 tests):

* **Conversions**: all 256 e4m3 codes decode identically to the cpu oracle (`fp8_e4m3_to_f32`); the encoder is identical to the oracle
  (`f32_to_fp8_e4m3`) over all 65 536 bf16 patterns (NaN included), every rounding midpoint and its f32 neighbours, saturation, and
  200 000 random f32 bit patterns: 266 305 values, 0 differences.
* **Decode attention vs f64**: 3 shapes (32/8/128, 8/8/64, 16/2/256), 11 positions each, worst error 0.63 (fp8) and 0.56 (int8) of the
  tolerance (1.5 bf16 ulp relative + 1e-3 / 2e-3 absolute) on a random cache with a wide dynamic range. fp8 additionally gives **0 of
  96 000 bf16 outputs different from the bf16 kernel run on the decoded cache** (the decode is exact; the f32 summation order of the
  4-unit step differs slightly, so equality of the rounded outputs is observed, not guaranteed).
* **rows == 1-row, bitwise (g0b style)**: `attn_prefill_{f8,i8}_splitkv_rows` for k in {2, 3, 5, 8} at start positions covering every
  split boundary, 4 shapes: row r of a k-row call equals the 1-row `attn_decode_*` launch at `pos0 + r`, byte for byte; 92 / 92 / 80 / 64
  (k, pos0) windows per shape, for fp8 and for int8. This is the property `--spec ngram` needs.
* **Prefill**: `attn_prefill_{f8,i8}_flash[_nax]` are bitwise the bf16 flash kernels on the bf16 dequantisation of the cache, 9 to 10 (T, pos0)
  combinations including chunk-straddling and the cap edge, for the NAX and the non-NAX flash kernel.
* **Append**: `kv_append_{f8,i8}` equal the host quantiser byte for byte (data and scales), 5 (T, pos0) cases; the fused prologue
  (`qk_norm_rope_kv_{f8,i8}`, decode and prefill forms) writes the quantisation of the bf16 kernel's cache rows, one T-row call
  writes the same bytes as T 1-row calls, q output equals the bf16 kernel's, rows outside the append are untouched (sentinel).
* Registration: the quantised kernels accept only their cache dtype and no bf16 kernel accepts a quantised one.

Spec exactness end to end (`generate --spec ngram` vs spec off, same KV dtype, 256 tokens, 6 prompts): **identical ids for fp8 (6/6) and
int8 (6/6)**, `raw/.../ids/summary.txt`.

## 5. Accuracy

fp32 HF goldens (`golden/qwen3-8b-fp32`, `-long-fp32`, `-4k-fp32-top20`), `intarsia accuracy --gate-class bf16` (top-1 >= 99.5 percent,
KL <= 1e-3), Metal, `--prefill-chunk` recipe default 2048. Raw: `acc/`, table script `acc_table.py`.

| KV | golden | prefill top-1 % | prefill KL | decode top-1 % | decode KL | prefix reuse |
|---|---|---|---|---|---|---|
| bf16 | short | 100.00 | 2.83e-4 | 100.00 | 3.31e-4 | - |
| fp8 | short | 97.93 | 5.93e-3 | 96.88 | 7.78e-3 | - |
| int8 | short | 99.77 | 9.33e-4 | 100.00 | 1.50e-3 | - |
| bf16 | long | 99.72 | 3.70e-4 | 100.00 | 3.37e-4 | - |
| fp8 | long | 98.33 | 9.80e-3 | 96.88 | 8.20e-3 | - |
| int8 | long | 99.16 | 4.09e-3 | 100.00 | 1.86e-3 | - |
| bf16 | 4k | 99.28 | 2.27e-4 (21-way) | 98.96 | 2.55e-4 | PASS |
| fp8 | 4k | 96.42 | 7.41e-3 | 96.35 | 5.92e-3 | **FAIL** (continue) |
| int8 | 4k | 98.75 | 9.81e-4 | 100.00 | 1.07e-3 | PASS |

`qualify --kv-dtype fp8` (acc stages only, #2058, `qualify-fp8/`): `qualify` picks a looser gate class for an fp8 KV cache (KL <= 5e-3) and
**fp8 still fails it**: acc-short FAIL (KL 5.93e-3 / 7.78e-3), acc-long FAIL (9.80e-3 / 9.18e-3), acc-4k ERROR (the accuracy subcommand exits 1,
for the gate failures in the table above). `qualify --kv-dtype int8` was not run (the stage table is the same `accuracy` command; the
`accuracy` runs above are its content).

Bar: KL <= 1e-3 on all three, long top-1 >= 99.5. **fp8: fails every cell. int8: passes KL only on short and 4k prefill (by 7 and 2
percent), fails long (4.1x over, top-1 99.16) and decode KL on all three.** The decode windows are 128 / 192 positions, so their top-1
is coarse (one position is 0.5 to 0.8 points).

4k top-1 against the recorded Metal 4k baseline (`golden/baselines/qwen3-8b-4k-fp32-top20.metal.baseline.json`: prefill 99.28 /
decode 98.96, KL21 2.27e-4 / 2.55e-4; the baseline itself is below the 99.5 floor, its envelope run, chunk 2048 -> 256, moves nothing):
fp8 is 2.86 points below on prefill and 2.61 below on decode (KL 33x / 23x the baseline); int8 is 0.53 points below on prefill and 1.04
above on decode (192 positions: 2 more correct), KL 4.3x / 4.2x the baseline. Neither is "within the envelope"; the unstamped dev build
is why `accuracy` also prints UNSTAMPED, the numbers are what they are.

Fake-quant probe (`INTARSIA_METAL_KV_FAKEQ=<block>,<mask>`, a bf16 cache holding bf16(dequant(quant(x))) per block, mask 1 = K, 2 = V;
`probe/`; the shipped int8 kernel reproduces the 32/K+V row's prefill numbers exactly), long golden prefill / decode KL, top-1 %:

| quantised | block | long prefill top-1 / KL | long decode KL | short prefill KL |
|---|---|---|---|---|
| none (bf16) | - | 99.72 / 3.70e-4 | 3.37e-4 | 2.83e-4 |
| V only | 32 | 99.79 / 4.60e-4 | 4.05e-4 | 2.95e-4 |
| K only | 32 | 99.23 / 4.46e-3 | 1.76e-3 | 1.01e-3 |
| K only | 16 | 99.16 / 2.41e-3 | 7.97e-4 | 4.39e-4 |
| K only | 8 | 99.30 / 1.28e-3 | 5.80e-4 | 3.90e-4 |
| K only | 4 | 99.58 / 6.94e-4 | 4.40e-4 | 4.04e-4 |
| K + V | 32 | 99.16 / 4.09e-3 | 1.64e-3 | 9.33e-4 |
| K + V | 16 | 99.30 / 2.42e-3 | 9.20e-4 | 5.02e-4 |
| K + V | 8 | 99.58 / 1.32e-3 | 6.07e-4 | 5.05e-4 |

V is free at block 32. K's error is dominated by blocks with a large channel (an outlier makes every other element of the block coarse),
so it falls with the block size, and a block of 4 is the first that passes. (A "128" row in the raw data is a block of 32: the probe
kernel's absmax shuffles stop at the simdgroup; ignore it.) The real int8 decode KL is a little worse than the probe's at the same
block (1.5e-3 vs 1.3e-3 short): not investigated, within what 128 positions can resolve; the f64 kernel tests do not show a defect.

### Greedy ids against bf16 KV (256 tokens, 6 prompts, temperature 0)

`raw/.../ids/summary.txt` (first divergence position = index of the first generated token that differs, 0-based):

| prompt | fp8 | int8 |
|---|---|---|
| short | token 1 | token 49 |
| 4k | token 1 | identical (256) |
| open (chat) | token 98 | token 243 |
| grounded (chat) | identical | identical |
| golden-docs | token 14 | token 58 |
| golden-code | token 85 | identical |

Both diverge as expected; fp8 diverges at token 1 on short and 4k (the first sampled tokens already sit on near-ties), int8 diverges
later or not at all. The default path against the `spike/metal` binary: **identical ids on all 6 prompts at 256 tokens** (section 7).

### 16k and above: not covered by any golden

The goldens stop at 4k. **Nothing here shows the accuracy of either mode at 16k or 32k**, and the error grows with context (K noise
adds across more keys, though softmax also averages it), so 4k numbers are not an upper bound. What exists is a weak proxy
(`long16k/`, bf16 vs quantised at the last prompt position of the 16 384 and 32 764 token profile prompts, top-50 logits, 129 greedy ids):

| context | KV | last-position top-1 | common ids in top-50 | max abs logit shift | 129 greedy ids vs bf16 |
|---|---|---|---|---|---|
| 16384 | fp8 | same (id 220) | 49/50 | 0.44 | first divergence at token 24 |
| 16384 | int8 | same | 48/50 | 0.25 | identical |
| 32764 | fp8 | same | 46/50 | 0.40 | identical |
| 32764 | int8 | same | 50/50 | 0.20 | identical |

The top-1 token is the same in every run and the KL over the common top-50 is 5e-12 to 1e-10, i.e. the position is dominated by one token and the KL says
nothing; the logit shift (0.2 to 0.4 on the leading logits; no bf16-vs-bf16 reference shift was measured to compare it with) and the greedy ids are the only signal, and one
prompt, one position, 129 tokens is no accuracy measurement. Proposal to cover it properly, in order of cost:

1. **Self-golden at 16k / 32k (cheap, no new reference)**: `intarsia accuracy --golden <16k prompts> --write-self-golden` with the bf16
   build, then `--self-golden` with `--kv-dtype fp8|int8`: teacher-forced KL of the quantised cache against our own bf16 over every
   prefill position and the decode window (`accuracy/selfgolden.rs` already does this for build-to-build comparisons). It measures the
   quantisation cost directly, bf16 being only 3e-4 from fp32 at 4k. Needs a 16k / 32k prompt file (the profile prompts will do).
2. **fp32 HF golden at 16k**: `tools/golden.py --topk 20` on spark1 with the long prompts, the same top-20 format as the 4k golden (an fp32
   8B at 16k is about 40 GB of activations plus attention; fits spark1's 116 GB). Gives the distance to the reference, as the 4k golden does.
3. A needle-in-a-haystack / long-QA pass (localeval's long-context suites) as the end-to-end check, since KL on near-deterministic text is
   a poor stand-in for retrieval.

## 6. Speed

Engine side: `intarsia generate`, `--max-seq 40960 --prefill-chunk 2048 --max-tokens 129 --ignore-eos`, profile prompts of 4096 / 16384 /
32764 / 40000 tokens, one exclusive fleetq slot (#2075), order **bf16, fp8, int8, int8, fp8, bf16** per context, macmon 100 ms samples
(`/Users/josef/bin/macmon`), GPU clock = median over the busy samples of the decode window, flagged when the median is under 1450 MHz or
40 percent of samples are. Raw: `speed2/` (`summary.txt` has the clock columns for every run). tok/s is decode only.

| context | bf16 | fp8 | int8 | fp8 vs bf16 | int8 vs bf16 | decode clock (MHz) | usable? |
|---|---|---|---|---|---|---|---|
| 4096 | 35.25 / 35.32 | 35.27 / 35.39 | 35.35 / 35.44 | +0.2 % | +0.3 % | 1328 to 1376 | flagged (<1450); runs agree to 0.4 % |
| 16384 | 31.91 / 31.93 | 32.78 / 32.85 | 32.63 / 32.74 | **+2.8 %** | **+2.4 %** | 1461 to 1484 | **yes** |
| 32764 | 28.00 / 24.60 | 25.51 / 25.17 | 26.92 / 26.01 | n/a | n/a | 790 to 1475, collapsing | **no**: the clock fell from 1474 to ~800 MHz during the series |
| 40000 | 24.75 / 24.37 | 26.58 / 27.30 | 26.48 / 27.42 | ~ +8 to +12 % | ~ +8 to +12 % | 1050 to 1280 | flagged; clock is steady-low here, so the quantised-vs-bf16 gap is indicative |

The 32k series (cooldown gaps up to 90 s at 72 C) hit the throttled state the profile doc describes (790 to 1100 MHz) from the second run
on; its ordering is clock, not dtype, and is **not** a result. The 40k series is throttled throughout (1050 to 1280 MHz, same state
for all three dtypes); relative numbers are indicative only. A re-run of 32k with a stricter temperature gate was prepared
(`workH.sh`) but did not fit the queue before the deadline.

Why the gain is small, checked at the kernel level (`attn_bw`, one K/V cache per layer so reads come from DRAM, per call, `slot1c/`,
`slotD/attn_bw.txt`, the policy default):

| context | bf16 us (GB/s) | fp8 us (GB/s) | int8 us (GB/s) | kernel time saved |
|---|---|---|---|---|
| 4096 | 36.2 (464) | 30.7 (274) | 30.9 (288) | -15 % |
| 16384 | 121.5 (552) | 91.7 (366) | 92.1 (387) | -24 % |
| 32768 | 234.0 (574) | 172.1 (390) | 172.2 (414) | -26 % |

The bf16 kernel is at 90 to 93 percent of the 614 GB/s peak; the quantised ones are at 60 to 67 percent. Halving the bytes should halve
the time; the dequant ALU work and the smaller loads per lane (8 bytes instead of 16) are what stands in the way. Prediction from these
numbers at 16k: 36 layers x 30.4 us = 1.09 ms of the 31.3 ms token = +3.6 % (measured +2.4 to +2.8 %); at 4k: +0.7 % (measured +0.2 to
+0.3 %); at 32k: 36 x 62 us = 2.2 ms of 35.7 ms = +6.5 %. The consistency of prediction and measurement says the engine-side number is
right and the shortfall against +7.5 / +14 / +15.6 % is the kernel, not the engine. Splitting policy (`INTARSIA_METAL_SPLITKV`, min keys /
target threadgroups 128/120 default vs 128/240 and 128/360): the default was fastest for the quantised kernels too (a 64/480 and a 256/120 point were queued but the sweep log is incomplete).

What would close the gap: a wider load per lane for fp8 (16 bytes = 16 elements per lane, D/16 lanes per key) and a cheaper decode
(e.g. packed `bfloat2` / half2 arithmetic for the int8 path). None tried.

Prefill (same runs): 16k, tok/s bf16 2909 / 2961, fp8 2957 / 2961, int8 2872 / 2899: **no change beyond noise (+/- 2 %)**; the staging
decode is invisible. 32k and 40k prefill are clock-confounded (bf16 32k: 15.8 s at 1428 MHz vs 28.3 s throttled).

### localeval speed (`local-model-bench`, `localeval speed --sizes 4096,16384 --gen 256 --reps 3`, through `intarsia serve`, merged binary)

localeval supports the 16k size. One server per KV dtype, `--max-seq 20480`, order bf16, fp8, int8 (x2, failed to start, see below), fp8, bf16, one slot
(#2081), decode tok/s median (min to max of 3 reps); raw `localeval2/` (`result.json` paths in each `localeval.out`):

| prompt tokens | bf16 run 1 | fp8 run 1 | fp8 run 2 | bf16 run 2 |
|---|---|---|---|---|
| 4088 decode | 35.1 (34.8 to 35.3) | 35.2 (35.1 to 35.6) | 35.5 (35.3 to 35.5) | 35.4 (35.3 to 35.5) |
| 16654 decode | 31.7 (31.6 to 31.8) | 32.6 (32.6 to 32.8) | 32.1 (32.0 to 32.1) | 31.6 (31.2 to 31.9) |
| 16654 TTFT s | 6.28 | 5.86 | 7.33 | 6.19 |

fp8 decode at 16k is +1.5 to +3 % over bf16 (32.6 and 32.1 against 31.7 and 31.6), at 4k +0.3 % to +1.1 %: the same as the engine-side numbers.
The 16k TTFT of the second fp8 run (7.33 s) is a throttled run (prefill 2271 tok/s), not a regression; the first fp8 run's 5.86 s
is faster than both bf16 runs, so prefill is a wash within the clock noise, as in section 6. `fp8` through `serve` loaded and admitted
cleanly (`--kv-dtype fp8` under `require_admitted`: the fp8 kernels are admitted against the oracle).

**int8 through `serve`: found broken, fixed, verified.** Both int8 server starts in the run above died at load with "no kernel for
attn_decode ... w=I8". Cause: `serve` loads with admission on, `plan::admit` builds its cases from the model's own calls and the cpu oracle
cannot run a block-scaled cache, so the int8 kernels are Rejected. Fix (`ModelArgs::for_serving`): an int8 cache is served without
admission, with a warning on stderr. Rerun in a short slot (#2093, `localeval3/`, merged binary + the fix, order int8, bf16, int8):

| prompt tokens | bf16 | int8 run 1 | int8 run 2 |
|---|---|---|---|
| 4088 decode | 35.3 (35.2 to 35.6) | 35.3 (33.5 to 35.5) | 35.6 (35.6 to 35.7) |
| 16654 decode | 31.8 (31.0 to 32.0) | 32.6 (31.9 to 32.6) | 32.4 (32.3 to 32.7) |
| 16654 TTFT s (prefill tok/s) | 6.01 (2772) | 6.61 (2522) | 6.15 (2709) |
| 4088 prefill tok/s | 3923 | 3093 | 3545 |

int8 decode at 16k is +1.9 to +2.5 % over bf16 (engine-side +2.4 %: agrees). The int8 servers' **prefill looks 0 to 10 % slower** here (first
server of the slot, the 4k prefill figure carries ranges down to 2853 tok/s): not reproduced engine-side (16k prefill 2872 / 2899 tok/s int8 against
2909 / 2961 bf16, 4k 3491 / 3474 against 3429 / 3419, section 6), so I read it as warm-up and clock noise, but a dedicated
A/B of int8 prefill (the staging decode is the only extra work) was not run.


## 7. Gates of the default path

All on the **merged tree** (branch merged with `spike/metal` `912f3f8`, cb-m1 batching; one conflict, `common.rs` `for_serving`, resolved
by keeping both changes), slot #2081, binary `intarsia-merged`; raw `gatesM/ idsM/ accM/ qualifyM-bf16/`. The same gates ran
earlier on the pre-merge branch (`gatesF/ gatesG/ idsG/ qualifyG-bf16/ acc/`) with identical numbers.

| gate | result |
|---|---|
| `cargo test -p intarsia-core -p intarsia-cli -p intarsia-server --release` | pass (no GPU) |
| `cargo test -p intarsia-backend-metal -p intarsia-models --release -- --test-threads=1`, in the slot | **pass, 198 tests, 0 failed**. Includes `kv_fp8` (8), `kv_i8` (7), the existing g0b / multirow / wrap_host suites, and the batch bitwise identity on real Qwen3-8B (below) |
| `intarsia kernels test --backend metal` | **exit 0, 33 kernels, 388 cases, all ADMITTED.** The fp8 kernels are admitted against the f64/cpu oracle (`kv_append_f8`, `qk_norm_rope_kv_f8`, `attn_decode_f8_splitkv` 15 cases max_abs 1.5e-5, `attn_prefill_f8_flash` 14 cases 5.2e-3, `_flash_nax` 7 cases 4.8e-3, `attn_prefill_f8_splitkv_rows` 9 cases 1.0e-3: all inside the bf16 tolerance). The int8 kernels are not listed here by design (section 3) |
| greedy ids, bf16 KV, merged binary vs the `912f3f8` binary, 256 tokens | **identical on all 6 prompts** (short, 4k, open, grounded, golden-docs, golden-code). Pre-merge branch vs the `56ec2d5` binary: also identical on all 6 |
| `qualify --backend metal` (fresh, full, default KV) | **PASS**, 13 stages: 11 pass, 2 skip (`ab`, `tp`: no reference server / tp 1). acc-short 100.00 % / KL 2.83e-4; acc-long 99.72 % / 3.70e-4; acc-4k 99.28 % / 2.44e-4 (identical to the baseline); bench 36.2 tok/s at 128 tokens (89 % of the byte floor) |
| `cargo check --workspace --all-targets` | clean |
| `cargo check -p intarsia-cli --features cuda` / `--features hip` | clean |

The one early failure is worth recording: the first `qualify` run (pre-merge, before the int8 listing fix) **failed at stage 4**
(kernels test: "NO CASES (untested)" for the int8 kernels, which the oracle cannot admit). Fixed by listing them only when
`--kv-dtype int8` is requested; the rerun above passes. A matching serve-side find: `serve` loads with `require_admitted`, so
`--kv-dtype int8` found no attention kernel; `for_serving` now relaxes `require_admitted` for int8 only.

**Batching (cb-m1) with a quantised cache.** The batched step runs the 1-row quantising prologue and the 1-row decode kernel
per sequence on that sequence's own cache (`attn_rows` takes the cache dtype from the slot like `attn_block` did), and
`plan::check_batch_ops` counts ops, not dtypes, so no new case was needed. Checked, not assumed: `batch_metal.rs` gained fp8 and int8 variants
(`batch_bitwise_equals_solo_steps_qwen3_8b_metal_{fp8,int8}_kv`): **138 (row, step) pairs for B = 2, 3, 4, 6, 8 at mixed lengths up to 4000,
and 48 decode rows beside 256-token prefill chunks, bitwise identical to the solo steps, for fp8 and for int8** (12.6 to 13.5 s each).
So batching with a quantised cache is supported and exact; there is nothing to refuse.


## 8. Shared-code changes (everything outside the new files)

* `crates/intarsia-core/src/engine.rs`: **none.**
* `model.rs` (+18): the KV dtype check in `Model::load` (fp8 needs cpu / metal, int8 needs metal and `head_dim % 32 == 0`), and the
  per-token KV bytes now go through `kv_plane_bytes`. No change to `attn_block`, `forward`, or any op emission.
* `state.rs` (+15): `KV_I8_BLOCK`, `kv_plane_bytes`, used by `SequenceState::new` and `used_bytes`. `plan.rs` (+2): same function.
  For bf16 / fp8 / f16 / f32 the result is the old expression exactly.
* `admission.rs` (+18): one extra bf16-activation / fp8-cache case for `QkNormRopeKv` (so the fp8 prologue is admitted); the other
  kernels' cases are unchanged.
* `intarsia-cli/src/common.rs` (+25): backend / dtype support check, int8 flag handling, help text. `recipe.rs` (+2): text.
* Metal backend: new kernels, launcher branches in `dense.rs` / `attn.rs` / `flash_nax.rs` / `lib.rs` (a `kv_stage` buffer), `kernels.rs`
  table entries, `bench.rs` helper for the conversion tests, `attn_bw` example. New shader files: `fp8_conv.metal`; additions to
  `common.h`, `dense_ops.metal`, `qk_rope_kv.metal`, `attention_splitkv.metal`. Existing bf16 entry points are textually untouched
  except that their templates gained defaulted parameters (`F8 = false, UN = 4, I8 = false`); the bf16 instantiations are the same
  kernels (greedy ids identical to the base binary on 6 prompts).
* A probe, not a feature: `INTARSIA_METAL_KV_FAKEQ` and the `qk_norm_rope_kv_fq*` kernels (bf16 cache, fake quantisation). Remove
  before merging if unwanted.

## 9. What to do next

1. Do not merge as a user-facing speed feature yet: +2.4 to +2.8 % at 16k for an accuracy loss the bar rejects.
2. If a quantised KV is still wanted: **K bf16, V int8 / block 32** (23 % fewer KV bytes, V is free at that block size) is the one
   configuration the data supports. It needs per-plane dtypes in `SlotSpec::AttnKv` (core change to the slot spec and sizing, the
   kernels are all here: the V half of `ask_step_i8`, `kv_append_i8`). Expected kernel gain about -11 %, engine +1 to +2 % at 16k, so
   mainly a memory feature (context length per GB), not a speed one.
3. K needs a smarter quantiser than a block absmax: per-channel scales (KIVI style: K scales per channel over groups of tokens) or a
   Hadamard rotation before the block quantiser, both need the probe harness here (`INTARSIA_METAL_KV_FAKEQ`) to evaluate first.
4. Bring the decode kernel to the bandwidth limit (wider loads). Without it the byte reduction does not reach the engine.
5. Add the 16k self-golden (section 5) to `qualify` before any quantised KV is allowed near a default.

## 10. Commits and fleetq slots

Branch `metal-kv-quant`, first-parent history from `spike/metal` `56ec2d5` (the final docs commit that contains this file is the
branch head; `912f3f8` merged in at `b3799f2`):

* `17f94ccc` metal: fp8 e4m3 KV cache kernels (KvAppend, QkNormRopeKv, split-KV decode + rows, prefill via exact staging decode) + CLI --kv-dtype guard
* `ef35bcde` metal fp8 KV: half-based exact e4m3 decode, units-in-flight tuning variants, backend refusal for fp8 KV outside cpu/metal
* `964b0ed3` kv-quant: slot scripts
* `39ec46a0` kv-quant: long-context proxy script
* `f6ab8fb2` kv-quant: sizing step
* `8a70b3fa` kv-quant: fake-quant int8 accuracy probe (INTARSIA_METAL_KV_FAKEQ), probe slot script
* `648843d9` kv-quant: trim slot B, slot C template
* `227fdb46` metal: int8 block-scaled KV cache (core sizing, append/prologue/decode/rows/prefill-staging kernels, tests)
* `98b6f04f` kv-quant: slot D script, int8 attn_bw
* `c2d7edcb` kv-quant: slot E (speed ABBA bf16/fp8/int8)
* `38fbc6f7` kv-quant: slot C for 3 modes
* `a1260c41` kv-quant: split policy sweep in slot E
* `9c8b310e` kv-quant: one slot F+E+C, int8 results
* `5ac06eda` metal: list the int8 KV kernels only when an int8 cache is requested (cpu oracle has no block-scaled cache; keeps kernels test green)
* `69bfbe2f` kv-quant: slot G (final gates rerun), trim C
* `9fe810ac` docs: kv-quant results draft
* `1c4ef307` cli: serve with --kv-dtype int8 does not require admission (int8 kernels have no oracle cases)
* `3d9cfb90` kv-quant: slot G uses final binary, localeval part
* `6e7889b9` kv-quant: fix timestamps
* `9fbf4f04` docs: kv-quant proxy table
* `8f2d748d` tests: kv_i8 requests the int8 kernels explicitly
* `b3799f26` Merge spike/metal 912f3f8 into metal-kv-quant (cb-m1 batching); resolve common.rs for_serving
* `33549d27` tests: batch bitwise identity with fp8 / int8 KV caches
* `8d44325c` kv-quant: slot M (merged-tree gates)
* `1446ace4` docs: kv-quant gates on merged tree
* `8006242a` cli: serve with --kv-dtype int8 runs without admission (oracle fails the model-shaped cases); slot I script
* `bc963092` docs: kv-quant localeval + int8 serve status

fleetq slots (all `--host endor --exclusive --mem 60`, all released, none forced; label prefix "metal kv quant"):
#2048 (first fp8 kernel tests; failed on the f32 denormal flush, led to the half-based decode), #2052 (fp8 kernel tests pass), #2056 (decode
variants microbench), #2058 (default-path gates, fp8 accuracy, ids, qualify bf16 PASS and fp8 acc stages), #2059 (speed ABBA bf16/fp8,
fake-quant probe, sizing), #2064 (int8 tests, more probe configs, int8 accuracy and ids), #2075 (final-ish gates incl. the qualify that
failed on the int8 listing, speed ABBA bf16/fp8/int8, sizing, split sweep, 16k / 32k proxy, first localeval where int8 serve failed),
#2081 (gates on the merged tree, localeval), #2093 (serve int8 check).
#2088 was queued for the merged-tree gates and cancelled before it ran (the same work went into #2081). Not run for lack of slot time:
the 32k ABBA re-run with a stricter thermal gate (`workH.sh`).
