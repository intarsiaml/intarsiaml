# MNQ: the producer of an fp8 GEMV's input also writes q(x) (Metal decode, 2026-10-29)

Lane `mnq`, branch `lane-mnq/metal-norm-fp8`, merged with `origin/main` `694b519cf` (#258). endor (Apple M5 Max, 128 GB), Qwen3.6-35B-A3B
NVFP4 (MTP off), Qwen3-8B BF16 for the one quick arm. All GPU runs inside fleetq slots (`--quiet gpu,cpu`).

**Result: bitwise identical, but not faster. Default OFF (`INTARSIA_METAL_NORM_FP8` unset = main). Draft PR, not for merge.**
The 35B 4k decode target (+3 % median) is missed: paired 10x medians are 121.6 tok/s (main), 120.2 (knob 1), 121.8 (knob 0).

## What was built

- `kernels/metal/common.h`: the e4m3 helpers (`fp8_enc`, `fp8_dec`, `fp8_qx`, ...) moved out of `fp8.metal` unchanged, plus `fp8_q_bf16`
  (q(x) of the stored bf16, exactly the GEMV's staging expression).
- Producers with a q(x) output (`dense_ops.metal`, `gdn.metal`): `silu_glu_bf16_q`, `sigmoid_mul_bf16_q`, and the norms
  `rmsnorm_bf16_exq{1,16,32}[_f]`, `rmsnorm_add_bf16_exq{1,16,32}[_f]`. The bf16 output is written as before.
- Consumer: `fp8_gemv_{bf16,f32}_m1_xd` (`fp8.metal`): the one-row GEMV reading q(x) from device memory, no staging pass, no
  threadgroup memory. Same fma chain as `_m1`.
- `crates/intarsia-backend-metal/src/prequant.rs`: the input scale belongs to the consumer, so a one-row producer launch is *held* (at most
  one) until the next launch. If that is an M=1 `matmul_fp8_gemv` on the producer's output with `is > 0`, the producer is encoded as the q
  variant and the GEMV as `_m1_xd`; any other dispatch/commit/flush/hold/discard encodes the producer plain first (stream order unchanged).
  A slot (16 KiB) keeps q(x) of one tensor with the scale bits it was made with, so the second/third GEMV on the same x (GDN z after qkv,
  attention k, v after q) reuses it; any write overlapping x (`pq_note_writes`) or host write/copy/zero drops it. Profiling runs never hand off.
- Knob `INTARSIA_METAL_NORM_FP8`: unset/`0` off (= main); `1` silu_glu + sigmoid_mul producers; `2` also the row norms.
  `INTARSIA_METAL_NORM_FP8_STATS=1` prints claim/reuse counts at exit.

## Correctness (all pass, knob 1 unless noted)

| Gate | Result |
|---|---|
| Greedy ids, 256 tokens, 4 prompts (short, 4k x2, 16k): main vs knob 1 vs knob 0 | md5 identical on all 4 (and equal to the pre-merge run); **bitwise** |
| MTP (`--spec mtp`) on short and 4k-a | knob-1 MTP ids = main MTP ids = spec-off ids (same md5) |
| EOS-terminated run (no `--ignore-eos`) | identical output (708 bytes) |
| `tests/norm_fp8.rs` (new): producer + GEMVs back to back vs the same with a synchronize between (= main's staging path), shapes incl. [heads,dv] / [1,heads,d] views, offsets, NaN, saturating and tiny inputs, bf16/f32 outputs, guard bytes, other-scale GEMV, write-to-x drop | 4 pass at knobs 1 and 2; negative control (scale q by 1.01) fails as it should |
| `rows_exact`, `row_tables`, `token_feedback`, `fused`, `dense`, `lowlat`, `qkv_fold`, `multirow`, `gdn` (Metal crate) | all pass; `rows_exact` also at knob 0 |
| `batch_metal_real` (35B batched = solo), `batch_recurrent` (14), `exact_reuse` (8) with `INTARSIA_TEST_REAL_MODEL=1` | pass |
| `kernels test --backend metal` | 48 kernels, 238 cases, all ADMITTED. The new kernels are launch variants, not registry entries, so they are covered by `norm_fp8.rs`, not admission |
| `plan show` (35B, metal), main vs knob 1 vs knob 0 | 0 diff lines (variant choice happens at launch) |
| `cargo test --workspace --release` on the merged tree | exit 0, 1113 passed, 0 failed |

Not run: `batch_metal_real` opt-in `real_35b_batched_engine_long_prompts_*` (ignored by default); the `rows_exact` hard-wired M=1..8 tests do not
exercise the handoff (it is M=1 only, rows launches are untouched).

## Speed

Decode noise on endor is about 5 % per run (some rounds drop 15 % for a round: see round 1 in the final tables), so medians and paired
alternation are used. Raw: `docs/results/2026-10-29-mnq-metal-norm-fp8/raw/`.

### Paired 35B 4k decode, 256 forced tokens, `generate`, 10 interleaved triples (`2026-10-29-mnq-metal-norm-fp8/raw/final/pair.log`)

| arm | median tok/s | vs main |
|---|---:|---:|
| main (694b519cf) | 121.57 | - |
| knob 1 | 120.16 | -1.2 % |
| knob 0 (same binary) | 121.81 | +0.2 % |

### Served A/B through localeval, 3 rounds interleaved M D D M M D (`2026-10-29-mnq-metal-norm-fp8/raw/final-ab/`, table `raw/final/ab-table.md`)

Final (merged main), median of rounds:

| cell | main | knob 1 | change |
|---|---:|---:|---:|
| 35B decode 1k | 124.3 | 122.7 | -1.3 % |
| 35B decode 4k | 106.3 | 107.3 | +0.9 % |
| 35B decode 16k | 101.6 | 99.5 | -2.1 % |
| 35B prefill 1k | 4587 | 4580 | -0.2 % |
| 35B prefill 4k | 3762 | 3767 | +0.1 % |
| 35B prefill 16k | 3130 | 3144 | +0.4 % |
| burst `--max-batch 16`, 4 streams (agg) | 186.3 | 186.9 | +0.3 % |
| burst 8 streams | 248.0 | 248.7 | +0.3 % |
| burst 16 streams | 236.9 | 236.5 | -0.2 % |

(Round 1 of the single-stream cells has a 15-20 % dip on both arms; medians absorb it. The 1-stream burst cell is the 1k row of the
`burst2` table in `2026-10-29-mnq-metal-norm-fp8/raw/final/ab-table.md`: 124.4 vs 123.7, -0.6 %.) No prefill or burst cell is slower than -1 %; the decode cells at 1k
and 16k are -1.3 % and -2.1 %, inside the ~5 % decode noise.
The pre-merge run (main 0d02d3f36; `2026-10-29-mnq-metal-norm-fp8/raw/pre-merge-ab/`) agrees within noise except two cells that were noise: 4k decode -5.4 % (one 114 round on each side)
and burst 8/16 -8.8 / -7.5 %, which a 5-round re-check (burst2) did not reproduce (-0.1 % / +0.3 %).
Qwen3-8B BF16 4k (one quick arm, pre-merge run, 2 rounds): decode 35.9 vs 36.0 tok/s (+0.3 %); the 8B has no fp8 GEMVs, so this only checks nothing else moved.

### Why it does not move: micro timings (`2026-10-29-mnq-metal-norm-fp8/raw/final/timing.log`, serial, weights rotated through 24 copies)

| sequence (us) | knob 0 | knob 1 | knob 2 |
|---|---:|---:|---:|
| fp8 GEMV alone 8192x2048 | 36.2 | 34.5-35.2 | 34.4-34.8 |
| 4096x2048 | 23.3 | 21.0-22.1 | 21.0-21.5 |
| 512x2048 | 6.7-6.8 | 5.2-5.3 | 5.2 |
| 2048x4096 | 24.4 | 19.4-20.5 | 19.4-19.5 |
| norm_add alone | 2.45 | 2.4 | 2.8 |
| norm_add + qkv GEMV | 38.9 | 38.9 | 44.5 |
| norm_add + z GEMV | 25.2 | 25.4 | 29.8 |
| norm_add + qkv + z (GDN in) | 61.7 | 61.7 | 66.2 |
| silu_glu + out GEMV | 23.8 | 21.0 | 21.0 |
| sigmoid_mul + o_proj | 23.6 | 21.0 | 21.0 |

- The staging cost is real (1.3 us on the 8192-row GEMV, 4 us on the 2048x4096 out projection). With the upstream q(x), the GEMV is
  1-4 us faster, matching MDEC's ~6 us staging estimate only for the largest.
- A norm that also writes q(x) costs more than the GEMV saves: knob 2 makes `norm_add + qkv + z` 4.5 us **slower** (61.7 -> 66.2), because the
  extra q write of the producer costs more than the staging it removes. The split between the producer's second store and the GEMV's
  device-memory read was not isolated.
- Knob 1 keeps only the winning pairs: the 30 GDN out projections and 10 attention o_projs, 2.8 us each, i.e. about 0.11 ms of the 8.3 ms
  token (1.3 %). That is the upper bound; the end-to-end medians (+0.9 % to -1.2 % at 4k) say it is inside noise. The extra host work per launch (hold,
  claim, drain) was not measured separately.
- To reach +3 % the three big GEMV groups (qkv/z, q/k/v) would need to profit from the norm-side write, and they do not on this GPU.

## Conclusion and next

- Same bits as main at every level, safe to keep behind a knob; default off since it is not faster. A draft PR carries the code for anyone
  who wants to build on it.
- Not tried: a dedicated tiny kernel that quantises x once per layer for all its GEMVs (adds a dispatch per layer).
