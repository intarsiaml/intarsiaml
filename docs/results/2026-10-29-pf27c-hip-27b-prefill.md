# Lane PF27c: Qwen3.8-27B MXFP4 W4A8 tp=2 prefill on R9700s after the push (HIP, 2026-10-29)

Branch `lane-pf27c/gemm-eff`, on main `ce2546b4b` (PR #239, `INTARSIA_HIP_AR_PUSH` on). tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`,
W4A8, MTP k2, `prefill_chunk 2048`. Radiance was stopped for every GPU job (one fleetq job holding both cards, `--takes-down radiance`).
Raw data: `docs/results/2026-10-29-pf27c-hip-27b-prefill/raw/` (m* = short checks, w1 / w2 = gate windows). Scripts: `scripts/pf27c/`
(copies of `scripts/arx/` plus `knames.py`, `gaps.py`, `gapapi.py`, `pushseq.py` for the traces).

**Result: one exact change, the W4A8 span cache (default on, `INTARSIA_HIP_MXFP4_A8_SPANCACHE=0` turns it off), worth +0.7 to +3.3 % of a
warm served prefill (W2 below). It does not reach the +5 % gate at 4k: draft PR. The attribution says where the remaining time is, and the
probes rule out the push-side and launch-merge ideas (sections 2 and 3).**

## 1. Attribution on main at 4k (rocprofv3, rank 0, `generate` of `4k-0`, 3858 tokens, two chunks; raw/m1)

Span 822 ms (4692 tok/s in the trace run), kernels 787 ms, idle 35 ms.

| Bucket | ms | Notes |
|---|---:|---|
| W4A8 GEMM, plain (direct 2824 tile, + 2224 for the GDN b / a gates) | 280.0 | 544 + 192 launches; gate / up at 726 us = ~251 TFLOP/s |
| W4A8 GEMM, pushing (down, GDN out, o_proj) | 234.1 | 256 launches; down 1047 us, GDN out / o_proj 770 us (raw/m1 `pushseq.py`) |
| all-reduce (`ar2p` after the push, `ar2ll` small) | 24.9 | was 208 ms before #239 |
| GDN (`gdn_delta_rows_v3` 107.8, conv 8.4, gated rmsnorm 6.7) | 122.9 | token-sequential scan, 192 one-wave blocks per call |
| activation quant (`mx_a8_quant_tiled` 25.5), small-N rows GEMM (12.0), span pre-pass (`a8_wspan` 4.0) | 41.5 | |
| attention (flash 31.1, qk-norm / rope / kv 4.2, sigmoid-mul 1.6) | 36.9 | |
| norms / elementwise (rmsnorm-add 21.2, silu-glu 18.3) | 39.5 | |
| lm head, fills, copies, empty per-group launches | ~7 | 800 launches of the unused fold path exit at once (1.8 ms + their gaps) |
| idle | 35.2 | 4.3 ms of it is a GPU-side 260 us bubble after each chunk-2 flash (the host queued the next kernel ~400 ms earlier, raw/m10 HIP API trace) |

Against PF27b's split (GEMM 414, AR 208): the push moved ~180 ms of all-reduce into the pushing GEMMs (GEMM total 514 ms, AR 25 ms).

At 16k (`16k-0`, 15469 tokens, raw/m1): span 3503 ms; plain GEMM 1076, pushing GEMM 936, flash 441 (128 x 3.44 ms, ~53 TFLOP/s causal),
GDN delta 433, quant 103, ar2p 91, rmsnorm 85, silu 73, idle 107. Flash grows from 4 % (4k) to 13 % (16k) of the prefill: it is the
16k-specific gap to Radiance (0.72x), next to the GEMMs.

## 2. Probes that ruled things out (short checks)

- **Pushing GEMMs are link-bound.** `tests/ar_push.rs push_gemm_timing` (new, ignored; raw/m4, m8, m9) at the 27B's row-parallel shapes,
  M = 2048, N = 5120, per call on both ranks:

  | K | op alone | plain op + AR | push op + reduce | push into local memory (diag) | push, last tile not pushed (diag) |
  |---|---:|---:|---:|---:|---:|
  | 8704 (down) | 860 us | 1686 us | 1262 us | 1117 us | 1205 us |
  | 3072 (GDN out / o_proj) | 327 us | 1153 us | 943 us | 501 us | 902 us |

  The 21 MB partial needs ~830 us of link at ~25 GB/s (ARX's measured rate), so at K = 3072 the push cannot hide behind a 327 us GEMM, and
  at K = 8704 only partly. A normal (non-persistent) push grid was slower (2098 / 1453 us, m8); the 8-wave tile for the push (DTILE 2,4,2,4)
  was slower end to end (847 vs 822 ms span, m3). Under exactness the bytes are fixed: no more push variants are worth a window.
- **One launch per call / per pair** (both fold paths in one kernel, gate | up as one grid; patch in raw/m7): bitwise (hashes 48/48 under
  both paths, m6), kernels 5 ms faster, but a 242 us GPU-side gap appeared before every following `silu_glu` from layer 3 on, on both ranks
  (31 ms, raw/m6 / m7; gone with the pair merge off). Net slower (830 vs 813 ms). Not kept. Putting both paths into one kernel also costs
  registers: one kernel holds the larger path's VGPRs (fold 180, per-group 217 -> 6 waves per SIMD); quarter tiles for the per-group path
  bring it back to 180 / 8 waves without scratch.
- **GDN**: the per-element order of the token-sequential recurrence fixes its bits (a chunked / WY form or more lanes per dot change the
  sums), so no exact restructuring was attempted.

## 3. The change: W4A8 span cache (`crates/intarsia-backend-hip/src/device.rs` `SpanCache`, `kernels/hip/mxfp4_a8_gemm.hip`)

Every `matmul_mxfp4_a8_mma` / `_pair_` call ran a span pre-pass over the weight scales (memset + `a8_wspan_kernel`: wref = each row's
largest scale byte, fold flag) and then launched both fold-path kernels, the unselected one exiting at once: 640 + 1600 launches per 4k
prefill. The span result is a pure function of the scales, so the backend now computes it once per scale tensor (`ik_mxfp4_a8_spans`, into
a pooled device buffer), copies the flag back without a wait into a page-locked slot, and from then on a call launches only the quant
pre-pass and the selected path's kernel (`ik_matmul_mxfp4_a8_mma_cached`, `ik_matmul_mxfp4_pair_a8_mma_cached`). Until the flag has
arrived (the host runs ~400 ms ahead of the GPU) both kernels are launched as before. Scope: the direct kernel (M > 64), eager, stream 0;
graphs, side streams and the staged kernel keep the per-call pass. Staleness: every `HipBuffer` has a never-reused id and a write count
(upload, copy, zero, relayout, launch outputs); a write to the scales buffer refills the entry. Same kernels, same wref and flag values:
bitwise the uncached call.

Cold single-prompt `generate` (one process, raw/m11): no gain (813 vs 818 ms at 4k), because the first chunk pays the fills (496 scale
tensors per rank, one extra D2H each) and the second chunk's calls are queued before the flags land. A server is warm after its first
request: the W2 numbers below are served.

## 4. Gates (W1, raw/w1; W2, raw/w2)

| Gate | Result |
|---|---|
| greedy ids, 18 prompts x (spec off, MTP), base vs dev | 36 / 36 SAME |
| `a8_mma_hashes` (48), default and `PERGROUP=1` | 0 differing, both (m11) |
| W4A8 tests (`a8_`, incl. new `a8_span_cache_reuses_and_refills`), default / `PERGROUP=1` / `DIRECT=0` | 8 passed each (m11) |
| `ar_push`, `comm`, `ar_norm` tests (default, `AR_DRAIN=1`, `AR_WIDE=0`) | all pass, negative controls fire |
| `kernels test --backend hip --quick` | exit 0, 75 admitted |
| `plan show` 8B / 35B | byte-identical |
| 18 served texts (md5 of 256 greedy tokens, MTP server) | 18 / 18 identical |
| Mac `cargo test --workspace --release` | pass (1104 tests, 0 failed) |

**Served A/B** (W2, localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, seeded nonce `scripts/pf27c/leval.py`; MTP 3 interleaved
rounds per arm, spec off 1 round per arm; median of the per-run medians, tok/s; `raw/w2/speedtab.txt`):

| | main 1k | dev 1k | | main 4k | dev 4k | | main 16k | dev 16k | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill, MTP | 4182 | 4319 | +3.3 % | 4746 | 4780 | +0.7 % | 4236 | 4243 | +0.2 % |
| prefill, spec off | 4240 | 4343 | +2.4 % | 4756 | 4800 | +0.9 % | 4242 | 4305 | +1.5 % |
| decode, MTP | 93.1 | 93.0 | -0.1 % | 88.3 | 88.3 | 0.0 % | 74.9 | 76.6 | +2.2 % |
| decode, spec off | 55.8 | 55.8 | +0.1 % | 54.9 | 54.9 | +0.1 % | 52.6 | 52.0 | -1.2 % |

Decode never takes the changed path (M <= 8 / MTP verify rows go to the GEMVs); the 16k spec-off -1.2 % is one round per arm and the MTP
16k row moves the other way. Gates: prefill +5 % at 4k NOT met (+0.7 %); no length slower: met; decode within 0.5 %: met at 1k / 4k,
16k within single-round noise. Hence a draft PR.

## 5. What would move the gate (not done here)

1. **Fused all-reduce + rmsnorm-add + activation quant for prefill** (`ar2p` -> `rmsnorm_kernel<true,true>` -> `mx_a8_quant_tiled`, 256
   times per 4k prefill, ~190 us together): one pass that keeps the rounding points (bf16 sum, the norm's reduction order, per-token fp8
   scale) would save the re-reads and two launches, est. 20-25 ms at 4k (2.5-3 %). Needs the pair GEMM to accept a pre-quantised input.
2. **Chunk wavefront**: run chunk c+1's layer l - 1 beside chunk c's layer l on a second stream, so the GDN scan (192 one-wave blocks, the
   card mostly idle) and the link-bound push overlap compute. Same kernels and rows, so exact, but a model-level change (two workspace
   sets, a deterministic comm order on both ranks).
3. **16k**: the flash kernel at ~53 TFLOP/s (441 ms at 16k) is the length-specific gap; a faster schedule with the same key-block order
   (BK 32, same online-softmax sequence) would be exact.
