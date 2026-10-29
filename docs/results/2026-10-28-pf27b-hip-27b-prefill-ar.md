# Lane PF27b: Qwen3.8-27B MXFP4 prefill on the R9700s (HIP, tp=2): one quant pre-pass per GEMM pair, a wide prefill all-reduce (2026-10-28)

Branch `lane-pf27b/prefill-overlap`, built on origin/main `a35339fe9` (PR #228). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8, MTP k2 default, `prefill_chunk = 2048`. Base = a fresh build of `a35339fe9` (`pf27b-base-cli`), dev = this branch (`pf27b-dev-cli`), both built in this lane's volumes. Raw data: `docs/results/2026-10-28-pf27b-hip-27b-prefill-ar/raw/`, scripts `scripts/pf27b/`.

## Headline

| | |
|---|---|
| Asked | Close the R9700 27B prefill gap further: >= +5% at 4k (stretch: pass the vLLM-fork's 5399 at 4k). Items: (1) deduplicate the activation-quant pre-passes, (2) overlap the prefill all-reduce with compute, (3) the large-message all-reduce bandwidth. |
| Result | **prefill x1.053 / x1.072 / x1.058 at 1k / 4k / 16k** (localeval, MTP on, 4 base and 3 dev runs interleaved: 3738 / 4029 / 3758 -> 3936 / 4320 / 3976 tok/s), TTFT at 4k 1.014 -> 0.945 s. Decode unchanged (MTP off x0.999 / x1.000 / x1.000). Bitwise identical to main. |
| What did it | (1) `matmul_mxfp4_pair_a8_mma`: the four pair sites (GDN qkv\|z and b\|a, attention k\|v, FFN gate\|up) quantise their shared activation once (992 -> 640 pre-passes per 4k prefill), one span launch per pair, the fold flags zeroed by the pre-pass instead of 992 memsets, the 128-token tile for N <= 64. (3) the wide fence path: prefill all-reduces run one 64-block launch on their own 20 MiB slot instead of five 4 MiB launches on 16 blocks (20 MiB: 1020 -> 840 us; link floor ~750 us). |
| Overlap (item 2) | Not shipped. `INTARSIA_TP_OVERLAP=2` + `GPU_MAX_HW_QUEUES=4` + chunk 4096 on top of this branch: 888 vs 888 ms (no gain); it needs a process-wide 4-queue setting that the R9700 lessons tie to a ~20% decode loss. The all-reduce is now 208 ms per rank against a ~192 ms link floor: hiding it is the only lever left on it (section 6). |
| Gap | Radiance 1.12.0 5214 / 5994 / 5869 (R3): 0.75 / 0.72 / 0.68x (were 0.72 / 0.67 / 0.64x against this base). vLLM-fork 4k 5399, 16k 5305 (R123 / R126 **old**): 0.80x / 0.75x. Still 3/3; the stretch (5399 at 4k) is not reached. |
| Windows | 2 of 3 (W1 A/B + texts + mixids; W2 gates). 10 micro jobs under 3 min (m1-m10). Radiance answered on :8080 before every release (section 8). |

## 1. Attribution, 4k prompt, per rank (rank 0; `rocprofv3`, `generate --max-tokens 1 --spec off`, W2 traces `raw/w2/p-*.csv.gz`)

| bucket | base `a35339fe9` ms | dev ms | change |
|---|---:|---:|---:|
| span (profiled) | 949.7 | 895.7 | -54.0 |
| W4A8 GEMM | 416.6 | 414.0 | -2.6 |
| all-reduce | 236.1 (1408 launches) | 208.0 (384 launches) | -28.1 |
| GDN | 112.2 | 114.9 | +2.7 |
| activation quant + span + fills | 43.0 (992 + 992 + 992) | 29.2 (640 + 640 + 0) | -13.8 |
| norms / attention / elementwise | 95.7 | 97.2 | +1.5 |
| idle (gaps between kernels) | 43.9 | 30.1 | -13.8 |

Rank 1: span 949.4 -> 897.0, all-reduce 247.2 -> 222.0 (rank 1 waits for rank 0, whose GEMMs are ~13 ms slower). Plain runs (`generate`, 3 rounds each, micro m5 / m8): base 929-942 ms, dev 883-889 ms (x1.06 tok/s).

## 2. The link and the all-reduce (micro m1, `tests/comm.rs` `p2p_link_rate`)

Both cards sit behind PCIe 5.0 **x8** upstream ports (`01:00.0`, `05:00.0` trained at 32 GT/s x8; the x16 links are inside each card's switch), and P2P crosses the 9900X root complex. Per direction, both ranks at once:

| what | 4 MiB | 18.5 MiB (1810 rows) | 20 MiB (2048 rows) |
|---|---:|---:|---:|
| copy engine (`hipMemcpyAsync` into the peer) | 26.95 GB/s | 27.54 | 27.54 |
| push kernel alone, 16 / 64 blocks, with the system fence | 26.4 / 23.3 | 27.9 / 26.6 | 27.9 / 27.8 |
| local reduce alone, 16 / 64 blocks | 28.2 / 11.1 us | 111 / 34 us | 123 / 38 us |
| all-reduce, old path (16 blocks, 4 MiB slot) | 204 us (20.5 GB/s) | 909 us | 1020 us |
| all-reduce, wide 64 blocks, 20 MiB slot | 199 us | 798 us | **840 us (25.0 GB/s)** |
| wide 32 / 128 / 256 blocks, 20 MiB | 192 / 215 / 230 | 774 / 815 / 838 | 855 / 856 / 859 |
| wide 64 blocks, 4 MiB slot (blocks only) | 198 | 905 | 999 |

The link peaks at ~28 GB/s per direction and 16 blocks already push at that rate, so a better large-message kernel cannot raise the wire rate. What the old path lost: five launches per message, each with its own handshake and spin, and a reduce phase on 16 blocks (123 us per 20 MiB) that never overlaps a push. One launch per message is the gain; 64 blocks cut the reduce to 38 us. End to end, 32 / 64 / 128 blocks tie (887-891 ms, m5), so 64 stays. The floor of the current design is push + reduce ~= 790 us per 20 MiB; the trace's 208 ms per rank sits ~16 ms above 256 x 750 us.

## 3. What changed

* `kernels/hip/mxfp4_a8_gemm.hip`: `launch_a8` split into `a8_check` / `a8_quant` / `a8_spans` / `a8_gemm`; `launch_a8_pair` + `ik_matmul_mxfp4_pair_a8_mma` (one quant pre-pass, one span launch for both weights with their own wref and fold flag, then the two GEMMs, each exactly the single call's); `a8_wspan_kernel` takes an optional second weight; the direct pre-pass zeroes the fold flags (the staged path and the per-group knob keep their memsets); `a8_direct_run` takes the 64 x 128 tile (`2,2,2,4`) for N <= 64 (b / a: call level 92 -> 79 us, m7). Every direct tile gives the same bits on PF27's per-tile hashes; the 24-feature b / a shapes with the new tile are checked end to end (section 5).
* `kernels/hip/ik_mx_a8.h`: `mx_a8_quant_tiled_kernel` zeroes `nzero` flags (block 0).
* `crates/intarsia-backend-hip/src/quantmoe.rs`: `matmul_mxfp4_pair_a8_mma` registered for `Op::MatmulMxfp4Pair` with `matmul_mxfp4_a8_mma`'s predicate, exactness and priority, so a pair selects it exactly where each single call selects the single GEMM (M <= 8 keeps the pair GEMV at priority 10; verify / batched `exact_rows` calls keep the row kernels). No model change: the four `try_mxfp4_pair` sites now find a kernel at prefill M. `ShapeInfo::n1` (second output width of the pair ops) sizes its scratch.
* `crates/intarsia-backend-hip/src/comm.rs`: the wide fence path (`Wide`: its own scratch, flags and counters, `INTARSIA_HIP_AR_WIDE=BLOCKS[,SLOT_MIB]`, default 64 blocks and 20 MiB, `0` = the old path). All-reduces above the LL limit take it, a pure function of the message size. The LL path, mode 7 as the default, the fused all-reduce+norm and the all-gather are untouched. `p2p_bench` (benchmark only).
* `kernels/hip/allreduce.hip`: drain 9 (tests only, the fence path without its wait, the negative control); `ik_ar2_bench` (benchmark only: the push or the reduce phase alone).
* Tests: `tests/mxfp4_a8.rs` `a8_pair_is_bitwise_two_single_calls` (10 shapes x 2 out dtypes, the 27B pair shapes at 2048 / 1810 rows, staged and direct M, N1 > N0); `tests/comm.rs` `wide_path_is_bitwise_the_old_path_at_prefill_sizes`, `wide_path_survives_a_delayed_rank` (40 rounds x 16 mixed all-reduces, 8 elements to 22 MB, sizes off every slot multiple, one rank delayed up to 2 ms), `wide_path_without_the_wait_is_caught_by_the_stress_test`, `p2p_link_rate`.
* `docs/tuning.md`: `INTARSIA_HIP_AR_WIDE`.

## 4. Speed, localeval (W1)

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, max-batch 1, one server per arm change, order B D [texts, mixids] B B D D B (MTP on), then B D D B B D (MTP off). Table by `scripts/pf27b/abtab.py` (`raw/w1/abtab.txt`).

MTP on (the scoreboard lane):

| prompt | prefill base | prefill dev | ratio | decode base | decode dev | ratio | prefill runs base / dev | TTFT base / dev |
|---|---|---|---|---|---|---|---|---|
| 1k | 3738 | 3936 | **1.053** | 93.0 | 95.1 | 1.022 | 3758 3718 3107 3781 / 3613 3984 3936 | 0.265 / 0.251 s |
| 4k | 4029 | 4320 | **1.072** | 85.2 | 85.0 | 0.998 | 3997 4062 4070 3911 / 4367 4205 4320 | 1.014 / 0.945 s |
| 16k | 3758 | 3976 | **1.058** | 77.9 | 79.5 | 1.021 | 3775 3726 3740 3792 / 3976 3996 3967 | 4.430 / 4.187 s |

MTP off (decode check; the decode kernels and decode-sized collectives are unchanged):

| prompt | prefill base | prefill dev | ratio | decode base | decode dev | ratio |
|---|---|---|---|---|---|---|
| 1k | 3759 | 3937 | 1.047 | 55.7 | 55.6 | 0.999 |
| 4k | 4127 | 4286 | 1.038 | 54.9 | 54.9 | 1.000 |
| 16k | 3765 | 3998 | 1.062 | 52.0 | 52.0 | 1.000 |

The MTP-off 4k base runs read higher than the MTP-on ones (4094-4143 vs 3911-4070) for the same prefill path; the 4k prefill spread within one arm is ~4%, which is why the plain `generate` rounds are given too (dev 883-889 ms vs base 929-942 ms, x1.06).

## 5. Gates

| gate | result |
|---|---|
| prefill >= +5% at 4k, 3 interleaved rounds | **met**: x1.072 (MTP on, 4 base / 3 dev runs); MTP off x1.038 |
| no prefill length more than 1% slower | **met**: 1k x1.053, 16k x1.058 (MTP off x1.047 / x1.062) |
| decode not more than 1% slower | **met**: MTP off x0.999 / x1.000 / x1.000; MTP on x1.022 / x0.998 / x1.021 |
| greedy ids at tp=2, 18 prompts, spec off and MTP, equal main | **met**: 36 / 36 SAME (`scripts/dfh/ids-prompts`, 256 greedy ids, `generate --tp 2`, base vs dev binary; `raw/w2/ids-*`) |
| output bitwise identical to main | served texts: 18 / 18 md5 equal base vs dev (`raw/w1/texts-*`); mixids (max-batch 16, B prefilled beside A): dev A `915f1c0953` x3, B `dd42cb5b78` / `e74f590855` / `537fb01934`, PF27's main values (base: see section 7); single W4A8 GEMM hashes (`a8_mma_hashes`, 24 shapes x 2 out dtypes, none with N <= 64) 48 / 48 equal base vs dev binary, on the final binary (W2); the new N <= 64 tile is covered end to end (ids, texts, soak taps vs main) and by the pair test at N = 24; pair == two singles bitwise (`a8_pair`, default, per-group and staged runs); wide path == old path bitwise (`wide_path_is_bitwise_*`); a8 tests 7 / 7 under default, `PERGROUP=1`, `DIRECT=0` (W2) |
| `kernels test --backend hip --quick` | **met**: exit 0, 85 kernels / 462 cases (main: 84 / 456), 75 ADMITTED rows (main 74; the new one is `matmul_mxfp4_pair_a8_mma`, 6 cases, max_abs 0), the same 10 incomplete (no `--quick` cases) as main |
| Mac `cargo test --workspace --release` | **met**: 202 `test result: ok`, 0 failed |
| `plan show` 8B / 35B byte-identical | **met**: 1346 B / 3184 B, base vs the final dev CLI (m9; m6 on an earlier dev build agreed too; the W2 8B row failed to start on a container-name clash in the harness, fixed) |
| all-reduce: delayed-rank stress, negative control | **met**: whole `tests/comm.rs` (13 passed) and `tests/ar_norm.rs` (4) under the default env, `INTARSIA_HIP_AR_DRAIN=1` and `INTARSIA_HIP_AR_WIDE=0`: the LL delayed-rank stress and its no-tag-check control (1380 wrong), the mode 8 control, the new wide-path stress (0 wrong in 40 rounds) and its no-wait control (246 wrong); wide == old path bitwise |
| all-reduce: 120-process tapped soak | **met**: 120 fresh tp2 processes of the dev CLI, prompt `4k-0` (both 2048-row chunks take the wide 20 MiB launches), MTP k2, 16 tokens: 120 / 120 rc 0, 1 distinct id line, 120 / 120 taps byte-identical to each other and to main's (3 base processes, `c006426589a1...`; `raw/w2/soak-taps.md5`) |
| merged tree (origin/main `f2bb3410f`, after #229 / #231) | **met**: ids `short-0` / `4k-0` / `16k-0` x off / MTP 6 / 6 SAME vs a fresh main build, `plan show` 8B / 35B byte-identical, 4k prefill 933-935 -> 882-886 ms (m10). #229's side-stream projections do not engage here (`side_fork`: tp > 1 and rows > 16 return `None`) |
| CI | see the PR |

## 6. Item 2, the overlap: what `INTARSIA_TP_OVERLAP` is and why it stays off

Lane TPO (#202, `docs/results/2026-10-27-tpo-hip-27b-prefill-next.md`) splits a single-sequence prefill pass of >= 3072 rows into K row pieces and queues each piece's row-parallel all-reduce on stream 1 behind an event while stream 0 computes the next piece. Bitwise (per-row ops). Opt-in because: it only pays with a second hardware queue (`GPU_MAX_HW_QUEUES=4`, process-wide; the launchers pin 1, and `docs/research/amd-r9700-lessons.md` records a permanent ~20% decode loss from a copy stream on 4 queues) and with `--prefill-chunk 4096` (pieces of 2048 rows, else the GEMM drops to a smaller tile), and its 6 generate rounds gave x1.0445, under the +5% bar. The scoreboard's 3256 / 3755 / 3545 is that arm against its own base 3280 / 3527 / 3418 (x0.993 / x1.065 / x1.037), not worse than off. On this branch (m6, 2 rounds): 888 ms with `INTARSIA_TP_OVERLAP=2 GPU_MAX_HW_QUEUES=4 --prefill-chunk 4096` vs 887-889 ms without, so it adds nothing once the all-reduce is one wide launch.

What is left on the all-reduce: 208 ms per rank against ~192 ms of pure wire time (256 x ~750 us at 28 GB/s per direction). Only hiding it helps now, worth up to ~20% of the prefill. Without a second hardware queue the candidate is pushing the partial to the peer from the producing GEMM's epilogue (down / out projections), with the handshake and reduce in a short kernel after it: a change to the shared W4A8 GEMM and to the model's projection path, and a new data-then-flag race across two kernels that the comm stress tests cannot see. Not attempted here.

## 7. Observations

* mixids on the **base** binary: in its first pair (B = `4k-0`) stream A's 200-chunk text was `e70fbb862c` instead of `915f1c0953` (the other two pairs and all three dev pairs gave `915f1c0953`). One occurrence, main's binary, max-batch 16 with a prefill beside a decode; not investigated here.
* The 128 small all-reduces per 4k prefill are the LL path (`ar2ll_kernel`, ~9 us each).

## 8. Windows and micro jobs (fleetq, tatooine, each took radiance-rank0 and radiance-rank1 down)

| job | kind | what | radiance answered after |
|---|---|---|---|
| #6650 | micro m1 | P2P link rate, a8 pair bitwise | 10 s |
| #6652 | micro m2 | wide-path comm gates | 10 s |
| #6653 | micro m3 | 4k generate base / dev / dev `WIDE=0` | 10 s |
| #6654 | micro m4 | dev trace + 4 generate rounds | 10 s |
| #6656 | micro m5 | pair, hashes, `AR_WIDE` 32 / 64 / 128 sweep | 10 s |
| #6658 | micro m6 | plan show, negative-control count, max-seq 20480, TP_OVERLAP stack | 35 s |
| #6662 | micro m7 | small-N a8 tile timing | 10 s |
| #6664 | micro m8 | a8 tests + generate rounds | 10 s |
| #6667 | **W1** | localeval A/B, texts, mixids (22:19:46-22:42:06) | 15 s |
| #6675 | **W2** | gates: a8 / comm / ar_norm tests, `kernels test --quick`, ids 36, 120-process soak, traces (22:43:09-23:04:17) | 26 s |
| #6679 | micro m9 | plan show on the final CLI | 25 s |
| #6684 | micro m10 | merged tree vs main `f2bb3410f`: ids x6, plan show, 4k generate | 40 s |

(The first W1 start, #6665, exited at once with a permission error in the driver before stopping radiance.) CPU jobs: builds #6648, #6649, #6655, #6661, #6663.

## 9. Next

* Hide the all-reduce (the last ~20%): a GEMM-epilogue push to the peer for the row-parallel projections, needing its own race gates.
* GDN (115 ms per rank, token-sequential f32 scan): a WMMA chunked kernel changes numerics, opt-in only.
* The GEMM is ~60% of the fp8 peak; the remaining launches cost ~30 ms of gaps per 4k prefill.
