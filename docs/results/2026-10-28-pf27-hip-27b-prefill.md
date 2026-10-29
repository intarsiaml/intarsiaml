# Lane PF27: Qwen3.8-27B MXFP4 prefill on the R9700s (HIP, tp=2): direct W4A8 GEMM (2026-10-28)

Branch `lane-pf27/prefill`, built on origin/main `318d508fb` (PR #224). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`, W4A8 (`mxfp4_act = fp8`, the recipe default), MTP k2 default, `prefill_chunk = 2048`, localeval against `intarsia serve`. Raw data: `docs/results/2026-10-28-pf27-hip-27b-prefill/raw/`, scripts `scripts/pf27/`.

## Headline

| | |
|---|---|
| Asked | Close the R9700 27B prefill gap (last of 3: 0.68 / 0.64 / 0.61x the vLLM-fork). Target >= 0.80x the vLLM-fork at 4k (~4800 tok/s); any merged gain >= +5% at 4k is worth keeping. |
| Attribution (W1, main) | Per rank, 4k prompt (3858 tokens): W4A8 GEMM **48%** (497 ms, 200 TFLOP/s = 55% of the 366 fp8 peak), all-reduce **23%** (239 ms), GDN 11% (113 ms), activation quant 4%, norms 3%, attention 3%, idle 5%. |
| Lever | The W4A8 prefill GEMM (`matmul_mxfp4_a8_mma`). New `a8_direct_kernel`: the activation pre-pass writes the fp8 tokens in WMMA fragment order, each wave loads its token fragments straight from global memory (scalar base + lane offset, one stage ahead, LDS-only stage barrier), only the weights go through LDS. Tile 64 features x 512 tokens (M >= 768), else 64 x 256. **Bitwise equal output** (same WMMA operands in the same order, same epilogue). |
| GEMM | Call time (kernel + pre-passes) over the five 27B tp2 shapes **-16.3% at M = 2048** (4895 -> 4098 us), -17.6% at 1024, -15.3% at 4096; 164 -> 193..200 TFLOP/s per shape. |
| Prefill, localeval (W2) | 3 interleaved rounds, MTP on: **1k / 4k / 16k x1.063 / x1.076 / x1.067** (3483 / 3789 / 3543 -> 3703 / 4076 / 3782 tok/s). MTP off, 2 rounds: x1.024 / x1.069 / x1.059. Decode MTP off x1.010 / x1.004 / x1.004. |
| Gap | vs the scoreboard's best same-quant numbers (Radiance 1.12.0 5214 / 5994 / 5869, R3): 0.68 / 0.64 / 0.61x -> **0.71 / 0.68 / 0.64x**; vs the vLLM-fork (4k 5399, 16k 5305, R123 / R126 **old**): 4k 0.71 -> 0.75x, 16k 0.67 -> 0.71x. Target 0.80x at 4k (~4800) **not reached**; still rank 3/3. |
| Gates | All met (table below). |
| Windows | 3 of 3 (W1 attribution + kernel gates, W2 A/B, W3 ids gate + branch trace); 5 kernel micro-benchmark jobs under 3 min (not windows). Radiance restored and `/v1/models` confirmed on :8080 before each release: 30 s, 35 s, 10 s (W1 / W2 / W3; micro jobs 10 s each). :8000 does not answer (HTTP 000) even with Radiance up; the live service is on :8080. |

## 1. Attribution (W1, main `318d508fb`)

`generate` of `scripts/cmtp/ids-prompts/4k-0.txt` (3858 tokens), `--max-tokens 1 --spec off`, tp=2, under `rocprofv3 --kernel-trace` (`raw/w1/p-pf27-base-cli.csv.gz`, `scripts/pf27/trace_tp2.py ... --proj --shapes --names`). Profiler on: 1030 ms span (plain run 1009-1015 ms).

| bucket | rank 0 ms | % span | rank 1 ms | % span | launches | note |
|---|---|---|---|---|---|---|
| W4A8 GEMM (`a8_gemm_kernel`, rows kernels) | 497.3 | 48.3 | 490.9 | 47.5 | 2304 | 470 ms in the 128 x 256 tile; 11.5 ms the 16-row peel's fused rows kernel |
| all-reduce (`ar2_kernel<0>`) | 239.0 | 23.2 | 247.1 | 23.9 | 1409 | includes the wait for the peer (one-shot spin), so an upper bound of wire time |
| GDN (delta rule v3, conv) | 112.6 | 10.9 | 111.1 | 10.8 | 480 | `gdn_delta_rows_v3_kernel` 104 ms: token-sequential f32 scan, no WMMA |
| activation quant + weight span | 39.1 | 3.8 | 38.7 | 3.8 | 1984 | 992 pre-passes: inputs shared by 2-4 GEMMs are quantised once per GEMM |
| norms | 32.3 | 3.1 | 32.4 | 3.1 | 579 | |
| attention (flash prefill) | 30.5 | 3.0 | 30.2 | 2.9 | 64 | |
| elementwise | 20.4 | 2.0 | 20.4 | 2.0 | 244 | silu-glu 18.7 ms |
| runtime (fills, copies) + lm head | 5.3 | 0.5 | 5.5 | 0.5 | | |
| idle / gaps | 53.0 | 5.1 | 56.3 | 5.5 | | |

GEMM achieved rate per projection (rank 0, kernel time, M = 2048 / 1794 chunks):

| projection | ms | TFLOP/s | % of 366 (repo's fp8 peak) | % of 412 (Radiance's figure) |
|---|---|---|---|---|
| gate + up (N 2 x 8704) | 220.0 | 200.1 | 55 | 49 |
| down + GDN qkv + out + o_proj (N 5120) | 200.8 | 196.7 | 54 | 48 |
| GDN z (N 3072) | 30.0 | 194.4 | 53 | 47 |
| attn q + gate (N 6144) | 19.7 | 197.4 | 54 | 48 |

Top two buckets and their headroom: the GEMM (48%; 45% below the fp8 peak on paper; lane LDF measured the card at its 300 W cap in this kernel) and the all-reduce (23%; owned by lane ARS, kernels not touched here). GDN is third (11%): a WMMA chunked rewrite (Radiance's design) would change numerics, so it could only ship opt-in.

After (W3, branch `24b8915b3`, same prompt and command, `raw/w3/p-pf27-dev-cli.csv.gz`): span 966 ms (rank 0; was 1030), plain run 967 ms.

| bucket | rank 0 ms before | after | rank 1 ms before | after |
|---|---|---|---|---|
| W4A8 GEMM | 497.3 | **437.5** | 490.9 | **421.6** |
| all-reduce | 239.0 | 238.2 | 247.1 | 255.6 |
| GDN | 112.6 | 115.7 | 111.1 | 114.1 |
| activation quant + weight span | 39.1 | 41.5 | 38.7 | 41.3 |
| norms / attention / elementwise | 83.2 | 84.6 | 83.0 | 83.9 |
| idle | 53.0 | 42.8 | 56.3 | 47.9 |

The main GEMM kernel: 470 ms (128 x 256 staged, ~198 TFLOP/s) -> 424 ms (`a8_direct_kernel<2,8,2,4>`, ~220 TFLOP/s, 60% of 366) on rank 0 for the same work. The tiled pre-pass costs 37.0 ms vs 35.0 for the row-major one.

## 2. The lever: why the GEMM, and what limited it

Prior art: lane GLR (#206) found the staged kernel's time collapses without its LDS fragment reads (-43%); lane LDF (draft #211) found the LDS read rate itself is not the limit and named "activation fragments straight from L0" as the open question. Radiance (`/data/src/radiance/libr4d/r4d_gemm_mxfp4a8_tiled.hip`, Apache-2.0, read for technique only, no code copied) feeds the activations from global memory in a pre-tiled fragment order and only the weights through LDS.

Steps, each a kernel micro-benchmark job (`timing_a8_tp2`, call level, sum of the five shapes; hashes on 48 shape/dtype cases each time):

| step | sum M = 2048 (us) | vs main | raw |
|---|---|---|---|
| main (staged kernel, 128 x 256) | 4895 | | m4 |
| direct kernel, first build (tokens from global, 128 x 256 tile) | 4764 | -2.7% | m2 |
| ... end to end: 4k prefill 1011 vs 1009-1015 ms (no change) | | | w1 `T` |
| same, scalar token addressing + zero pad group (no per-load 64-bit address VALU, no tail selects) | 4299 | -12.2% | m4 |
| + 64 x 512 tile (16 waves) | **4098** | **-16.3%** | m4, m5 |

Timing-only diagnostic builds of the first direct kernel (`raw/isa/mxfp4_a8_gemm.diag.hip`, `raw/m2`), sum at M = 2048: no token loads 3851 (-19%), no weight LDS fragment reads 3856 (-19%), no weight decode / LDS store 4260 (-11%), no weight global loads 4147 (-13%), no WMMA 2998 (-37%), no barrier 4672 (-2%). Unlike the staged kernel (GLR: no WMMA -1%), the direct kernel's WMMAs are on the critical path; the steady loop had 204 VALU per 64 WMMA (`scripts/pf27/loopmix.py`), mostly the per-lane 64-bit token addresses and `kg < KG` selects, against 88 in the staged kernel. Making the wave index uniform (`readfirstlane`), the token address a scalar base plus a 32-bit lane offset, and the K tail a zero group written by the pre-pass cut it to **80 VALU per 64 WMMA**, which is where the -12% came from. Interleaving the two k-halves across accumulators (no back-to-back dependent WMMAs) changed nothing (`raw/m3`).

Tile sweep (`raw/m5`, sum over the five shapes per M; us): the 64 x 512 tile (`2,8,2,4`) wins from M = 1000 (1984 vs 2078 for 64 x 256) and ties at 512; 64 x 256 (`2,4,2,4`) wins below (M = 130: 615 vs 780 main). Tiles with 4 x 4 fragments per wave spill (256 VGPR) and lose. Resources: both shipped tiles 181 VGPR, no scratch, 8 waves/SIMD, 10 KB LDS (main's tile: 175 VGPR, 60 KB).

M = 2048 per shape, call level (`raw/m4/t-base2.txt`, `t-dev-2824.txt`):

| shape (N x K) | main us | TFLOP/s | branch us | TFLOP/s | change |
|---|---|---|---|---|---|
| gate+up 17408 x 5120 | 2220.7 | 164.4 | 1887.2 | 193.4 | -15.0% |
| down 5120 x 8704 | 1102.8 | 165.5 | 912.0 | 200.1 | -17.3% |
| q+gate 6144 x 5120 | 780.9 | 165.0 | 644.1 | 200.0 | -17.5% |
| z 3072 x 5120 | 397.0 | 162.3 | 330.6 | 194.9 | -16.7% |
| o 5120 x 3072 | 393.4 | 163.8 | 323.6 | 199.1 | -17.7% |

What else was tried in W1 (`raw/w1`): `--prefill-chunk 4096` with the first direct kernel 1057-1063 ms vs 1011-1015 at 2048 (slower).

## 3. What changed

* `kernels/hip/mxfp4_a8_gemm.hip`: `a8_direct_kernel` (FOLD and per-group paths, as the staged kernel), `a8_lds_barrier`, `a8_direct_run` (tile table), `a8_use_direct` (M > 64, not when `INTARSIA_HIP_MXFP4_A8_TILE` or `_PD=1` is set), scratch over round16(M) x round64(K).
* `kernels/hip/ik_mx_a8.h`: `mx_a8_quant_tiled_kernel`, the same quantisation helpers (same xs and e4m3 bytes) written in fragment order, pad rows and the K % 64 == 32 pad group zeroed.
* `crates/intarsia-backend-hip/src/quantmoe.rs`: `matmul_mxfp4_a8_mma` scratch = `mx_a8_scratch(round16(m), round64(k)) + round16(n) + 64` (27B only; `plan show` of the 8B and 35B unchanged).
* Knobs (`docs/tuning.md`, `ENV_KNOBS`): `INTARSIA_HIP_MXFP4_A8_DIRECT=0` restores the staged kernel, `INTARSIA_HIP_MXFP4_A8_DTILE=WN,WM,TN,TM` forces a direct tile.

## 4. Speed, localeval (W2)

`localeval speed --sizes 1024,4096,16384 --gen 128 --reps 3`, max-batch 1, one server per arm change; order A B B A A B (MTP on, 3 per arm) then A B B A (MTP off, 2 per arm). Base = `pf27-base-cli` (main `318d508fb`), branch = `pf27-dev-cli` (`raw/w2/steps.log`, table by `scripts/pf27/abtab.py`).

MTP on (the scoreboard lane):

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio | prefill runs base / branch |
|---|---|---|---|---|---|---|---|
| 1k | 3483 | 3703 | **1.063** | 92.1 | 93.8 | 1.019 | 3105 3548 3483 / 3729 3703 3263 |
| 4k | 3789 | 4076 | **1.076** | 80.4 | 86.4 | 1.075 | 3789 3861 3678 / 4127 4076 3993 |
| 16k | 3543 | 3782 | **1.067** | 78.6 | 76.2 | 0.969 | 3543 3554 3531 / 3782 3761 3803 |

MTP off (decode check: the decode kernels are unchanged, M <= 64 keeps the old paths):

| prompt | prefill base | prefill branch | ratio | decode base | decode branch | ratio |
|---|---|---|---|---|---|---|
| 1k | 3546 | 3632 | 1.024 | 55.2 | 55.7 | 1.010 |
| 4k | 3861 | 4128 | 1.069 | 54.6 | 54.8 | 1.004 |
| 16k | 3558 | 3770 | 1.059 | 52.0 | 52.2 | 1.004 |

TTFT at 4k (localeval `ttft_s`, median of runs): MTP on 1.078 -> **1.002 s** (runs 1.078 1.058 1.117 / 0.990 1.002 1.029), MTP off 1.058 -> 0.990 s. Radiance native: 0.678 s.

MTP-on decode varies 5-10% run to run (as in GLR / LDF); the 16k x0.969 and 4k x1.075 are that spread, the MTP-off decode is flat.

Base and branch are both built on `318d508fb`, which predates #225 (all-reduce mode 7 default; lane ARS measured the 27B unchanged by it). The branch is merged with main after #225 for the PR; the GEMM change and the all-reduce mode touch disjoint kernels, and no window was left to re-measure the merged tree.

Competitors (scoreboard row "R9700 27B MXFP4 W4A8, TP2"): Radiance 1.12.0 5214 / 5994 / 5869 (R3, 10-25) -> 0.71 / 0.68 / 0.64x (was 0.68 / 0.64 / 0.61x); vLLM-fork TP2 4k 5399, 16k 5305 (R123, R126 **old**) -> 0.75 / 0.71x. The lane brief called the 5214 / 5994 / 5869 figures the vLLM-fork's; the scoreboard attributes them to Radiance. Either way the 4k target (0.80x of 5994 = 4795) is not reached.

## 5. Gates

| gate | result |
|---|---|
| prefill 4k >= +5%, 3 interleaved rounds | **met**: x1.076 (MTP on, 3 vs 3), x1.069 (MTP off, 2 vs 2) |
| no prefill length slower than 1% | **met**: 1k x1.063, 16k x1.067 (MTP off x1.024, x1.059) |
| decode not slower than 1% | **met**: 1k MTP on x1.019, MTP off x1.010 / x1.004 / x1.004 |
| greedy ids at tp=2 equal main, 18-prompt ids set (`scripts/dfh/ids-prompts`, as det / vrc) | **met**: 36 / 36 SAME (18 prompts x spec off and MTP default, 256 greedy ids, `generate --tp 2 --chat --no-think --show-ids`, base vs branch CLI) (`raw/w3/ids-*`) |
| also: 18 gate prompts x 256 tokens through `serve` (`scripts/cmtp/ids-prompts`, texts.py) | **met**: md5 identical base vs branch (`raw/w2/texts-*`) |
| also: batched == solo (`mixids.py`, max-batch 16, a 48-token stream B prefilled beside A) | **met**: A `915f1c0953`, B `dd42cb5b78` / `e74f590855` / `537fb01934` for base and branch, the values GLR / LDF recorded for main |
| numerics | unchanged by construction; GEMM hashes on 48 shape/dtype cases identical to main's binary for the default, every direct tile, `DIRECT=0`, and the per-group path (`raw/w1/hash-*`, `raw/w2/gates-15/hash-*`, `raw/m1`, `raw/m4`). No KL / MMLU run needed (bitwise) |
| a8 correctness tests (M- and position-independence, reference, wide-span group path, flipped code) | **met**: 6 / 6 `a8_` tests pass on default, `DIRECT=0`, `PERGROUP=1`; `a8_mma_rows_do_not_depend_on_m` also checks staged (M <= 64) == direct (M > 64) rows in one process |
| `kernels test --backend hip --quick` | **met**: exit 0, 74 admitted rows, 84 kernels, 456 cases (10 kernels with no quick case, as on main) (`raw/w2/gates-15/adm.log`) |
| Mac `cargo test --workspace --release` | **met**: 202 `test result: ok`, 0 failed, exit 0 (`--no-fail-fast`) |
| `plan show` unchanged for the other models | **met**: 8B 1346 B and 35B 3184 B byte-identical, base vs branch CLI (`raw/w2/gates-15/plan-*`) |
| CI | see the PR |

## 6. Windows and micro jobs (fleetq, tatooine, all took radiance-rank0 and radiance-rank1 down)

| job | kind | what | radiance answered after |
|---|---|---|---|
| #6598 | **W1** | attribution traces (base, first direct build), 4k generate timing, a8 tests, hashes, tile timing, kernel traces, admission, plan show | 30 s |
| m1 | micro, 55 s | 128 x 512 / 256 x 256 / 64 x 512 tiles (features x tokens): timing + hashes | 10 s |
| m2 | micro | diagnostic builds (timing only) | 10 s |
| m3 | micro | interleaved WMMA order: timing + hashes | 10 s |
| m4 | micro | scalar token addressing: timing + hashes | 10 s |
| m5 | micro | tile table at M = 130 .. 2048 | 10 s |
| #6625 | **W2** | localeval A/B, texts, mixids, hashes, a8 tests, admission, plan show | 35 s |
| #6631 | **W3** | ids gate, branch attribution trace | 10 s |

CPU jobs (builds, ISA dumps): #6588 and the build1..build7 jobs in `pf27-out` (not committed).

## 7. Leftovers and next

* Still 0.68x the vLLM-fork at 4k. The GEMM is now ~195-200 TFLOP/s call level (53-55% of 366); the remaining big buckets are the all-reduce (23%, lane ARS) and GDN (11%).
* GDN: Radiance's chunk-64 WMMA pair (`r4d_gdn_kkt_solve_k128_c64_bf16.hip`, `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip`) keeps the state in WMMA accumulators; ours is a token-sequential f32 scan (104 ms per rank at 4k). A WMMA chunked kernel changes the summation order (and rounds the inverse), so it could only ship opt-in behind a KL gate.
* Quant pre-passes: 992 per 4k prefill, one per GEMM; GEMMs sharing an input (GDN qkv / z / a / b, attention q / k / v, gate / up) quantise it 2-4 times, and the weight-span pre-pass reruns every call on constant weights. Deduplicating both is bitwise and worth ~2-3%.
* The GEMM's remaining VALU (80 per 64 WMMA): weight-chunk addresses and selects, the fold-table load, the decode. A weight layout pre-shuffled at load time would remove most of it but changes the 27B load path.
