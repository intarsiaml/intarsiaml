# MGD: Metal 35B decode KL is not a decode-path defect; GDN delta-rule prefill 2.7x faster, bitwise (2026-10-27)

Lane MGD (`lane-mgd/metal-gdn-kl`), 2026-10-27 03:40 to 04:40 EDT, Apple M5 Max (128 GB). Qwen3.6-35B-A3B NVFP4. Base `origin/main` 2859c1a89 (PR #193, NXP: NAX fp8 GEMM and MoE prefill). Every GPU run was a fleetq slot; timed runs were `--quiet gpu,cpu` with `pgrep -x rustc/cargo` logged before and after (0 in all arms). Raw files, scripts and logs: [raw/](2026-10-27-mgd-metal-gdn-kl/raw/).

## Summary

| question | answer |
|---|---|
| Part A: why is Metal decode KL about 2x CUDA's? | **It is not a decode-path defect.** Decode and prefill sit at the same distance from the f64 CPU oracle (paired difference -4.7e-6 over 5494 positions, 95% CI +-7e-4 on 9.5e-3). The 16-position gate window is a heavy-tail sample that moves 1.7x with prefill accumulation order alone. Nothing to fix; no engine change for Part A. |
| Part B: GDN delta rule prefill | New `gdn_delta_rule_bf16_pre` (T >= 64): **bitwise** the old kernel, 6.18 to 2.28 ms per 2048-token call (2.7x). 4k prefill **+21%**, 16k **+18%** (localeval, 3 interleaved rounds), decode 0.994x / 1.000x. |

## Part A: accuracy

Method: no per-op bisection was needed. A path-level test bounds every decode-only op at once, because decode and prefill use the same weights, and the same `gdn_delta_rule` kernel (T = 1 against T = 2048, same f32 state), and differ in GEMV against GEMM, split-KV against flash attention, and the MoE decode against grouped kernels. `intarsia accuracy --mode both` with `--decode-positions 100000` (clamped to n-1) runs teacher-forced decode over every position instead of the gate's last 16 (`raw/partA/`).

1. **Full-window decode against the vLLM short golden** (35b-short, 10 prompts): decode KL21 5.42e-3 over 1431 positions, prefill 6.45e-3 over 1441. Decode is not worse than prefill. (CUDA's gate-window figures: prefill 6.65e-3, decode 7.34e-3.)
2. **Against the independent f64 CPU oracle** (`golden/oracle/...cpu-oracle.safetensors`, oracle set: the 10 short prompts and `long_docs_4k`, chunk 256, 5494 positions; the oracle is deterministic, so vLLM's sampling noise drops out): decode 9.451e-3, prefill 9.456e-3 on the same positions; paired difference decode - prefill -4.7e-6, block-bootstrap (blocks of 64) 95% CI [-7.0e-4, +6.7e-4] (`raw/partA/an3-oracleset-paired.txt`). The set includes a 4064-token sequence, so a GDN state drift over 4k decode steps would show. Any decode-only error of the candidates (GDN recurrent state precision, softmax accumulation, nvfp4 GEMV dequant rounding, MoE summation order, rmsnorm precision) is bounded under 7% of the distance both paths have to the oracle (about 9.5e-3, which is the bf16-activation floor the other backends share: fx1 measured CUDA 9.33e-3, HIP 9.86e-3, Metal 9.88e-3 prefill on the scored rows).
3. **The gate window is a heavy-tail sample.** The window is the last 16 positions of each of 10 prompts (160 positions). The top 10 positions carry 72% of its mean (`raw/partA/an1-window.txt`). Same weights, same decode kernels, only the prefill numerics changed:

   | arm | window prefill vs oracle | window decode vs oracle | window decode vs vLLM golden (KL21 all rows / scored) | full-window decode vs vLLM |
   |---|---:|---:|---:|---:|
   | NAX prefill off (`INTARSIA_METAL_FP8_NAX=0 INTARSIA_METAL_MOE_NAX=0`, the M35P build) | 1.02e-2 | 8.80e-3 | 1.164e-2 / 1.169e-2 | 5.42e-3 |
   | NAX prefill on (current main) | 6.03e-3 | 8.80e-3 | 8.20e-3 / 6.95e-3 | 5.42e-3 |

   Decode is bit-identical between the arms (max abs per-position KL difference 0.0); only the prefill that writes the KV and GDN state before the window moved, and the window figure moved with it from 1.16e-2 to 8.2e-3. Equivalent builds spread 1.7x on this window, which is the size of the "1.6 to 2.0x CUDA" gap M35P reported. Random 160-position subsets of the full decode have a mean of 5.1e-3 with a 95% range of [2.6e-3, 1.0e-2].
4. **Status of the "fails 3 of 10 prompts" check, today** (`baseline-gate.sh`, `raw/partA/short-vs-cuda-baseline.log`): against the CUDA baseline, 1 of 10 prompts fails per prompt (`chat_multiturn`, 2.28e-2 against a 1.24e-2 limit), driven by one near-tie position (`chat_multiturn` position 55, KL21 0.35). The aggregate window decode is 8.19e-3 against CUDA's 7.34e-3 (1.12x). Metal against its own recorded baseline and envelope: PASS on 35b-short. The 4k Metal baseline still FAILS decode (known from NXP: recorded before the NAX change, not re-recorded).

Limit: no CUDA decode-against-oracle number exists (the committed oracle self-goldens carry prefill rows only), so "Metal decode is not worse than CUDA decode" is shown through the shared reference, not head to head. HIP was not run (no HIP box in this lane); the argument is the same, because its decode window reads the same way (1.172e-2) and its decode kernels are separate code.

No fix, so no MMLU run for Part A.

## Part B: GDN delta rule prefill

### Profile and roofline (35B: Hk 16, Hv 32, dk = dv = 128, 30 GDN layers; 2048-token prefill chunk)

| | base `gdn_delta_rule_bf16` | `gdn_delta_rule_bf16_pre` |
|---|---:|---:|
| ms per 2048-token call (kernel mode, 60 calls at 4k) | 6.18 | 2.28 (+ 0.08 prep) |
| GPU time at 4k (of 1112 / 893 ms) | 371 ms, 33.4% | 137 ms + 5 ms, 15.9% |
| GPU time at 16k (of 5971 / 5077 ms) | 1458 ms, 24.4% | 536 ms + 20 ms, 11.0% |
| achieved (7 flop per state element: 3.67 MFLOP per token per layer, 7.5 GFLOP per call) | 1.2 TFLOPS | 3.3 TFLOPS |

The recurrence is serial over tokens per (head, value row) and carries 2 simd_sum reductions per row-step, so it is bound by reduction latency and issue rate, not by memory (about 100 MB per call, 44 GB/s). Both are far below the fp32 vector peak, which was not measured here. The old kernel also recomputed, in each of the 128 value rows of a head, the per-token work that does not depend on the state (q and k l2 norms, two simd_sums, rsqrts, sigmoid, softplus, exp).

### Kernel

`gdn_delta_rule_bf16_pre` (`kernels/metal/gdn.metal`, `crates/intarsia-backend-metal/src/gdn.rs`; registered priority 5, `.when(T >= 64, dv % 16 == 0)`, plan scratch 34 MB at T = 2048, which fits inside the plan's existing 84 MB; `INTARSIA_METAL_GDN_PRE=0` hides it):

1. a prep pass, one simdgroup per (token, key head), writes the normalised q and k (f32, `bf2f(q) * qr`, `bf2f(k) * kr`) and per-(token, value head) beta and decay to scratch, computed with the original's own expressions and lane layout;
2. the recurrence keeps only the state chain. A threadgroup of 4 simdgroups covers 16 value rows (4 per simdgroup, independent chains whose simd_sums overlap), stages a tile of 16 tokens (q, k, beta, decay, v) in threadgroup memory while the next tile is fetched into registers, and so reads no device memory in the serial loop.

Every float operation of the recurrence is the original's, in the original's order. Steps tried (kernel ms at 2048 tokens): prep pass only 4.82; plus threadgroup tile 3.47; 2 rows per simdgroup 2.78; 4 rows 2.28; 8 rows 2.22 (kept 4); tile 8 instead of 16 2.26 (no change). At 4 rows the kernel is issue-bound; the next step would be a non-bitwise chunked/WY form, not tried (see below).

Decode (T = 1), MTP verify (T <= k+1), and `gdn_delta_rule_bf16_rows` use the old kernel unchanged.

### Gates

| gate | result |
|---|---|
| kernel admission (`kernels test --backend metal --model 35B --op gdn_delta_rule`) | `gdn_delta_rule_bf16_pre` 8 cases, 0 failed, max_abs 1.2e-4, rel_l2 4.7e-5, ADMITTED (old kernel identical numbers) |
| planted mutants (`raw/mutants/`) | **7 of 7 caught**: m1 no q scale, m2 beta/decay swapped, m3 partial tile run full, m4 v rows swapped, m5 wrong key-head map, m6 last row's state not written: REJECTED by admission; m7 0.01% perturbation of one staged gate: admitted (rel_l2 1.7e-4) but caught by `tests/gdn.rs::delta_rule_prep_form_is_bitwise_the_original` |
| bitwise against the old kernel | new test: 7 shapes, every dk, one call and split calls, 2-byte operand offset: outputs and f32 state equal. All `tests/gdn.rs`, `rows_exact`, `row_tables` pass |
| KL vs vLLM goldens (`raw/kl/`) | bit-identical to base: short 6.448e-3 / 8.200e-3 (prefill / decode KL21), long 4.621e-2 (scored 1.461e-2) / 1.568e-2, 4k 9.978e-2 (scored 1.209e-2) / 8.958e-3 for GDN_PRE on and off; the top-64 ids and log-probs written by `--write-self-golden` for prefill and decode are byte-identical (`cmpsg.py`: 0 of 108 tensors differ) |
| MMLU | not run: ids are bitwise unchanged, so the MMLU 0.842 of NXP stands |
| greedy ids (`raw/ids/`) | 256 ids on short-0, 4k-0, 16k-0: cand equals base for `--spec off`, `--spec mtp --spec-k 2` and `--prefill-chunk 512`; MTP equals spec-off. (`--spec ngram` is unsupported on this hybrid model in both builds, not a regression.) |
| batched rows equal solo (`raw/g3/`) | `batch_metal_real` (2 tests, single thread; the first parallel run failed on memory, "only 0 MiB free", with two 35B copies loaded), long prompts 2.6k/4k/16k equal local engine, `batch_recurrent` 14 passed, dense 8B `batch_metal` passed |
| 4k prefill >= 10%, 3 interleaved rounds, localeval | **+21.3%**: median 3367 to 4084 tok/s (base 3652/3301/3367, cand 3866/4084/4093); 16k +18.2% (2427 to 2868); 1k +21.5% |
| decode >= 0.99x | 0.994x (1k, 4k), 1.000x (16k): 123.2 / 119.2 / 108.8 base against 122.5 / 118.5 / 108.8 cand tok/s. The decode kernel is untouched |
| `plan show` other Metal models | Qwen3-8B, Llama-3.1-8B, Qwen2.5-7B identical. The two GDN models (35B and Qwen3.8-27B MXFP4) gain exactly one registry row (`gdn_delta_rule_bf16_pre`); plan scratch unchanged at 84 MB for the 35B. `qwen38-27b` does not load in either build |

Timing hygiene: all speed arms ran in one `--quiet gpu,cpu` slot (fleetq #6065), `rustc+cargo` 0+0 before and after every arm. The per-arm CPU samples (`raw/speed/*-load.log`) show one `rustc` at 78% briefly in round 1's cand arm (another lane) and background tailscale and spotlight bursts; cand beats every base round regardless.

### Not done

- No chunked/WY or NAX form. The bitwise form already clears the speed gate, and the remaining GDN share is 15.9% at 4k, 11% at 16k, so a further 2x would add about 8% at 4k and 5% at 16k at the price of leaving bitwise ids and chunk-size invariance. A reduce-scatter replacement of the simd_sum chains was considered and not tried (it relies on the hardware reduction tree, which can only be checked empirically).
- HIP and CUDA were not touched or run.
- Metal's recorded 4k baseline (decode) still fails from before this lane; not re-recorded.

### Reproduce

`raw/scripts/`: `e1.sh`..`e4.sh` and `an1.py`..`an3.py`/`sg.py` (Part A), `prof.sh` + `perop.py` (profile), `mutants.py`/`mutrun.sh`/`mutadm.sh`, `gates.sh`, `kl.sh` + `cmpsg.py`, `ids.sh`, `plans.sh`, `speed-ab.sh`. Slots: #6000 to #6065 (see each log).
