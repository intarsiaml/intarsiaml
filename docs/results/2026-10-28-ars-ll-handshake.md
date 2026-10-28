ars: sequence-tagged (LL) all-reduce handshake, HIP tp=2, 2026-10-28. Lane `lane-ars/ll-handshake`, base `origin/main` after #221. tatooine, 2x R9700 (gfx1201), `serve`/`generate --tp 2 --devices 0,1`, 27B `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and 35B `Qwen3.6-35B-A3B-NVFP4`. Raw: `docs/results/2026-10-28-det-tp2-nondeterminism/raw/ars-w2` (v1 kernels), `ars-w3` (v1 A/B), `ars-w4` (v2 kernels, the numbers below).

Status: DRAFT PR, not the default. `INTARSIA_HIP_AR_DRAIN=1` (fence, #221) stays the default; `=7` is the opt-in. Mode 7 is safe (0 divergent, bitwise equal to mode 1) and recovers all of mode 1's cost on the 35B (and a bit more), but none of it on the 27B or on MTP k2 decode. The merge bar was "safe and recovers at least half of mode 1's cost"; the 27B misses it, so it is not merged as the default.

## 1. Design

Mode 1 puts a system fence in every thread between its data stores and the flag store. Mode 7 drops the flag and the fence. A 16-byte word goes out as four 8-byte lines `(dword | seq << 32)`, each one a single 64-bit system-scope atomic store, and the reader polls a word's four lines until all four carry this launch's `seq` (the per-block launch counter; both ranks advance it in lockstep). The reader only uses data whose own tag says it landed, so no ordering between data and flag is needed. The LL scratch is a separate zero-filled fine-grained region, double buffered by counter parity. It doubles the bytes on the wire, so only messages up to `INTARSIA_HIP_AR_LL_MAX` (default 64 KiB) take it; larger ones use the fence path (a pure function of the size, so both ranks agree). Covers `ik_ar2_oneshot`, `ik_ag2_gather` and the fused `ik_ar2_add_rmsnorm` (`kernels/hip/ar_ll.h`, `allreduce.hip`, `ar_norm.hip`). Mode 8 is mode 7 without the tag check, a negative control.

v2 changes (w4): operand loads are issued before the first remote store (`ll_put` takes its word by value; v1 read `x4[k]` lazily after the first atomic store, which serialised loads and stores), and the fused kernel polls 4 items per round (`ll_get_n`). ISA checked with `scripts/ar/hipcheck.sh`: pushes are `global_store_b64 ... scope:SCOPE_SYS`, polls `global_load_b64 ... scope:SCOPE_SYS`, no fence or writeback on the path.

## 2. Safety gates (all pass)

| gate | result |
|---|---|
| GPU tests (`crates/intarsia-backend-hip/tests/comm.rs`): every collective (all-reduce in place and f32 to bf16, all-gather in place and disjoint, fused all-reduce+add+rmsnorm at hidden 2048 and 5120) below, at and above the LL limit, bitwise equal to mode 1 on both ranks | pass (`ll_handshake_is_bitwise_the_fence_mode_across_ops_and_sizes`) |
| stress: 120 rounds x 24 mixed collectives, one rank delayed by random host sleeps (alternating rank), every result checked against the host sum | 0 wrong (`ll_handshake_survives_a_delayed_rank`) |
| negative control: same stress without the tag check (mode 8) | produces wrong sums, and the test asserts that (`ll_without_the_tag_check_is_caught_by_the_stress_test`): the stress test can see a stale read |
| the comm and ar_norm test files under `=1`, `=7`, and `=7` with LL up to 2 MiB | all pass (`ars-w4/pre/tests-*.log`) |
| `kernels test --backend hip --quick` with `=7` | exit 0 |
| tapped soak, 250 fresh processes, short-5, `--spec mtp --spec-k 2`, `=7`, v2 kernels | 0 divergent: 250 taps byte-identical, and identical to the mode-1 tap of `ars-w1`; 250 id lines equal (v1 kernels, w2: also 250/250) |
| greedy ids at tp=2, 18 prompts (`--spec off`, 256 tokens), `=7` vs the main (=3) majority | 18/18 equal |
| Mac `cargo test --workspace --release` | green |

## 3. Speed A/B (w4: one binary, 4 interleaved rounds, rotating arm order; d1 = fence, d3 = old racy default, d7 = LL)

Recovery = mean over rounds of (d7 - d1) / (d3 - d1): 1 means as fast as the old default, 0 as slow as mode 1.

```

ars-w4-speed27 decode tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d7 |
|---|---|---|---|
| 1024 | 56.80 (56.71-56.86, n=4) | 55.30 (54.71-55.67, n=4) **-2.64%** (worst -3.78%) | 55.29 (54.66-55.85, n=4) **-2.66%** (worst -3.83%) |
| 4096 | 56.02 (55.99-56.06, n=4) | 54.76 (54.42-54.88, n=4) **-2.25%** (worst -2.80%) | 54.90 (54.83-54.93, n=4) **-2.01%** (worst -2.19%) |
| 16384 | 52.97 (52.96-52.99, n=4) | 51.98 (51.92-52.01, n=4) **-1.88%** (worst -1.97%) | 52.02 (51.97-52.05, n=4) **-1.80%** (worst -1.92%) |

ars-w4-speed27 prefill tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d7 |
|---|---|---|---|
| 1024 | 3569.47 (3539.67-3592.40, n=4) | 3542.25 (3531.43-3552.86, n=4) **-0.76%** (worst -1.70%) | 3542.85 (3527.65-3563.97, n=4) **-0.74%** (worst -1.80%) |
| 4096 | 3883.39 (3877.29-3889.09, n=4) | 3864.07 (3854.50-3869.19, n=4) **-0.50%** (worst -0.63%) | 3868.14 (3864.51-3870.32, n=4) **-0.39%** (worst -0.55%) |
| 16384 | 3560.67 (3548.90-3582.05, n=4) | 3537.57 (3523.87-3546.82, n=4) **-0.65%** (worst -0.98%) | 3533.80 (3520.81-3542.38, n=4) **-0.75%** (worst -1.11%) |

Recovery of d7 vs d1 (1 = as fast as d3, 0 = as slow as d1): mean over rounds of (new-d1)/(d3-d1); and the paired delta of d7 vs d1
| metric | size | recovery | d7 vs d1 |
|---|---|---|---|
| decode | 1024 | 0.02 (rounds -0.30 to 0.18, n=4) | -0.03% |
| decode | 4096 | 0.09 (rounds -0.02 to 0.32, n=4) | +0.25% |
| decode | 16384 | 0.04 (rounds 0.00 to 0.10, n=4) | +0.08% |
| prefill | 1024 | 4.00 (rounds -3.08 to 19.12, n=4) | +0.02% |
| prefill | 4096 | 0.19 (rounds 0.00 to 0.41, n=4) | +0.11% |
| prefill | 16384 | -0.20 (rounds -0.42 to -0.12, n=4) | -0.11% |


ars-w4-speed35 decode tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d7 |
|---|---|---|---|
| 1024 | 190.65 (189.74-192.05, n=4) | 182.62 (178.37-185.68, n=4) **-4.22%** (worst -6.20%) | 193.35 (193.19-193.47, n=4) **+1.42%** (worst +0.74%) |
| 4096 | 188.95 (188.74-189.10, n=4) | 182.17 (181.29-182.69, n=4) **-3.59%** (worst -3.94%) | 190.28 (190.09-190.46, n=4) **+0.70%** (worst +0.53%) |

ars-w4-speed35 prefill tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)
| size | d3 | d1 | d7 |
|---|---|---|---|
| 1024 | 8722.70 (8451.53-8840.23, n=4) | 8664.17 (8280.42-8847.52, n=4) **-0.69%** (worst -2.02%) | 8751.11 (8642.23-8823.78, n=4) **+0.36%** (worst -1.65%) |
| 4096 | 11938.37 (11721.61-12110.13, n=4) | 11832.74 (11487.24-11989.20, n=4) **-0.89%** (worst -2.00%) | 11962.32 (11861.09-12085.00, n=4) **+0.22%** (worst -1.45%) |

Recovery of d7 vs d1 (1 = as fast as d3, 0 = as slow as d1): mean over rounds of (new-d1)/(d3-d1); and the paired delta of d7 vs d1
| metric | size | recovery | d7 vs d1 |
|---|---|---|---|
| decode | 1024 | 1.35 (rounds 1.22 to 1.52, n=4) | +5.90% |
| decode | 4096 | 1.19 (rounds 1.15 to 1.23, n=4) | +4.45% |
| prefill | 1024 | 4.47 (rounds -1.21 to 12.37, n=4) | +1.08% |
| prefill | 4096 | 0.79 (rounds -0.25 to 2.55, n=4) | +1.13% |


ars-w4 MTP k2 decode tok/s per category (mean of rows; mean over rounds (min-max, n)); paired delta vs d3 (mean, worst round)
| category | d3 | d1 | d7 |
|---|---|---|---|
| rcp-code | 108.74 (108.53-108.92, n=4) | 107.33 (106.97-108.12, n=4) **-1.30%** (worst -1.78%) | 106.96 (106.84-107.07, n=4) **-1.64%** (worst -1.78%) |
| rcp-prose | 83.44 (83.35-83.49, n=4) | 82.19 (82.11-82.34, n=4) **-1.50%** (worst -1.62%) | 82.03 (81.96-82.11, n=4) **-1.69%** (worst -1.77%) |
| rcp-recall | 99.03 (98.81-99.27, n=4) | 97.55 (97.45-97.61, n=4) **-1.50%** (worst -1.75%) | 97.41 (97.18-97.62, n=4) **-1.64%** (worst -1.90%) |
| all | 97.07 (97.01-97.16, n=4) | 95.69 (95.55-95.90, n=4) **-1.43%** (worst -1.62%) | 95.46 (95.36-95.52, n=4) **-1.66%** (worst -1.76%) |

Recovery of d7 vs d1 (1 = as fast as d3, 0 = as slow as d1), per category and over all categories
| category | recovery | d7 vs d1 |
|---|---|---|
| rcp-code | -0.53 (rounds -1.94 to 0.03, n=4) | -0.34% |
| rcp-prose | -0.13 (rounds -0.26 to -0.05, n=4) | -0.19% |
| rcp-recall | -0.10 (rounds -0.32 to 0.13, n=4) | -0.15% |
| all | -0.18 (rounds -0.35 to -0.05, n=4) | -0.23% |
```

Reading: 35B tp2 decode 1k/4k: d7 is +1.4% / +0.7% vs d3 (recovery 1.35 / 1.19) where d1 is -4.2% / -3.6%. 27B MTP off: d7 equals d1 at 1k/4k/16k (recovery 0.02 / 0.09 / 0.04; d7 -2.7 / -2.0 / -1.8% vs d3). MTP k2 rcp-*: d7 equals d1 or is slightly slower (recovery -0.53 / -0.13 / -0.10, all -0.18). Prefill is within noise of d1 and d3 everywhere.

## 4. Why the 27B does not benefit

Kernel micro-rates (`all_reduce_rate` / `ar_norm_rate`, eager launches, us per call; `ars-w4/pre/rate-d*-default.log`):

| call | mode 1 | mode 3 | mode 7 |
|---|---|---|---|
| all-reduce 8 KiB | 11.7 | 5.8 | 4.8 |
| all-reduce 64 KiB | 12.6 | 8.2 | 9.8 |
| fused ar+add+norm, H=2048, 1 row, f32 partial (35B) | 12.5 | 6.4 | 9.6 |
| fused, H=5120, 1 row, bf16 | 15.5 | 9.8 | 9.3 |
| fused, H=5120, 1 row, f32 partial (27B) | 16.0 | 9.6 | 19.5 |
| fused, H=5120, 3 rows, f32 | 16.7 | 11.2 | 40.3 |

The 27B's fused all-reduce+norm with f32 partials at hidden 5120 is slower in LL than in mode 1 (19.5 vs 16.0 us; 3 rows 40.3 vs 16.7): a likely cause (not isolated: the `diag` stage was not run) is that the f32 partial is 20 KiB per row, 40 KiB on the wire with the tags, and a row's norm waits for all of its words. The 35B (hidden 2048) is under that cliff and gains. The plain all-reduce wins only at 8-16 KiB and loses to mode 3 by 64 KiB. The 27B does 129 all-reduces per decode step (1.8 MB per rank), the 35B 81 (0.99 MB).

Not done (the lane's four GPU windows are used): an LL variant for the fused path that keeps the fence kernel for f32 partials with hidden >= 4096 (so `=7` would be LL only where it wins), the staged `diag` (decode tok/s by mode with LL off and mode 8) and `graphrate` (all-reduce cost inside a replayed graph, `ar_graph_rate`) stages in `scripts/ar/stages-soak.sh`, which would confirm the attribution. With the 27B still at mode-1 speed, the fix has to come from there, or from sending the f32 partial as bf16 (not bitwise equal to mode 1, so out of scope here).

## 5. Knobs

`INTARSIA_HIP_AR_DRAIN=7` (LL up to `INTARSIA_HIP_AR_LL_MAX` bytes, default 65536, at most 2 MiB; mode 1 above), `=8` (tests only). Default unchanged (1). `docs/tuning.md` has the rows.
