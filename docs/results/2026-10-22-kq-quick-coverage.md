# `kernels test --quick` covers every registered kernel (lane kq)

Branch `lane-kq/kernel-quick-cover`. Requirements 1 and 3: the quick sweep leaves no registered kernel untested, and a test
fails when one is.

## Before

| Backend | Registered kernels in the quick sweep | With no quick case (`NO CASES (untested)`) |
|---|---|---|
| CUDA (GB10) | 51 (149 cases, from `2026-10-21-m5-reproducibility.md`) | 10: `attn_prefill_bf16_flash`, `attn_prefill_f8kv_flash`, `gdn_delta_rule_bf16_lanes`, `gdn_delta_rule_bf16_split`, `matmul_fp8_gemm`, `matmul_fp8_gemm_f8`, `matmul_fp8_gemm_lt`, `matmul_fp8_gemm_reg`, `moe_experts_nvfp4_mma_bf16`, `moe_experts_w4a16_bf16` (found with the new guard on origin/main's sweep) |
| Metal (endor) | 28, 84 cases | 2: `attn_prefill_bf16_flash`, `attn_prefill_bf16_flash_nax` |
| HIP (R9700) | not measured before the change | not measured |

## Why they were untested

Each shape predicate (`when`) excludes every quick-sweep shape. None of the 12 kernels has a bug found here; the sweep did not
reach their preconditions.

| Kernel(s) | Predicate | Quick sweep had |
|---|---|---|
| flash prefill (CUDA, Metal) | `head_dim` 64/128/256, CUDA also `T > 16`; Metal NAX `head_dim == 128` | `head_dim` 32, `T <= 16` |
| `gdn_delta_rule_bf16_lanes`, `_split` | `dk` 64 or 128 | `dk` 32 |
| `matmul_fp8_gemm*` | `K % 128 == 0`; `_reg` `M >= 256`; `_lt` `N % 16 == 0` | `K` 64 and 96, `M <= 5` |
| `moe_experts_w4a16_bf16`, `moe_experts_nvfp4_mma_bf16` | `T >= 768` / `T >= 1024`, `H` and `I` multiples of 128 | `T <= 36` |

## Change

All in the sweep, none in a kernel or a tolerance. The harness grades each kernel against the `tol` it declared in its
registration, and the new cases use that unchanged.

- `crates/intarsia-core/src/admission.rs`: quick attention heads `(4,2,32)` plus `(4,2,64)` and `(4,2,128)`; one more prefill chunk
  `T=24 pos0=5`; quick GDN shapes plus `(Hk=1, Hv=2, dk=128, dv=32, K=4)` (the shape the `GdnConvDelta` quick sweep already used).
- `crates/intarsia-core/src/quant/cases.rs`: quick `MatmulFp8` gets the prefill-shaped `M=260 N=144 K=256` with the same three
  dtype/act-quant variants the full sweep has at `M=300 N=3080`; quick `MoeExperts` gets `T=1024 E=16 K=8 H=256 I=128`. The full
  sweeps are unchanged (same cases, same random data).
- Guard: `intarsia_core::admission::kernels_without_quick_case(&[KernelImpl])`, used by
  `crates/intarsia-backend-{cuda,hip,metal}/tests/quick_coverage.rs`. It needs no GPU and ignores the arch filter, so any machine
  checks all three backends. One exception: `moe_experts_nvfp4_mma_bf16`'s predicate contains `cfg!(ik_sm120a)`; the CUDA test
  skips that one name when the crate was built without it (a GPU build has it on, and the sweep on spark1 covers it).
- `kernels/README.md`: new section "Adding a kernel and its test".

## Results

| Backend | Where | Command | Result |
|---|---|---|---|
| Metal | endor, fleetq slot | `kernels test --backend metal --quick` | 28 kernels, 121 cases (was 84), 0 FAIL, 0 NO CASES. Raw: `raw/2026-10-22-kq/metal-guard-quick.log` |
| HIP | tatooine, build + window, ref `882aa27b` | `deploy.sh tatooine --backend hip` smoke = `kernels test --quick` | 54 kernels, 288 cases, 0 FAIL, 0 NO CASES, `doctor` rc 0. Raw: `raw/2026-10-22-kq/quick-hip.log` |
| CUDA | spark1 (GB10), ref `86a89163`, fleetq #3758 | `deploy.sh spark1 --backend cuda` smoke = `kernels test --quick` | 54 kernels, 362 cases (was 51 kernels / 149 cases; 3 more kernels are now registered on main), 0 FAIL, 0 NO CASES, `doctor` rc 0, 1.3 s. All 10 former gaps ADMITTED. Raw: `raw/2026-10-22-kq/quick-cuda.log` |
| Guard test, all 3 | endor | `cargo test -p intarsia-backend-{cuda,hip,metal} --test quick_coverage` | 3 passed. `cargo test -p intarsia-core --lib`: 97 passed |

CUDA was run on the final head. `882aa27b` is a first commit of the same crate changes that also carried a build directory by mistake; the code under
`crates/` is identical to the final head (`461a2b83` plus a README commit). HIP was run on `882aa27b` and not repeated.

Quick sweep time after the change: Metal 0.6 s, HIP 0.9 s.

### Guard fails on a removed case

With `tp.push((24, 5))` commented out in `admission.rs` (the case that gives the flash kernels their shape), on endor:

```
every_cuda_kernel_has_a_quick_case FAILED  cuda kernels with no quick sweep case: ["attn_prefill_bf16_flash", "attn_prefill_f8kv_flash"]
every_hip_kernel_has_a_quick_case  FAILED  hip kernels with no quick sweep case: ["attn_prefill_bf16_flash", "attn_prefill_bf16_flash_v2"]
every_metal_kernel_has_a_quick_case ok     (Metal's flash kernels take T=5 at head_dim 64, so they keep a case)
```

Reverted with `git checkout`; the same three tests then pass. Raw: `raw/2026-10-22-kq/guard-negative.log`.

### Downtime

tatooine radiance down 00:11:19 to 00:12:13 EDT (54 s), restored by the EXIT trap, `/v1/models` answered. Log: `raw/2026-10-22-kq/downtime.log`.

### Note on the FP4 experts kernel

`moe_experts_nvfp4_mma_bf16` is ADMITTED at max_abs 1.8e-1 / rel_l2 1.6e-1 on its one quick case. That is inside the contract the
kernel declares (`NVFP4_ACT_TOL`, W4A4: activations are quantised to e2m1, so it differs from the bf16 reference by design) and
nothing was loosened. It is opt-in (`INTARSIA_CUDA_MOE_NVFP4=1`); this only shows that the quick sweep now exercises it.
