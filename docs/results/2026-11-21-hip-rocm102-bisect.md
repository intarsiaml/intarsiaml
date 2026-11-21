# HIP: why ROCm 10.2 moves the 35B logits (lane rocm102-bisect)

Status: 2026-11-21, diagnosis done. **Nothing is flipped:** the images, `dev.sh` and every default are unchanged. The fix is
opt-in (`-DIK_GDN_CHUNK_STABLE_LOG=1`, off by default). Follows `2026-11-21-hip-rocm102.md` (PR #484), whose 10.2 arm failed
qualify's same-build gates: mean KL21 8.6e-3 against a 1e-4 limit, about 200 top-1 flips.

## Answer

- **Cause: the compiler, not the runtime.** The tested ROCm 7.2.4 kernels give 7.2.4's bits on the 10.2 runtime, and 10.2's
  kernels give 10.2's bits on the 7.2.4 runtime.
- **The TU: `kernels/hip/gdn_chunk.hip`.** Of the 28 kernel TUs, it is the only one whose 10.2 object moves the logits. Swapped
  alone into a 7.2.4 build, it gives the full 10.2 shift, to every printed digit (KL 8.5756e-3, 228 flips).
- **The op: `logf` in `gdn_chunk_norm_kernel`**, which computes the log q / k normalisation factors `lq`, `lk`. AMDGPU expands
  `llvm.log.f32` into `v_log_f32` (log2), then multiplies by ln 2 in two parts (hi `0x3f317217` + lo `0x3377d1cf`):
  - **7.2.4:** the last step is an FMA, `out = fma(y, ln2_hi, err)`.
  - **10.2:** the last step is a multiply plus a separate add, `out = y * ln2_hi + err`.

  That moves `lq` / `lk` by up to about half an ulp. The chunked kernels fold them into the decay exponents (the
  normalised-k gram is `P_raw[i][j] * e^{lk_i + lk_j}`), so the change reaches every GDN layer's prefill output and then the
  logits.
- **10.2 is the compiler that follows the TU's own rule.** `gdn_chunk.hip` says `#pragma STDC FP_CONTRACT OFF` ("no fp
  contraction, as libr4d is built"). Compiling with 7.2.4 plus `-ffp-contract=off` gives **exactly the 10.2 bits** (KL
  8.5756e-3, 228 flips). So 7.2.4 fused this one add against the pragma, and 10.2 does not. That 7.2.4's backend took
  the TU-wide default rather than the pragma at the call site is an inference: the result fits it, but I did not trace it
  in LLVM.
- **The device libraries are not the cause.** 10.2's clang with 7.2.4's device-library bitcode
  (`--rocm-device-lib-path`) gives the 10.2 bits. The library swap did change the gate kernel's `log1pf`: one exponent
  extract becomes `v_frexp_exp_i32_f64` instead of `_f32`. Those are the same value for every finite float, and the
  logits show it changes no bits.
- **The 10 kernels with no admission cases are cleared.** They are all bf16 / fp8 WMMA matmuls (`matmul_bf16_wmma`,
  `matmul_fp8_wmma`, the pair / rmsnorm / gated / sigmoid fusions). Their TUs are `matmul.hip` and `fp8.hip`, and the
  swap scan cleared both: a 10.2 object in a 7.2.4 build gives 7.2.4's bits.

## Recommendation

**A code change makes 10.2 reproduce 7.2.4's bits; a flag does not.** `gdn_chunk.hip` now has an opt-in `gc_logf`
(`IK_GDN_CHUNK_STABLE_LOG=1`). It writes out 7.2.4's `logf` expansion with explicit `__builtin_fmaf` calls:
`v_log_f32`, the same denormal scaling, then hi / lo ln 2 ending in an FMA. Explicit FMAs leave no contraction choice to
the compiler. Measured with it:

| build | tp1 acc-oracle self-golden | tp2 self-golden |
|---|---|---|
| 7.2.4, as on main (`r7ctl`) | KL 7.3853e-13, 0 flips, PASS | KL 1.8737e-13, 0 flips, PASS |
| 10.2, as on main (`r10ctl`) | KL 8.5756e-3, 228 flips, FAIL | KL 8.5879e-3, 205 flips, FAIL |
| **10.2 + fix, full build (`f10`)** | **KL 7.3853e-13, 0 flips, PASS** | **KL 1.8737e-13, 0 flips, PASS** |
| 7.2.4 + fix, full build (`f7`) | KL 7.3853e-13, 0 flips, PASS | not run |

The fixed results equal 7.2.4's to every printed digit. The 7e-13 / 2e-13 floor is the same for the recorded build itself.
So with the fix, the checked-in 35B HIP baselines hold on both images, and a 10.2 switch would need no re-recording for
this.

Options, for the user to decide:

1. **Make `gc_logf` unconditional** (my pick). A one-file code change. It keeps 7.2.4's bits on 7.2.4 (`f7`, `x-c7fix`)
   and gives the same bits on 10.2 (`f10`). Before merging, run the usual qualify on 7.2.4 (`kernels`, acc-short,
   acc-oracle tp1 / tp2). Only the acc-oracle gates were run here.
2. **Switch to 10.2 and re-record instead.** Accuracy against vLLM is equivalent (#484 section 2), and 10.2's rounding is
   the one the TU's pragma asks for. This costs a baseline re-record (35B acc-short + envelope + oracle tp1 / tp2 dumps,
   and the 27B) and gives up cross-image bit equality.
3. **No flag route.** `-ffp-contract=off` on 7.2.4 moves 7.2.4 to 10.2's bits, the opposite direction. I know of no 10.2
   flag that re-enables fusion inside the `log` expansion while keeping the pragma, and none was tried. `-ffp-contract=fast` for this TU
   would also contract the source's own multiply-adds, which the pragma keeps unfused on purpose.

This is the only `logf` in `kernels/hip`. Other libm calls (`expf`, `rsqrtf`, `log1pf`, and the fp8 conversions) give
the same bits under both compilers in this model's path, per the swap scan. Some other model or path could still hit a
different lowering change. A pinned-compiler bit-equality check is the general guard: `qualify`'s same-build gates already
are one.

## Method

Fleet jobs on tatooine: #10531 (both images' CLIs + 30 relinks, CPU), #10533 (window 1), #10534 (window 2), #10537 /
#10539 (single-TU compiles, CPU), #10542 (window 3). The first window 3 attempt, #10541, exited at its `FLEETQ_JOB` guard
before it stopped radiance. Radiance came back after every window (window logs `[window] radiance answers`). Tree `bd15509`
(code-tree `9712db1e34dc`) on the two lane images of #484 (`intarsia-hip-dev:rocm7.2.4-rust1.90.0`,
`intarsia-hip-dev:rocm10.2.0-rust1.90.0`). Target volumes `intarsia-cargo-target-rocm102b-{r7,r10}`. Everything is
under `~/intarsiaml-rocm102-bisect/.bisect` on tatooine.

**Relinking with mixed kernel archives.** `build.rs` gets an opt-in `INTARSIA_HIP_KERNELS_NOBUNDLE=1`, off by default and a
debugging aid only. With it, the kernel archive is linked `static:-bundle`, so the CLI's final link reads
`libintarsia_hip_kernels.a` from a `-L` directory. `scripts/rocm102/bisect/mkbins.sh` builds the CLI in each image, captures
rustc's final link command (`--print link-args`, `-C save-temps`) and copies out `build.rs`'s per-TU objects. `relink.sh`
then re-runs that link command with an archive mixed from both compilers' objects, in about 0.5 s per variant.
The Rust objects are 10.2's (`LINK=r10`) in every relinked binary. Controls: `mixnone` (every TU from 7.2.4, relinked) and
`mixall` (every TU from 10.2, relinked) reproduce the cargo-built 7.2.4 / 10.2 binaries' results exactly. The relink
therefore adds nothing, and the Rust side's compiler does not matter.

**The check.** `qualify`'s acc-oracle drift check, run by hand: `accuracy --self-golden` against the checked-in
`golden/baselines/qwen36-35b-a3b-nvfp4-cpu-oracle.hip{,-tp2}.selfgolden.safetensors` (recorded at `4379dd0` on 7.2.4),
`--mode prefill --decode-positions 16 --max-seq 4096`. The same gate shape and limits as qualify's (KL 1e-4, 27 flips).
Each run also did a 4-token `generate` with `INTARSIA_DET_TAP`. The tap and the tokens did not separate even the two
cargo-built binaries on that short prompt, so only the drift check is used as evidence.

### Window 1: compiler vs runtime (tp1)

| binary | runtime image | kernels from | KL21 | flips | gate |
|---|---|---|---|---|---|
| r7ctl | 7.2.4 | 7.2.4 | 7.3853e-13 | 0 | PASS |
| x7on10 | 10.2 | 7.2.4 | 7.3853e-13 | 0 | PASS |
| r10ctl | 10.2 | 10.2 | 8.5756e-3 | 228 | FAIL |
| x10on7 | 7.2.4 | 10.2 | 8.5756e-3 | 228 | FAIL |
| mixnone-on7 / -on10 | both | 7.2.4 (relinked) | 7.3853e-13 | 0 | PASS |
| mixall-on7 / -on10 | both | 10.2 (relinked) | 8.5756e-3 | 228 | FAIL |

The bits follow the compiler and ignore the runtime in both directions.

### Window 2: one TU at a time (tp1, 10.2 runtime)

Each `s-<tu>` binary is 7.2.4's kernels with that one TU's object from 10.2. That covers 28 TUs: allreduce,
allreduce_wht6, argmax, ar_norm, attention, attention_flash, attention_splitkv, attention_win_splitkv, bridge, dflash,
elementwise, fp8, gate_qkr, gdn, gdn_chunk, gdn_fused, ik_runtime, matmul, moe, moe_block, moe_block_grouped, moe_shared,
mxfp4, mxfp4_a8, mxfp4_a8_gemm, norm, nvfp4, topk.

- **27 TUs:** KL 7.3853e-13, 0 flips, PASS: 7.2.4's bits.
- **`s-gdn_chunk`:** KL 8.5756e-3, 228 flips, FAIL: 10.2's full shift.

### Window 3: what inside gdn_chunk (tp1, 10.2 runtime; 7.2.4 kernels except gdn_chunk)

| gdn_chunk object | KL21 | flips | gate |
|---|---|---|---|
| x-c7: 7.2.4, recompiled with build.rs's flags | 7.3853e-13 | 0 | PASS |
| x-c10: 10.2, recompiled | 8.5756e-3 | 228 | FAIL |
| x-c10dl7: 10.2 clang, 7.2.4 device libs | 8.5756e-3 | 228 | FAIL |
| **x-c7nc: 7.2.4 + `-ffp-contract=off`** | **8.5756e-3** | **228** | **FAIL** |
| x-c10nc: 10.2 + `-ffp-contract=off` | 8.5756e-3 | 228 | FAIL |
| x-c10dl7nc: 10.2, 7.2.4 devlibs, contract off | 8.5756e-3 | 228 | FAIL |
| x-c7fix: 7.2.4 + `IK_GDN_CHUNK_STABLE_LOG=1` | 7.3853e-13 | 0 | PASS |
| **x-c10fix: 10.2 + `IK_GDN_CHUNK_STABLE_LOG=1`** | **7.3853e-13** | **0** | **PASS** |

**The ISA.** The full kernels are in `raw/norm-kernel-*.s`. The norm kernel's two `logf` calls, after `v_log_f32` (y):

```
7.2.4:  r = v_mul_f32  0x3f317217, y
        e = v_fma_f32  0x3f317217, y, -r
        e = v_fmac_f32 0x3377d1cf, y      ; e += y * ln2_lo
        e = v_fmac_f32 0x3f317217, y      ; out = fma(y, ln2_hi, e)
10.2:   r = v_mul_f32  0x3f317217, y
        e = v_fma_f32  y, 0x3f317217, -r
        e = v_fmac_f32 0x3377d1cf, y
        out = v_add_f32 r, e              ; separate add, no fusion
```

Across all of gdn_chunk's kernels, the FP op mix also differs in the gate kernel, which holds the device library's
`log1pf`. The c10dl7 result above shows that difference changes no bits. The scan and the WMMA `kkt` kernels have the
same FP op counts. They differ only in VOPD dual-issue packing, which does not change rounding.

## Files

- `crates/intarsia-backend-hip/build.rs`: `INTARSIA_HIP_KERNELS_NOBUNDLE` (opt-in, debugging).
- `kernels/hip/gdn_chunk.hip`: `gc_logf`, opt-in via `IK_GDN_CHUNK_STABLE_LOG`. The default build is unchanged: `logf`.
  Checked with fleetq #10543: on both compilers, the edited file's default ISA equals the pre-edit file's except for the
  `__hip_cuid_*` symbol name, a hash of the source. That gives 0 differing instructions.
- `scripts/rocm102/bisect/`: `lib.sh`, `mkbins.sh`, `variants.sh`, `relink.sh`, `tu.sh`, `exp1.sh`, `exp2.sh`,
  `stages.sh`, `window.sh`, `sync.sh`, `tapcmp.py`, `kernmap.py`.
- `raw/`: per-window `batch.log` / `steps.log` and every `*.acc.out` report, CPU job logs, the norm-kernel ISA.
