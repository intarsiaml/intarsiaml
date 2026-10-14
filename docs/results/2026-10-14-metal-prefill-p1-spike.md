# Metal prefill P1 spike: NAX bf16 GEMM go/no-go (2026-10-14)

**Outcome: GO.** A bf16 GEMM on `mpp::tensor_ops::matmul2d` (Metal 4, the M5 neural accelerators) builds, loads, runs and is correct on endor. At M = 1024 on 12288x4096 (gate/up) it runs at **64.6 TFLOPS** with the same weights every dispatch and **59.2 TFLOPS** with distinct weights (a ~3 GiB pool) streamed from DRAM, against the gate of 20 TFLOPS, and it passes `intarsia kernels test --backend metal` admission at the existing `BF16_TOL`. Precision mode is a non-question: `relaxed_precision` false and true give **bitwise identical output** on every shape and M tested (also on heavy-tailed data) and the same speed, so the strict mode costs nothing and is the one to ship. Every tile configuration gives bitwise identical output too, so the "one tile for all M" constraint of the plan holds trivially: the choice of tile cannot change a row.

The microbench rate is at MLX parity (MLX 0.32.1 measured in the same slot, section 3). The whole-layer mix of the Qwen3-8B GEMMs runs at about 61 TFLOPS at M = 1024 (57 with DRAM-streamed weights), where today's tiled `simdgroup_matrix` GEMM runs at 5.5. Nothing is wired into the model prefill path; the kernel is registered at priority -1 (graded by `kernels test`, never selected).

Host endor (Apple M5 Max, `applegpu_g17s`, macOS 27.0, Xcode SDK `MacOSX27.0.sdk`, Metal compiler 32023.921), branch `metal-p1-gemm` from `spike/metal` at `25127d9`. fleetq: #1871 (device probe), #1873 (first bench, first MLX run), #1877 (tests, admission, final burst, outlier check, first full `cargo test`), #1880 (final `cargo test`, `kernels test` and CLI build after the fingerprint change); all exclusive, all released. `git merge spike/metal` at the end: already up to date (`spike/metal` is still at `25127d9`, no decode-attention work had landed). Raw logs: `docs/results/raw/2026-10-14-metal-prefill-p1/`.

## 1. Step 1: metal4.0 on the device

A 64x32-tile `matmul2d` over device `tensor_inline` operands (bf16 in, bf16 out, `execution_simdgroups<4>`), compiled `-std=metal4.0`, loaded from a separate metallib, pipeline created, run on M = 100, N = 96, K = 64 (M not a multiple of the tile): max abs error 2.8e-2 on outputs up to 11.65 (a bf16 half-ulp at that magnitude). No API, SDK or OS blocker; `xcrun -f metal` resolves with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` as `build.rs` already arranges. Two compile findings for the next person: `tensor_inline` tensors take `slice<ext0, ext1>(o0, o1)` (not `static_slice`, which only exists on documented `tensor_handle`s), and the tensor element type must not be `const` (a `tensor<device const bfloat, ...>` triggers a `static_assert` "Unsupported type" inside the primitive; use `tensor<device bfloat, ...>`).

## 2. The kernel

`kernels/metal/gemm_nax_bf16.metal`, template `gemm_nax_t<BM, BN, NSG, RELAXED>`: `C[M,N] = A[M,K] . W[N,K]^T`, bf16 in and out, one threadgroup per `BM x BN` tile of `NSG` simdgroups. A, W and C are `tensor_inline` views built over the bound pointers with extents `(K, M)`, `(K, N)`, `(N, M)`; `matmul2d_descriptor(BM, BN, dynamic_extent, false, true, RELAXED)` (transpose_right: W is read as stored, no repack) runs the K loop itself and reads operands straight from device memory, as MLX's NAX GEMM does (MLX loads 16x16 fragments per simdgroup by hand into cooperative tensors; the tensor form of the primitive needed none of that to reach the same rate). Interior tiles use static-extent slices (no bounds checks), M, N edges use dynamic slices, which the primitive bounds-checks against the tensor extents; K is dynamic and its tail is zero-filled (K = 17, 18, 33, 70, 100 tested). The accumulator is fp32 inside the primitive and the output is rounded to bf16 once (see section 4: bias -0.001 ulp, so round to nearest even).

Operands: bound at the 4-byte-aligned offset with the 0 or 1 leftover bf16 element passed as `skip` (the convention of `dense.rs` / `common.h`), so any 2-byte-aligned address works. Tested at offsets 0, 2 and 6 (odd element) and 8 bytes (shard 1's weights, 8 mod 16) for x, w and y; the guard bytes around y were untouched. The tensor path needed no special case for 8-byte alignment.

Variants compiled (all in `NAX_VARIANTS`, `src/gemm_nax.rs`): tiles 64x64 (4 simdgroups), 128x64 (4), 64x128 (4), 128x128 (8), each strict and relaxed. Registered kernels: `matmul_bf16_nax` (strict, `gemm_nax_t128x64w4`) and `matmul_bf16_nax_relaxed`, predicate `m >= 16 && k >= 16`, bf16 activations, weights and output, `BF16_TOL`, `Exactness::Deterministic`, priority -1. The launcher is `src/gemm_nax.rs` (new file, in `LAUNCHER_SOURCES`).

Things tried and dropped, kept in the raw data: a float cooperative-tensor destination with an explicit RNE to bf16 and element-wise stores (`_f` variants in `bench-all-tiles-first-run.txt`, `err-tile-and-f32dst-equivalence.txt`): bitwise identical output and 1 to 25% slower; 64x32 and 128x128/4-simdgroup tiles: slower; threadgroup swizzle 0 to 3 (`bench-swizzle.txt`): no effect beyond noise.

## 3. Step 3: TFLOPS

`examples/gemm_nax.rs` (`bench`), the four Qwen3-8B shapes, `TFLOPS = 2*M*N*K / time`, time = GPU timestamps (`GPUEndTime - GPUStartTime`) of 10 back-to-back dispatches in one serial command buffer, median of 15 rounds interleaved across candidates, p10 to p90 in `bench-same-weights.txt`. GPU idle before the burst (ioreg `Device Utilization %`, 3 samples after a 5 s settle): 4, 4, 0; `pmset -g therm`: no thermal, performance or CPU power warning recorded (`gpu-idle.txt`). The "after" ioreg sample read 81, 0, 4: the first sample follows the last benchmark process by under a second, the ioreg figure is a windowed average.

**Default tile (`gemm_nax_t128x64w4`), same weights every dispatch**, strict | relaxed, next to MLX 0.32.1 bf16 `x @ w.T` on the same shapes measured in the same slot, before and after my burst (`mlx_gemm_chain.py`, `chain` = 8 independent GEMMs with 8 distinct weights per `mx.eval`, back-to-back GPU work, the comparable method; `single` = one `mx.eval` per GEMM as in the phase C plan, includes the per-eval launch round trip):

| Shape | M | NAX strict | NAX relaxed | best tile (strict) | NAX, ~3 GiB pool of distinct weights | MLX chain before / after | MLX single before / after (phase C plan) |
|---|---:|---:|---:|---:|---:|---:|---:|
| q/o 4096x4096 | 256 | 51.9 | 51.9 | 54.7 (64x64) | 51.5 | 52.0 / 51.8 | 24.9 / 25.5 (16.8) |
| q/o | 1024 | 60.9 | 60.9 | 60.9 | 60.4 | 56.6 / 55.4 | 45.7 / 47.0 (45.0) |
| q/o | 2048 | 64.0 | 64.0 | 64.0 | 60.1 | 60.7 / 53.5 | 53.5 / 49.2 (51.9) |
| k/v 1024x4096 | 256 | 28.0 | 28.0 | 38.8 (64x64) | 23.5 | 21.1 / 22.4 | 10.8 / 10.5 (5.1) |
| k/v | 1024 | 51.9 | 52.0 | 55.0 (64x64) | 52.0 | 43.4 / 42.4 | 28.4 / 27.2 (28.5) |
| k/v | 2048 | 58.5 | 58.5 | 58.5 | 58.5 | 56.8 / 48.7 | 38.3 / 34.8 (36.8) |
| gate/up 12288x4096 | 256 | 58.8 | 58.9 | 60.1 (64x128) | 58.4 | 57.1 / 57.0 | 38.2 / 41.7 (37.4) |
| **gate/up** | **1024** | **64.6** | **64.6** | 64.7 (64x128) | **59.2** | 58.7 / 54.9 | 56.6 / 54.6 (56.1) |
| gate/up | 2048 | 64.1 | 63.9 | 64.6 (64x128) | 56.0 | 61.5 / 51.5 | 60.3 / 54.9 (60.2) |
| down 4096x12288 | 256 | 52.2 | 52.1 | 52.2 | 50.3 | 48.5 / 47.2 | 34.7 / 35.8 (33.8) |
| down | 1024 | 57.7 | 55.0 | 57.7 | 52.0 | 50.0 / 47.9 | 50.6 / 48.8 (50.3) |
| down | 2048 | 50.5 | 50.3 | 55.0 (64x128) | 51.6 | 46.8 / 44.3 | 53.5 / 50.9 (54.8) |

Whole-layer mix (the seven Qwen3-8B projections' FLOPs over their summed times, default tile, strict): **53 / 61 / 60 TFLOPS** at M = 256 / 1024 / 2048 with the same weights, 57 at M = 1024 with distinct weights; MLX chain 55 at M = 1024 (before).

Reading it:

* **GO gate**: 64.6 TFLOPS strict, 64.6 relaxed at M = 1024 on 12288x4096; 59.2 with the weights streamed from DRAM (a pool of distinct weights totalling about 3 GiB, 30 weights of 100 MB for this shape, as a real layer stack streams its weights). The gate was 20.
* Against MLX, like for like (MLX `chain` streams 8 distinct weights, so compare it with the distinct-weights column): within -9% to +23%, ahead on k/v (+3 to +23%) and down (+4 to +16%), level on q/o and gate/up (the one deficit is gate/up at M = 2048, 56.0 against 61.5 before and 51.5 after). With the same weight every dispatch (column 3) the margin over MLX is larger (+0% to +33%), but that flatters the kernel. The big MLX deficits in the "single" column at M = 256 are launch/sync overhead of one `mx.eval` per GEMM, not kernel speed; the chain column is the like-for-like one. MLX itself dropped 2 to 16% between the before and after runs (gate/up M = 2048: 61.5 to 51.5; possibly thermal, not established); my burst sat between the two MLX runs. Treat all figures as +-8%. No thermal warning level was recorded by `pmset`, but I did not sample clocks.
* The down shapes and k/v at M = 256 are the weak spots. For k/v at M = 256 the likely cause is too few threadgroups for 40 cores (1024/64 x 256/64 = 64 threadgroups with 64x64, 32 with 128x64; not profiled), which is why the smaller tile is 38.8 against 28.0 there. A per-shape tile choice is numerically free (tiles are bitwise equal) and is worth up to +40% on k/v at M = 256 and +9% on down at 2048. The plan's P4 (fused k+v / gate+up) addresses the same thing from the other side.
* Strict and relaxed differ only by noise (the down M = 1024 gap, 57.7 vs 55.0, is inside its own p10-p90 of 49 to 59).

## 4. Step 4: errors, per precision mode

`examples/gemm_nax.rs err`: GPU output against an **f64 CPU reference** (sums in f64 of the bf16 inputs) on 30 sampled rows per case (24 evenly spread plus the last 5 rows, so the M edge is always checked) and all N columns; x ~ U[-1, 1), w ~ U[-1, 1) / sqrt(K) (the admission sweep's scales). Table: default tile, M in {205, 256, 973, 1024, 2048} (205 and 973 are the prompt tails of the 1k prompt), all four shapes; strict and relaxed give the same numbers to every printed digit and the same output hash, so one row serves both (`err.txt` has all 20 shape x M cases for all 8 variants).

| Shape | M | max_abs | max_rel (outputs >= 0.1 rms) | rel_l2 | max err in bf16 ulp |
|---|---|---:|---:|---:|---:|
| q/o | 205 / 256 / 973 / 1024 / 2048 | 3.90e-3 / 3.89e-3 / 3.90e-3 / 3.90e-3 / 3.87e-3 | 3.89e-3 | 1.65 to 1.66e-3 | 0.50 |
| k/v | 205 / 256 / 973 / 1024 / 2048 | 3.89e-3 / 3.88e-3 / 3.87e-3 / 3.82e-3 / 3.87e-3 | 3.87e-3 | 1.65 to 1.66e-3 | 0.50 |
| gate/up | 205 / 256 / 973 / 1024 / 2048 | 3.90e-3 / 3.90e-3 / 3.90e-3 / 3.90e-3 / 3.90e-3 | 3.89e-3 | 1.65 to 1.66e-3 | 0.50 |
| down | 205 / 256 / 973 / 1024 / 2048 | 3.89e-3 / 3.89e-3 / 3.88e-3 / 3.90e-3 / 3.88e-3 | 3.89e-3 | 1.65e-3 | 0.50 |

Reading: 3.9e-3 max relative is 2^-8, the bf16 rounding of the output, and rel_l2 1.65e-3 is the rms of that rounding alone (outputs have rms 0.33). "ulp" is measured at max(|ref|, 0.1 rms) so that near-zero outputs do not dominate; the signed bias is within +-0.002 ulp, i.e. round-to-nearest-even. No non-finite value. The accumulation is as accurate as the reference needs: the f32 accumulation noise is below the output's own rounding. In the admission table the existing tiled `matmul_bf16` shows rel_l2 1.2e-4 against the NAX kernel's 8.1e-5.

Additional checks, all in `tests/gemm_nax.rs` (6 tests, 1.4 s) and `err-outliers.txt`:

* f64 reference on 11 cases with M tails 16, 17, 205, 222, 973, N tails (65, 100, 130), K = 64 to 12288, odd and 8-byte offsets, all 8 variants: worst error 0.97 of the tolerance (half a bf16 ulp + 2e-6 of the L1 mass, the existing `gemm_matches_f64_reference` budget).
* K tails (70, 33, 100, 18, 17): correct, so the registration predicate is `k >= 16`, not `k % 16`.
* **Row independence**: rows of a 1024x1024x4096 call are bitwise equal to the same rows computed alone in calls of 5, 17, 64, 205, 512, 1021 rows at other row offsets, other output offsets (0 or 2 bytes), and with tile configs 64x64, 128x64 and relaxed 128x128/8 simdgroups; and all 8 variants give identical bits for the whole call. This is the CUDA prefill gate (`matmul_fp8_gemm_f8_rows_do_not_depend_on_the_call`) the plan carries over; the primitive's K reduction order does not depend on tile or M.
* Run-to-run deterministic (4 runs, same bits).
* Heavy-tailed data (every 97th activation x100, every 89th weight x30; outputs rms 11, max 57): max error still 0.50 ulp, rel_l2 1.7e-3, relaxed == strict bitwise on all 4 cases x 8 variants.
* **Admission** (`intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B -v`, `kernels-test.txt`, 21 kernels, 252 cases, 37.2 s): both `matmul_bf16_nax` and `matmul_bf16_nax_relaxed` **ADMITTED**, 7 cases each, 0 failures, max_abs 3.185e-3, rel_l2 8.1e-5 (budget `BF16_TOL`: 1.5e-2 abs, 6e-3 rel_l2), against `matmul_bf16`'s 3.185e-3 / 1.231e-4 on its 21 cases. The sweep only exercises M = 32 for matmul (M in {1, 3, 32}, and `m >= 16` leaves 32), so the M tails are covered by `tests/gemm_nax.rs`, not by admission; adding M values such as 17 and 205 to `admission.rs` would close that (core change, not made).
* `cargo test -p intarsia-backend-metal --release`: all pass (`cargo-test-metal-full.txt`, #1880: 6 + 10 + 8 + 6 + 5 + 6 + 6 + 5 + 1 + 5 + 3 tests); `cargo check --workspace`: clean.

Not tested: NaN / Inf propagation, subnormal inputs, and K above 12288; the model-level accuracy gates (KL against the goldens, greedy ids, `qualify`), which belong to the full P1.

## 5. Build mechanism (to adopt)

* `build.rs`: `METAL4_SOURCES: &[&str] = &["gemm_nax_bf16.metal"]`. Every other file in `kernels/metal` is compiled exactly as before, `-std=metal3.1`, into `intarsia.metallib`, whose bytes are unchanged: the diff against `spike/metal` under `kernels/` is the one new file, and the main metallib's SHA-1 (`3acc235d...`) equals that of the metallib built in the main checkout's target directory before this change. The `METAL4_SOURCES` files are compiled `-std=metal4.0 -O3 -fno-fast-math` into `intarsia_metal4.metallib`. A failed metal4.0 build is a cargo warning, not an error: the file is left empty. Adding a metal4.0 kernel file = add its name to `METAL4_SOURCES`, its entry points to `gemm_nax::NAX_VARIANTS`-style list reachable from `kernels::metal4_shaders()`.
* `lib.rs`: `METALLIB4` embedded beside `METALLIB`; `MetalBackend::new` loads it after the main library unless `INTARSIA_METAL_NO_METAL4` is set (non-empty and not `0`); `MetalBackend::new_without_metal4()` and `has_metal4()` for tests and tools. The load is all-or-nothing: if the library is empty, `newLibraryWithData` fails, or any pipeline in `kernels::metal4_shaders()` fails, no metal4 pipeline is kept, `has_metal4()` is false, and the reason is printed once with `INTARSIA_METAL_DEBUG=1`. `Backend::kernels()` then filters out `kernels::requires_metal4` names, so the registry never offers them and `launch` of one by name returns an error.
* The metal4 library bytes are folded into `metallib_hash()` and `has_metal4()` into `Backend::fingerprint()` (`...|metallib:H|metal4:0|1|launch:H`), so cached admission is never shared between a run with the NAX kernels and one without (`INTARSIA_METAL_NO_METAL4`, an old OS); `src/gemm_nax.rs` is in `LAUNCHER_SOURCES`.
* `src/kernels.rs` and `src/lib.rs` are the shared files a parallel agent will also touch: additions are one block next to `shader_names()`, two `TABLE` rows, one registration arm, one `if` in `launch`, and the `new()` split.

## 6. Fallback test

`tests/metal4_fallback.rs` (1 test, 0.6 s): (a) with the library present, builds a `Registry` from `be.kernels()` with the NAX kernels raised to priority 5 (what the full P1 registration will be) and checks the selection for a prefill matmul (M = 205) is `matmul_bf16_nax` and correct; (b) sets `INTARSIA_METAL_NO_METAL4=1` and constructs the backend through `MetalBackend::new()`: `has_metal4()` false, no NAX name in `kernels()`, no `gemm_nax_*` pipeline, the same raised-priority registry now selects `matmul_bf16`, the launch runs and matches the f64 reference, and launching `matmul_bf16_nax` by name is an `Err`; (c) `new_without_metal4()` gives the same. Passed. Limit: the fallback is exercised by forcing it, not on a real macOS < 26 (no such host in the fleet); the code path for a failing `newLibraryWithData` / pipeline is the same all-or-nothing branch.

## 7. What is left for the full P1

1. Registration: keep only the strict kernel (relaxed is bitwise identical and no faster, so `matmul_bf16_nax_relaxed` and the `_r` shaders can go), raise priority to 5, drop the unused tiles. A per-shape tile choice (64x64 for k/v and for M <= 512, 128x64 or 64x128 otherwise) is numerically free here; it should still sit behind the row-independence test, which already checks three tiles.
2. The model-level gates of the plan: `intarsia accuracy --gate-class bf16` on both goldens (KL gate 1e-3; prefill KL now 2.6e-4 and 3.7e-4), greedy ids, `qualify --backend metal` including the chunk-perturbation envelope, all untouched by this spike; decode tok/s unchanged (the GEMV path is unaffected: M = 1 stays at priority 10).
3. Same-slot prefill measurement (`profile.sh` + `analyze_prefill.py`) with the kernel selected, to replace the microbench with real per-shape TFLOPS inside the engine (epilogue-free, weights as the model maps them, `wrap_host` buffers with a non-zero `base`; `launch_nax` goes through `bind()` like the tiled GEMM, but this spike did not run a mapped weight).
4. Admission coverage of M tails (core change to the sweep, or accept the test file as the coverage).
5. Pipeline-creation cost of the 8 metal4 shaders at start-up was not measured (library load plus 8 PSOs on every `MetalBackend::new`); trim the variant list first.
6. The P3 chunk-size decision and the P4 fused k+v / gate+up can now be sized from these numbers: at 55+ TFLOPS the 4k matmul share drops from 10.3 s to about 1.0 s, so attention (P2) becomes the dominant cost, as the plan projected.

## 8. Limits

* Microbench only. No end-to-end prefill number; nothing is selected by the engine.
* One chip, one OS build, one run per configuration of the final burst (the first burst and the swizzle run agree within a few percent). Run-to-run spread is in the raw p10 to p90 columns; down shapes at M >= 1024 are the noisiest (49 to 59).
* The MLX comparison uses MLX 0.32.1's own dispatcher (whatever tile it picks) on random normal data; my kernel on uniform data. TFLOPS do not depend on the data here (a fixed dense multiply), but power-dependent clocks could in principle.
* TFLOPS are GPU-time of dispatches executed back to back; they exclude command-buffer setup and the gaps between dependent kernels in a real prefill.
* `relaxed_precision` has no observable effect for bf16 inputs on this device and SDK; the finding is empirical (bitwise equal over 20 shape x M cases, 4 outlier cases, random uniform and heavy-tailed data), not a documented guarantee, and could differ on another OS or GPU.
* Thermal state: `pmset -g therm` recorded no warning; clocks were not sampled; MLX's two runs bracket my burst and differ by up to 15%.

## 9. Reproduce

```sh
cd /opt/code/intarsiaml   # branch metal-p1-gemm
cargo build --release -p intarsia-backend-metal --example gemm_nax
cargo build --release -p intarsia-cli --features metal
fleetq acquire --host endor --exclusive --mem 24 --label "metal P1 NAX gemm" --wait   # id N
FLEETQ_JOB=N docs/results/raw/2026-10-14-metal-prefill-p1/run.sh      # MLX before, err, bench x2, MLX after
FLEETQ_JOB=N cargo test -p intarsia-backend-metal --release
FLEETQ_JOB=N ./target/release/intarsia kernels test --backend metal -m /Users/josef/models/Qwen3-8B -v
fleetq release N
```

Raw files: `bench-same-weights.txt`, `bench-pool3g.txt`, `bench-swizzle.txt`, `bench-all-tiles-first-run.txt` (includes the dropped float-destination variants), `err.txt`, `err-outliers.txt`, `err-tile-and-f32dst-equivalence.txt`, `mlx_gemm_chain-{before,after}.jsonl` (+ `mlx_gemm_chain.jsonl` from the first slot), `kernels-test.txt`, `cargo-test-nax.txt`, `cargo-test-metal-full.txt`, `cargo-check-workspace.txt`, `gpu-idle.txt`, `run.sh`, `mlx_gemm_chain.py`.
