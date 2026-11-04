# DSv4.1 CUDA prefill: TF-shaped attention block (E2), interleaved softmax (E5) and in-place rope (E3), prepared as one bench job

Date 2026-11-03 (evening), branch `dsv41/attn-tfblock` (worktree `.worktrees/attn-tfblock`, from origin/main `7eeb23f2f`). Follows the Fable report `.worktrees/fable-attn/report.md` (experiments E2, E3, E5 and its "stop" list), [the 2blk doc](2026-11-03-dsv41-attn-2blk.md) and [the fusion doc](2026-11-03-dsv41-attn-fusion.md).

## Status: prepared, NOT compiled with nvcc, NOT run

No nvcc, no GPU and no fleet access tonight (both Sparks are reserved; no fleetq job was queued). There is **no result in this document**. What exists is a single script that builds and runs everything in one fleetq job when a Spark frees up (`fleetq when` says spark2 about 07:05 behind #8255), the patches it applies, and the offline checks listed under "What was checked tonight". `kernels/cuda` on the branch is unchanged; every variant sits behind a `-D` flag in a bench copy, as `attn-2blk.patch` did. Everything is under `docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench/`.

## 1. The variants

| name | flag(s) | what | files |
|---|---|---|---|
| base | none | the unmodified `dsv41_attn.cu` (md5 `b036645e...` as in the 2blk doc) and unmodified `dsv41_misc.cu` | |
| e2 | `-DATTN_TFBLOCK` | E2: new kernel `attn_sparse_mma_tfb`, D = 512 and H % 16 == 0 only (other shapes fall through to the old kernel): 16 heads, 128 threads, grid z = H / 16, the warp's 32 x 4 = 128 Q fragment registers loaded once, `__launch_bounds__(128, 2)`, key metadata prepared one tile ahead (4 barriers per tile, as `attn-2blk.patch`) | `attn-tfblock.patch` |
| e2e5 | `-DATTN_TFBLOCK -DATTN_SM4` | E2 plus E5 | |
| e5 | `-DATTN_SM4` | E5 alone on the old kernel (cheap to add, shows whether the compiler already overlaps the four rows) | |
| ip | `-DROPE_INPLACE` (on `dsv41_misc.cu`) | E3, one thread per (re, im) pair, in place on the 64 rope dims, taken only when `x->data == out->data` | `rope-inplace.patch` |
| ipshfl | `-DROPE_INPLACE_SHFL` | E3, one thread per element exactly as `rope_tail_kernel`, the partner's value from the adjacent lane by `__shfl_xor_sync` | |

`ip` and `ipshfl` are two forms of the same experiment: they differ in how closely they keep the original code shape, which matters because of the contraction trap (below). Both are in the one job because a failed window is expensive and building a second form costs one more `nvcc` of `dsv41_misc.cu`.

Design choices where I deviated from the report's wording:

- **Q goes through the key-tile buffer.** The 16 x 512 Q block is copied once with the coalesced `uint4` loads of the old kernel into `Ks` in the old `Qs` layout (row stride 520; 16640 B, inside the 33280 B of the key tile), a barrier, then each thread reads its fragments with the *same* `lds32(qa + ...)` expressions into `qf[32][4]`, a barrier, and the first key tile overwrites the buffer. Reading 128 fragments per thread straight from global would also work; staging keeps the fragment expressions textually unchanged and the global traffic coalesced (it costs two barriers once per block). The 2blk variant read them from global on every tile, which is what made it flat.
- **A separate kernel with compile-time D = 512**, not the old kernel with a runtime `D`: the `qf` array stays in registers only if the k loop is fully unrolled over a constant. The global `kHeads` / `kThreads` are untouched (the SIMT kernel and the merge kernel use them); the mma-only constants are `kTfHeads`, `kTfThreads`, `kTfD`.
- **E3's minimal form in the report has a race.** "The same kernel body ... with `y == x`": the two threads of a (re, im) pair each load both values, and the even thread's store can land before the odd thread's load; `x` and `y` are also `__restrict__`. Hence the pair kernel (`ip`) and the shuffle kernel (`ipshfl`, where the shuffle orders both loads before either store).

## 2. The exact command

From the root of this worktree (`/opt/code/intarsiaml/.worktrees/attn-tfblock`; keep it: the branch is not pushed):

```
fleetq when --host spark2 --quiet gpu,cpu --mem 14 --expect 20      # read-only: when a job like this would start
fleetq run --host spark2 --quiet gpu,cpu --mem 14 --expect 20 \
    --label "dsv41 attn-tfblock: E2/E5/E3 build + A/B (one job)" -- \
    bash docs/results/2026-11-03-dsv41-attn-tfblock/raw/bench/attn-tfblock-bench.sh
```

Planned duration: 10 to 14 minutes of work (not measured; builds about 4 to 6 minutes at 4 concurrent nvcc, calibration under a minute, 36 attention processes at about 4 s and 24 rope processes at a few seconds), `--expect 20` as an honest high guess. `--mem 14`: four concurrent nvcc at roughly 2 to 3 GB plus the bench and slack; vLLM (38 GB) is a separate service in fleetq's accounting. The script refuses to touch the Spark unless `FLEETQ_JOB` is set (so it only runs inside the slot) and writes its logs to `raw/run/`.

**One job, why that is fine.** The fleetq rule asks builds to say `--load cpu`. This job has a cpu-heavy build phase and then a timing phase; `--quiet gpu,cpu` forbids every other job from loading the GPU or the CPU for the whole run, which covers what `--load cpu` protects (nobody else is disturbed by the build: nobody else is running), and the build is complete before the first timing starts, so the build cannot disturb the timings either. Splitting it into a `--load cpu` job and a `--quiet` job would only add a second queue wait and a second chance for the box to change between build and run.

What the script does (`attn-tfblock-bench.sh`; modes `--dry-run`, `--selftest`, `--remote`):

1. Driver (local, in the slot): creates the new tree `~/intarsia-dsv41-tfblock` on spark2 (the earlier `~/intarsia-dsv41-attnfuse` is deleted), copies `kernels/{abi,common,cuda}` and the bench dir there with tar over ssh (`ik_launch.h` includes `../common/ik_portable.h`), runs the remote half with `ssh -tt` (a dropped connection or the fleetq deadline then hangs the remote command up), and in an exit trap fetches the logs to `raw/run/` and removes `~/intarsia-dsv41-tfblock` also on failure or interruption.
2. Preflight: nvcc, cuobjdump, `nvidia-smi -L`, at least 20 GB `MemAvailable`, earlyoom state, md5 of `dsv41_attn.cu` / `dsv41_misc.cu`.
3. Build: nine compiles (hc, hc_sub, misc base, misc ip, misc ipshfl, attn base, e2, e2e5, e5) and the two bench objects, at most 4 at a time, then seven links (`ab_{base,e2,e2e5,e5}`, `rope_{base,ip,ipshfl}`), the same flags as `build.rs` (`-O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121`) and the link recipe of `ab-build.sh`. `MAX_JOBS=4`, `FLASHINFER_NVCC_THREADS=1`, and a watchdog that kills `nvcc` if `MemAvailable` falls under 10 GB (15 GB without earlyoom), per the Spark OOM rule. The occupancy probe is appended to each attention object.
4. `cuobjdump -res-usage` of every attention object (registers, `LOCAL`), and the SASS summary (FFMA / FMUL / FADD / MUFU counts) of the `rope_tail` kernels of the three misc objects, with the SASS kept in `raw/run/rope-sass-*.txt` so a hash mismatch can be diagnosed without another window.
5. **Calibration** (stop on failure, exit 3): the unmodified binary must give a median within 6 % of 1.30 ms at CSA T=512 and 5.04 ms at CSA T=2048, the five recorded attention hashes (`expected-hashes.txt`, generated from `ab-run-1.log`), the nine recorded rope hashes (from `bench-vec0.txt`), and one active block per SM.
6. **Alternating A/B**: 6 pairs, 15 reps of 20 launches per process; each round runs the three attention variants and the two rope variants, each as an adjacent (base, variant) pair, **odd pairs base first, even pairs variant first** (so drift cannot favour one side). Shapes: CSA T=37, 512, 2048 and HCA T=512 as asked, plus CSA T=11 (the split / merge path) for the hash only, as in the existing bench. Hashes are printed by every process; the occupancy probe by every process.
7. Tables (`tfblock-table.py`): per variant and shape the median of the per-process medians with its range, the ratio, the min/min ratio, the gain, hash equality (against the other side **and** the recorded hash), probe registers / local bytes / active blocks, and the verdict. Stop rules: **any hash mismatch is NEGATIVE (not tuning)**; e2 and e2e5 are NEGATIVE under 12 % gain at CSA T=2048; a variant whose probe does not report 2 active blocks per SM is INVALID; nonzero local memory is annotated as "spills, time not representative". e5 alone has no stop rule (informational). Also printed: E5's increment on top of E2 (`e2e5 / e2`) and, for rope, the extrapolated saving per 4K request (80 layer-chunks x the T=2048 q and o^-1 savings; from the report's count, not measured in situ).
8. Cleanup: build trees removed by an exit trap on the Spark, the whole tree removed by the driver after the log fetch.

## 3. Bit identity: the argument, and where I doubt it

### E2 (`attn_sparse_mma_tfb` against `attn_sparse_mma`)

The report's claim "bit-identical by construction" holds against the code, with the doubts listed below. Per stage, with original line numbers of `kernels/cuda/dsv41_attn.cu`:

- **Tile boundaries do not depend on the head grouping.** A tile is 32 consecutive entries of the *query's* key list (`e = tile * 32 + key`, window rows first, then `idx` picks; `:304-320`). `nkeys`, `ntiles` and the split policy `attn_nsplit(Tn, W, Kw, 32)` (`:489`) use only `Tn`, `W`, `Kw`; the new kernel computes them with the same helper `attn_eff` and the same code. Only the grid's z extent changes (`H / 16` instead of `H / 32`).
- **QK.** Old: warp `w` has `mt = w & 1`, `nt = w >> 1` and runs one chain of 32 `mma.m16n8k16` over `k0 = 0, 16, ..., 496` (`:337-343`) with A fragments `{qa[k0], qa[8 ld + k0], qa[k0 + 8], qa[8 ld + k0 + 8]}` (`qa = Qs + (mt * 16 + g) * ld + 2 tq`) and B fragments `{kb[k0], kb[k0 + 8]}` (`kb = Ks + (nt * 8 + g) * ld + 2 tq`). New: warp `w` has `nt = w`, head group = `blockIdx.z` (= old `2 * blockIdx.z + mt`), the same 32-step chain with the same fragment expressions; `g` and `tq` are unchanged, so every output element is computed in the same lane position from the same operand bits in the same order. The `* scale` stores are unchanged.
- **Softmax.** Warp `w` owns rows `4w .. 4w + 3` in both kernels (4 warps x 4 rows = 16 rows now, 8 x 4 = 32 before); per row the code is copied: `valid[lane]` select, `ik_warp_max` tree, `expf`, bf16 rounding of `p`, `ik_warp_sum` of the rounded values, lane-0 update of `alpha`, `l`, `m`. The state is per row, so a row sees the same sequence of tiles. (`ATTN_SM4` keeps the offsets 16, 8, 4, 2, 1 for each row's max and sum, level by level across the four rows, and the same lane-0 expressions.)
- **PV.** Old: warp `w` owns columns `[64 w, 64 w + 64)` = 8 n-tiles, both m-tiles; per accumulator `acc *= alpha`, then `mma` with `ks = 0`, then `ks = 1` (`:370-410`). New: warp `w` owns `[128 w, 128 w + 128)` = 16 n-tiles, one m-tile; the same per-accumulator sequence. The A fragments are the `Ps` rows `g` and `g + 8` of the block's own 16 heads; the B fragments are `ldmatrix.trans` of the same `Ks` bytes at the same `ks * 16 + (lane & 15)` rows and `cw + 8 j` columns. Which warp issues an accumulator's mma does not enter its value.
- **Epilogue and the split path.** The `1 / (l + expf(sink - m))` expression, the `bf16x2` stores and the partial / `pm` writes are copied with `nh == 16` (the guards `r >= nh` and `mt * 16 < nh` are dropped because the dispatch requires `H % 16 == 0`). I checked the one line the split path depends on, `pm[0] = m; pm[1] = l` written under `warp == 0 && j == 0 && tq == 0` (`:435`): with one m-tile, warp 0 / n-tile 0 / `tq == 0` still covers rows `g` and `g + 8` = all 16 rows once. `part` is indexed by the global head `h`, the merge kernel is unchanged.
- **Key metadata one tile ahead** (taken from the 2blk patch, which hashed identical at all five shapes): the same expressions on read-only inputs (`idx`, `p0`, `W`, `Nc`), computed earlier, stored in the other half of the double buffer.
- **Q**: the same bf16 bits (a copy through shared memory in the old layout).

Where I doubt it, or cannot show it:

1. **It is untested.** Nothing compiled. The remaining risk is *implementation error* (an index, a barrier, a buffer parity), not the design; that is what the T=11 hash (split path), T=37 (grid smaller than the slot count) and the three larger shapes catch. Any hash difference means negative.
2. **Hardware property assumed**: an `mma.sync` output element depends only on its operands and not on the issuing warp or the tile's role; here the lane position `(g, tq)` is also unchanged, which is the strongest form of the assumption.
3. **Contraction.** Expressions are copied, but nvcc decides FMA formation per compilation unit shape; the only mul-add in the softmax is `l_s[r] * a + sum`, the epilogue and scaling are multiplies. A different decision would show up as a hash difference at all shapes. `-lineinfo -O3` as the base.
4. **Dispatch limits.** Only `D == 512` and `H % 16 == 0` take the new kernel; `dpos` (device-position) launches run the same code but the bench does not exercise them (pos = nullptr).
5. The report's remark that small chunks "get no gain and may lose a little" should be read with the new block in mind: at T=37 the grid is 74 blocks of 128 threads on 48 SMs (if spread evenly, 26 SMs get two and 22 one), and a block alone on an SM now has 4 warps where the old one had 8, so each warp does twice the gather and PV work; the T=37 ratio may be above 1 for that reason, not because of a Q tax.

### E5

The same rows, trees and expressions; only the order of independent operations and the number of `__syncwarp` change. The old per-row `__syncwarp` is not needed for correctness: lane 0's stores to `m_s[r]`, `l_s[r]` follow the sum-tree shuffles that every lane has passed, and rows are distinct addresses.

### E3

Same expressions: `re * cs - im * sn` for even elements, `re * sn + im * cs` for odd ones, `ang = (float)(p0 + t * stride) * f[j]`, the same `sincosf`, the same sign flip for `inverse`. `ip` computes both expressions in one thread; `ipshfl` evaluates only its own element's expression as the ternary of the original does. The earlier row-vector kernel changed the expression shape and was not bit-identical; the recorded `bench-vec1.txt` hashes show how sensitive these inputs are (7 of 9 rope shapes differ), so a contraction difference would be seen. The rope bench is `rope_ip_bench.cu`: same shapes, seeds and generators as `hc_rope_bench.cu`, hashes compared with the unmodified kernel in the same job and with the recorded `bench-vec0.txt` hashes. For the in-place mode `y` is refreshed from a pristine `x` before the hashed application (in-place rotation is not idempotent); the timing reps then re-rotate the same buffer. The attention shapes are not used for E3. Not covered: the `dpos` rope path, other dtypes, and the graph wiring (the report's aliasing `q_rope` to `q`, `o_derot` to `o` makes `q` and `o` themselves rotated; whether anything reads the unrotated `q` or `o` afterwards in `graph.rs` was not checked).

## 4. Memory and register arithmetic (by hand; sm_121 figures from the 2blk doc)

Device: 48 SMs, 102400 B shared memory per SM, 1024 B reserved per block, 65536 registers per SM, 1536 threads per SM, default per-block dynamic limit 49152 B.

| | base `attn_sparse_mma` | e2 `attn_sparse_mma_tfb` | TensorFold (from the report) |
|---|---|---|---|
| threads / warps per block | 256 / 8 | 128 / 4 | 128 / 4 |
| dynamic shared | Qs 33280 + Ks 33280 + Ss 32x33x4 = 4224 + Ps 32x40x2 = 2560 = **73344** | Ks 32x520x2 = 33280 + Ss 16x33x4 = 2112 + Ps 16x40x2 = 1280 = **36672** (Q staging 16640 reuses Ks) | 50176 |
| static shared | 3x128 + 128 + 256 = **768** (matches the probe) | 3x64 + 2x32x4 + 2x32x8 = **960** | |
| + 1024 reserved | 75136 | 38656 | 51200 |
| blocks per SM by shared memory | 1 (2 x 75136 = 150272 > 102400) | 2 (2 x 38656 = **77312** <= 102400, slack 25088 B; 3 x 38656 = 115968 > 102400) | 2 (2 x 51200 = 102400, exactly) |
| registers | 107 per thread (cuobjdump, 2blk doc) | cap 255: 2 blocks x 128 threads x R <= 65536 gives R <= 256, allocation granularity makes 255 use 256 x 128 = 32768 per block, **exactly half the file** | 255 |
| threads per SM | 256 | 256 of 1536 | 256 |
| opt-in smem attribute needed | yes (> 49152) | no (36672 < 49152; the launch code sets the attribute anyway, harmless) | |
| grid at T=2048, 32 heads | 2048 blocks, 42.7 waves of 48 | 4096 blocks, 42.7 waves of 96 slots | 4096 |
| grid at T=37 | 37 blocks | 74 blocks | |

The static shared figure for the base (768 B) is what the clang device-only lowering of the unmodified file reports too, which supports the method; the E2 figure is the same count on the new arrays and may differ by padding after ptxas.

Register estimate (a hand count, not a measurement): Q fragments 128 and the O accumulators 16 x 4 = 64 are live across the whole tile loop (192); on top, the QK phase has `c4` (4), the B operands (2 per mma, more if ptxas hoists the 64 shared loads), the shared addresses and loop state (about 20 to 30), the PV phase `a` (4), `b0, b1` (2) and addresses. That is about 215 to 235 against 255, so it fits only if ptxas does not hoist the B loads far ahead. TensorFold's Triton kernel runs at 255 with the same Q-in-registers design, so it is feasible, but a spill (`LOCAL > 0` in `cuobjdump -res-usage`, `local` in the probe line) is the main risk; spills do not change bits, only time, and the table annotates them. The probe line (`active blocks per SM`) must say 2 for e2 and e2e5, 1 for base and e5, else the verdict is INVALID.

## 5. What was checked tonight (all offline)

All outputs are in `raw/selftest/`.

- `bash -n` of the script, and `attn-tfblock-bench.sh --dry-run` (`dry-run.txt`): prints every step with its exact commands (ssh, tar, nvcc job lists, loops), and runs nothing: no ssh, no fleetq, no nvcc.
- `patch --dry-run` of both patches against the current `kernels/cuda` files (in `--selftest`, `selftest.txt`); both apply cleanly. `unifdef -U` of every flag on the patched copies gives back the original files except for blank lines (`unifdef.txt`), i.e. the base build (which uses the unpatched file anyway) is untouched.
- The analysis code on the recorded runs (`selftest.txt`): `calib` passes on `ab-run-1.log` and `bench-vec0.txt` (1.3042 ms, 5.0366 ms, all five attention and nine rope hashes); `ab` on the recorded 2blk A/B prints the 2blk doc's rows exactly (CSA T=512 1.3042 [1.2875, 1.3075] against 1.4182, ratio 1.087 / 1.092; T=2048 5.0366 against 5.0546, 1.004; HCA 1.058; T=37 1.405) and `NEGATIVE (stop rule): gain -0.4 % at CSA T=2048 < 12 %`; `rope` on the recorded row-vector run (`bench-vec1.txt`) prints the hash mismatches and NEGATIVE.
- **Plumbing test** (`plumbing-test.sh`, `plumbing-test.txt`): the real driver and remote halves run end to end against a fake spark (fake ssh into a temp home, real tar pipes), fake nvcc / cuobjdump / nvidia-smi and fake bench binaries that replay the recorded outputs. It shows the right `-D` flag sets reach nvcc (19 invocations), the loops, logs, tables and verdicts work, a calibration failure stops with exit 3, logs are fetched and the remote tree is removed on success and on failure, and the script refuses to run without `FLEETQ_JOB`. The numbers it prints are replays of the old run, not measurements.
- **Parse and device-lowering check without the toolkit** (`syntax-check/check.sh`, `syntax-check.txt`): Homebrew clang 23.1.2 has the NVPTX backend; with minimal stand-in headers (`syntax-check/*.h`, because the CUDA SDK is not installed) all seven kernel configurations (`dsv41_attn.cu` base, e5, e2, e2e5; `dsv41_misc.cu` base, ip, ipshfl) parse, type-check and lower to PTX for sm_120, and the probe-appended attention files and both bench files parse on the host side. Structural counts of the PTX, as expected: tfb kernel 64 `mma.sync` (32 QK + 32 PV, both fully unrolled) against 33 in the old kernel, 6 `bar.sync` sites (Q staging, loop top, gather, S, softmax, epilogue), 40 shuffles in both softmaxes (5 levels x 2 trees x 4 rows, same in e5), static shared 960 B against 768 B. In clang's PTX the three rope kernels (original, `ip`, `ipshfl`) contract `re*cs - im*sn` and `re*sn + im*cs` into the same FMA shapes. **This is a syntax and lowering check, not a compile**: clang is not nvcc, the headers are stand-ins, ptxas never ran (so nothing about registers, spills, occupancy or SASS), and clang's PTX is not what nvcc emits.
- CI's "cargo check (linux, cuda cfgs)" job cannot cover the `.cu` side: `INTARSIA_CUDA_CHECK_ONLY=1` makes `build.rs` return before compiling any kernel, and the workflow comment says it does not check that the `.cu` sources compile. Since `kernels/cuda` is unchanged on this branch it would see nothing either way. It would be relevant only after the variant is wired into the engine.

## 6. Not checked

- **Nothing was compiled by nvcc and nothing ran on a GPU.** Whether nvcc accepts the new kernel and the patches, the register count, spills, whether two blocks really become resident (the occupancy probe decides), every timing, and every hash: unknown. The E2 numbers in the report (15 to 30 % of the kernel) remain the report's reasoning.
- Whether the SASS contraction of the in-place rope kernels matches the original (the clang PTX agreeing is weak evidence); the job collects the SASS and the hash decides.
- The unchecked paths: `dpos` launches of attention and rope, D other than 512, 16 heads per rank, F16 / F32 / SIMT kernels, the engine wiring of any variant (grid, `cudaFuncSetAttribute`, the `graph.rs` aliasing for rope), Metal / HIP no-ops, golden and plandiff.
- The driver half against a real spark2: real ssh and tar, `ssh -tt` behaviour at the fleetq deadline, the fleetq hook's view of the ssh commands, `xargs`, `pkill`, `cuobjdump -sass` function-name format (the awk assumes `Function : <name>` lines), python3 on the Spark (used by earlier lanes). All exercised only against fakes.
- **E4 (packed FP4 keys)** was not started: out of scope unless E2 is complete and simple, and E2 is not yet known to work. The report says to run E2 + E4 together if E2 alone is under 12 %.
- The expected build time and the 14 GB estimate are guesses.

## 7. How to read the outcome

- Calibration FAIL: stop; the numbers on this box or build differ from the recorded ones (check `src-md5.txt`: a changed `dsv41_attn.cu` can legitimately move hashes).
- e2 hashes identical and gain >= 12 % at CSA T=2048: positive; next is the report's step 3 (wire it: grid `H / 16`, `attn_mma_smem_tfb`, rope in place in the graph, one `run-full.sh` window with golden 640/640 and the 32K / 65K `longp` check against main).
- e2 hashes identical, gain under 12 %: negative per the stop rule; look at `e2e5 / e2` and at the register / `LOCAL` line before deciding on E2 + E4.
- Any hash difference: negative, not tuning; the SASS and logs are in `raw/run/`.
- ip / ipshfl: if one is bit-identical on all nine rope shapes it is the E3 candidate (the report expects about 90 ms per request; the table prints the extrapolation); if both differ, E3's in-place form is negative and the contraction is the first thing to read in `rope-sass-*.txt`.
