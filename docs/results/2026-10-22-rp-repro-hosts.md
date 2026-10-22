# RP: reproducible builds across directories and hosts (2026-10-22)

Requirement 5. Follows [M5](2026-10-21-m5-reproducibility.md) (deploy.sh, bench bundle) and [B1](2026-10-21-b1-reproducible-binary.md)
(repro-build.sh). Branch `lane-rp/repro-hosts`; code commit `af77bbf` (docs commits after it do not touch a build input).

## Headline

| Backend | Same sha, two directories | Deploy stamp (`deploy.sh`) | Status |
|---|---|---|---|
| Metal (endor, rustc 1.99 Homebrew) | IDENTICAL (`070a08a6362a6229...`, deploy tree vs a third directory; also `repro-check.sh`) | smoke pass, M5 Max | done |
| HIP (tatooine, pinned ROCm 7.2.4 container) | IDENTICAL (`72688024194830b8...`, two host directories, two cargo volumes, plus the deploy tree) | smoke pass, 54 kernels / 239 cases | done |
| CUDA (spark1 / spark2, nvcc 13.0.88) | PENDING, see "CUDA" | PENDING | see below |
| Cross-host replay | PENDING | | see below |
| CI | green on the PR (`cargo check` linux and macOS, `cargo test` linux, `reproducible build`) | | done |

The HIP two-directory result is weaker than it looks: both trees are mounted at `/work` and the cargo target at `/target`
inside the container, so hipcc never sees the host directory. It shows the container build is deterministic, not that hipcc
embeds nothing. Metal is the real two-path result.

## What changed

* `scripts/deploy-step.sh` / `scripts/deploy.sh`: the build step runs `scripts/repro-build.sh` (path remap, `SOURCE_DATE_EPOCH` =
  commit time, passed in because the deployed tree has no `.git`), falling back to plain `cargo build` for a ref that predates
  repro-build.sh (replaying an old bundle). This is the change PR #62 reverted until CUDA and HIP were verified.
* `crates/intarsia-backend-cuda/build.rs`: see below. Device code is unchanged.
* `crates/intarsia-models/tests/batch_engine.rs`: `bad_decode_row_fails_only_its_request` waits for the scheduler's `finished` count.
* `docs/reproducing-results.md`: the reproducible-binary section now says deploy.sh uses it and what CUDA needs.

## CUDA: what nvcc embeds

First full build at `7354a4b` (nvcc given relative paths only), spark1, `repro-build.sh --features cuda` in `~/rp/A/one` and
`~/rp/B/deeper/two`: DIFFER (`46d4a5de79e859bb...` vs `cbe76e4adad7e375...`). The final binary held 82 strings of the build
directory (`.../kernels/cuda`, `.../kernels/cuda/../common`, the `-lineinfo` directory table inside each cubin) and 23 nvcc temp
names (`tmpxft_<pid>_00000000-6_<file>.cudafe1.cpp`), and every object differed in its anonymous-namespace symbol names
(`_GLOBAL__N__<8 hex>_<n>_<file>_cu_...`). Four small experiments (`raw/nvcc-experiment-N.out`, 4 kernels each, spark1):

| Experiment | Result |
|---|---|
| relative path + `-frandom-seed=<stem>`, with or without `-lineinfo`, with `-ffile-prefix-map` | still DIFF in two directories: the hash in `_GLOBAL__N__<hash>` comes from the absolute source path (`nvcc --dryrun` shows `--orig_src_path_name /abs/path`), not from the seed |
| same directory compiled twice | 2 bytes differ: the `tmpxft_<pid>_` ELF file symbol |
| fixed absolute path (copy of `kernels/` at the same place for both trees), raw | DIFF (the pid name) |
| same, `objcopy --strip-unneeded` on the object | DIFF by 6 bytes in `.nv_fatbin`, 3 symbol names per cubin: the hash also depends on the file's mtime |
| same, files stamped with a fixed mtime, then `objcopy --strip-unneeded` | SAME for all 4 objects |

Fix (in `build.rs`, Linux only, std only): copy `kernels/{cuda,common,abi}` to `/tmp/intarsia-cuda-kernels-<FNV-1a content hash>`
(atomic rename; files and directories get a fixed mtime; an existing directory is reused only after its content re-hashes to
the same value, otherwise the build falls back to compiling in place with a warning), run nvcc there with relative paths and
`-I .`, and run `objcopy --strip-unneeded` on each object (missing objcopy: warning, build still works).
`INTARSIA_CUDA_NO_STAGE=1` compiles in place. Runtime assert messages from the kernels now show `/tmp/intarsia-cuda-kernels-.../cuda/x.cu`
instead of the checkout path.

## CUDA results

PENDING (queued behind Flare #3555, which holds both Sparks exclusively).

## Cross-host replay

PENDING.

## CI: `bad_decode_row_fails_only_its_request`

Not GPU bound: the test uses the cpu backend. It failed in CI runs 39638834059 and 39642946027 (both `lane/m5-reproducibility`)
at `batch_engine.rs:295`, `(m.failed, m.finished)` left `(1, 2)` right `(1, 3)`. The request's `Done` event is sent before the
scheduler counts it finished (`engine/batch.rs` `finish` calls `report_done`, the count follows in `scheduler.rs`), so the
test read the metrics one count early on a slow runner. The test now polls for `failed + finished >= 4` (bounded, 5 s) before
asserting. It passes 40 of 40 runs under 8 busy loops on endor (`raw/ci-bad_decode-x40-under-load.out`); the pre-fix failure
rate under the same load was not measured, so that run shows no regression rather than a reproduction. The fix is test-only.
CI on the PR is green. The CUDA/HIP GPU tests are skipped there as before (no GPU on the runner).

## Fleet use and downtime

* Jobs (fleetq): endor #3604, #3605, #3606, #3694, #3735; tatooine #3588, #3591, #3595, #3736, #3737; spark1 #3677, #3682, #3688, #3689 (nvcc experiments, 1 to 5 min, vLLM up, non-exclusive); others pending (see CUDA).
* tatooine radiance down (taken down only for the GPU smoke, EXIT trap restores, waits for `/v1/models`): 15 s, 23 s, 33 s = 71 s in 3 windows (`raw/downtime.log`); the longest was 33 s, well inside the 30 min cap. The builds ran with radiance up (CPU only).
* No Spark vLLM taken down by this lane.

## Mistake to report

Cleaning up my own queued jobs I used `pkill -f "fleetq run --host ..."` three times (endor around 23:04, spark1 around 23:34
and 00:27 EDT). Those patterns also match other agents' `fleetq run` clients on this Mac. Jobs of other owners that left the queue within
a minute of those commands: #3572, #3565 (endor); #3553, #3643, #3638, #3639, #3649 (spark1, around 23:34); #3656 (spark1, 00:27) and
#3658, #3667, #3699 (dropped "client went away" at 00:28). Their owners appear to have resubmitted (#3656 reappeared as a
"requeued" job, #3699 as #3762, #3667 as #3763) but they lost their queue position. No job of another owner was running at those
moments, and no `--takes-down` job was affected. Later cleanup used `fleetq release <id>` and saved PIDs only.

## Not done / uncertain

* The objcopy and staging step has not been run on a Linux host other than spark1/spark2 (CI runners have no nvcc, so the CUDA branch of `build.rs` is not compiled there beyond the stub path).
* Cross-toolchain identity (different rustc or nvcc versions) is not claimed.
* `-frandom-seed` is unnecessary and not used.
