# RP: reproducible builds across directories and hosts (2026-10-22)

Requirement 5. Follows [M5](2026-10-21-m5-reproducibility.md) (deploy.sh, bench bundle) and [B1](2026-10-21-b1-reproducible-binary.md)
(repro-build.sh). Branch `lane-rp/repro-hosts`; code commit `af77bbf` (docs commits after it do not touch a build input).

## Headline

| Backend | Same sha, two directories | Deploy stamp (`deploy.sh`) | Status |
|---|---|---|---|
| Metal (endor, rustc 1.99 Homebrew) | IDENTICAL (`070a08a6362a6229...`, deploy tree vs a third directory; also `repro-check.sh`) | smoke pass, M5 Max | done |
| HIP (tatooine, pinned ROCm 7.2.4 container) | IDENTICAL (`72688024194830b8...`, two host directories, two cargo volumes, plus the deploy tree) | smoke pass, 54 kernels / 239 cases | done |
| CUDA (spark1 and spark2, GB10, nvcc 13.0.88, rustc 1.90.0) | IDENTICAL on both hosts (`fcd8053febb15f14...`), and the same hash on spark1 and spark2 | smoke pass on both (54 kernels, 292 cases), device code identical to base (SASS, 22/22 objects) | done |
| Cross-host replay (spark1 bundle on spark2) | every stamp row matches including the binary hash; Qwen3-8B greedy 128 token ids byte-identical | M5 bundle too (sha, lock, toolchain, GPU, driver match; 149-case kernels output equal) | done |
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
(`_GLOBAL__N__<8 hex>_<n>_<file>_cu_...`). Small experiments (`raw/nvcc-experiment-N.out`, 4 kernels each, spark1):

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

## CUDA results (GB10, nvcc 13.0.88, rustc 1.90.0, `af77bbf`; spark2 fleetq #3732, spark1 fleetq #3797; non-exclusive, vLLM left alone, no model loaded except the replay below)

| Check | spark2 | spark1 |
|---|---|---|
| `repro-build.sh --features cuda` in `~/rp/A/one` and `~/rp/B/deeper/two` (36 s each) | IDENTICAL, `fcd8053febb15f14...` | IDENTICAL, `fcd8053febb15f14...` |
| `deploy.sh <host> --backend cuda` (build through repro-build in `~/intarsia-deploy/cuda[-rp]`) | binary `fcd8053febb15f14...` | binary `fcd8053febb15f14...` |
| `doctor` / `kernels test --quick` | rc 0 / 54 kernels, 292 cases, 0 failures (10 kernels have no quick case: untested, not failed) | same |
| stamp | `raw/DEPLOY_STAMP-cuda-spark2.json` | `raw/DEPLOY_STAMP-cuda-spark1.json` |

So the same sha gives the same bytes in three directories on one host and on two different machines (aarch64, same toolchain versions).
Device code unchanged: `cuobjdump -sass` of every kernel object, base build (origin/main merge base `f908732`, plain cargo build) vs this branch,
on spark2: 22 of 22 objects identical (695,820 normalised lines; only the `_GLOBAL__N__<hash>` names and the `identifier = <path>` lines were normalised; `raw/sass-recheck.out`).

The deploy smoke runs the kernels test on the linked, `objcopy`-stripped binary, so it launches kernels through the registered fatbins; it is a
functional check of the build, not one of the model gates (KL21, 256 greedy ids, prefix reuse, goldens): those were not run, the
device code is bit-identical per the SASS comparison, and the coordinator's CUDA no-op check on spark1 covers them before merge.

## Cross-host replay (spark1 bundle on spark2)

The M5 bundle (`5847f06`, packed from a spark1 deploy on 2026-10-21, 12 files, checksums verified on unpack) was replayed on spark2
by `bench-bundle.sh replay` (`raw/spark2-repro-sass-deploy-replay.out`, tree `replay-m5`; deploy.sh fell back to a plain `cargo build`
because that sha predates repro-build.sh):

| Row | Result |
|---|---|
| git sha, Cargo.lock `b64a06d84a56f46e`, rustc 1.90.0, nvcc 13.0.88, GPU GB10, driver 580.173.02 | match |
| binary sha256 | DIFFER (`823b41be...` on spark1, `90acdf46...` on spark2), expected: that sha was built with plain cargo in another directory |
| the recorded command (`kernels test --quick --json`), 149 cases | `kernels.json` of the replay equals the original in all but 2 of 2464 lines (`free_mem` and the measured `bandwidth_gbs`) |

Replay of a bundle made by this lane (`af77bbf`, packed on spark1 in fleetq #3797: Qwen3-8B BF16, `generate` greedy, 128 tokens, `raw/rp-bundle.tgz`), on spark2 (fleetq #3880, tree `replay-rp`, `raw/spark2-replay-greedy-ids.out`):

| Row | Result |
|---|---|
| git sha, Cargo.lock `7723f17b4e7280cb`, rustc 1.90.0, nvcc 13.0.88, GPU GB10, driver 580.173.02 | match |
| binary sha256 | **match**, `fcd8053febb15f14...` (built on spark2 from the exported sha, deploy tree path differs) |
| generated text (`ids.out`, 622 bytes) and the 128 token ids line in `gen.err` | byte-identical to the spark1 original (`raw/spark1-original-output/`, `raw/spark2-replay-output/`) |

The replay command needed one change: spark1 reads the model through `~/models/Qwen3-8B`, a link to `/mnt/models/Qwen3-8B`; spark2 has the model only under `/mnt/models` (file sizes and config/tokenizer sha256 equal on both), so the replay used `--cmd` with that path. The first attempt failed on the missing path (`raw/spark2-replay-attempt1-model-path.out`). The script's own ids comparison printed `GREEDY-IDS-IDENTICAL` over two empty selections (it grepped the wrong file); the comparison above was redone by hand with `cmp` and an md5 of the `ids:` line. Speed was not compared (prefill 511 ms / decode 15.26 tok/s on spark1, 466 ms / 15.71 tok/s on spark2, both with other jobs on the box).

## CI: `bad_decode_row_fails_only_its_request`

Not GPU bound: the test uses the cpu backend. It failed in CI runs 39638834059 and 39642946027 (both `lane/m5-reproducibility`)
at `batch_engine.rs:295`, `(m.failed, m.finished)` left `(1, 2)` right `(1, 3)`. The request's `Done` event is sent before the
scheduler counts it finished (`engine/batch.rs` `finish` calls `report_done`, the count follows in `scheduler.rs`), so the
test read the metrics one count early on a slow runner. The test now polls for `failed + finished >= 4` (bounded, 5 s) before
asserting. It passes 40 of 40 runs under 8 busy loops on endor (`raw/ci-bad_decode-x40-under-load.out`); the pre-fix failure
rate under the same load was not measured, so that run shows no regression rather than a reproduction. The fix is test-only.
CI on the PR is green. The CUDA/HIP GPU tests are skipped there as before (no GPU on the runner).

## Fleet use and downtime

* Jobs (fleetq): endor #3604, #3605, #3606, #3694, #3735; tatooine #3588, #3591, #3595, #3736, #3737; spark1 #3677, #3682, #3688, #3689 (nvcc experiments, 1 to 5 min, vLLM up, non-exclusive), #3797 (repro builds, deploy, pack); spark2 #3732 (repro builds, SASS, deploy, M5 replay), #3872 (failed on the model path), #3880 (replay). Dropped or released before running: #3578, #3586, #3590, #3580, #3654, #3697, #3698, #3731, #3734 (client killed by another agent, requeued as #3797).
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
