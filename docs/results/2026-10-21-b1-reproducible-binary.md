# B1: reproducible release binary (2026-10-21)

## Headline

The same git sha now builds to the same `target/release/intarsia` bytes in different directories on endor, for the default
(CPU) build and for `--features metal`. Spark1 CUDA is NOT verified (two builds failed in `intarsia-backend-cuda`'s
`build.rs` for a reason not yet found, see below). How to use it: [docs/reproducing-results.md](../reproducing-results.md#reproducible-binary).

## Cause and fix

Baseline (plain `cargo build --release --locked`, two directories, endor): `32dc02bf...` vs `98d97cab...`. The binary
contained the build directory twice: `env!("CARGO_MANIFEST_DIR")` in `intarsia-cli` (`accuracy.rs` `default_golden`,
`accuracy/refnoise.rs` `sidecar_candidates`, a runtime fallback to the checkout's `golden/`), plus registry and sysroot paths
(`~/.cargo/registry/...`, `/opt/homebrew/Cellar/rust/...`) in panic locations. `env!` is not touched by `--remap-path-prefix`.

* `golden/` fallback now found from the executable's location (`checkout_golden_dir()`: first ancestor with `golden/` next to a `Cargo.lock`). Same behaviour for a binary run from `<checkout>/target/<profile>/`.
* `scripts/repro-build.sh`: `--remap-path-prefix` for the tree, target dir, `CARGO_HOME` and the rustc sysroot; `SOURCE_DATE_EPOCH` (commit time) and `ZERO_AR_DATE`; fixed locale and TZ; `CARGO_INCREMENTAL=0`. The git sha stays in the binary through `INTARSIA_BUILD_STAMP` (an input).
* deploy.sh / deploy-step.sh are NOT changed (reverted before merge): repro-build.sh is opt-in until CUDA and HIP are verified; `scripts/repro-check.sh` + CI job `repro` (ubuntu, CPU-only).
* Not needed: build.rs changes. The metal, cuda and hip build scripts embed no timestamp or path (the metallib was byte-stable on endor).

## Numbers (endor, rustc 1.99.0 Homebrew, no fleet role involved; non-exclusive slots)

| Build | Dir A | Dir B | Result |
|---|---|---|---|
| default, plain cargo (baseline) | `32dc02bf...` | `98d97cab...` | DIFFER |
| default, repro-build.sh | `5e2f3288e151af25...` | same | IDENTICAL |
| `--features metal`, repro-build.sh | `6c4bb8f1b31d5d73...` | same | IDENTICAL |
| `scripts/repro-check.sh` (git archive HEAD, temp dirs, SOURCE_DATE_EPOCH = commit time) | `99b619673be4d91a...` | same | IDENTICAL |

Fleetq jobs: endor #3194, #3202, #3208, #3236.

## Not verified

* **CUDA on spark1: not verified.** Two attempts (fleetq spark1 #3241, #3274, non-exclusive, no role down) built in `~/b1repro/{A/one,B/deeper/two}` and both exited 101 inside `intarsia-backend-cuda`'s build script, with no stderr in the log (`raw/spark1-cuda-attempt2.log`). The second attempt added `/usr/local/cuda/bin` to PATH (as deploy.sh does) and failed the same way. Candidate causes: env that deploy.sh sets and a bare ssh lacks (`CUDA_HOME`, arch, `LD_LIBRARY_PATH`); the copy via `tar` from macOS leaving something out (the `.claude/` exclude, extended attributes); the script expecting a `git`-exported tree. A debug run of the build script (#3304) was queued; if its result is not appended below, it did not run before this PR was opened.
* HIP (tatooine container) not exercised. Cross-host identity not tested (endor Homebrew rustc 1.99 vs spark1 rustup 1.90.0 differ).
* Changed default behaviour: none for kernels or numerics; only the golden-dir fallback lookup changed.

Coordinator check: real `deploy.sh` build-only runs for CUDA (spark1 #3307) and HIP (tatooine #3308) were queued but did not get a slot within 30 min (spark1 sixth in line, tatooine behind an exclusive window), so the deploy.sh/deploy-step.sh changes were reverted. Cancelled #3304, #3307, #3308.
