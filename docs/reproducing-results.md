# Reproducing results

Requirement 5 (reproducible): a number in `docs/results/` can be re-measured on another box from a git sha, a toolchain
stamp and one bundle. Three tools and four pins make that work.

| Tool | What it does |
|---|---|
| `scripts/deploy.sh <host>` | Syncs a pinned git ref to a box, builds it with the right features in the right toolchain, smoke-tests it, prints and stores a **reproducibility stamp**. |
| `scripts/bench-bundle.sh pack` | Packs the stamp, device fingerprint, env knobs, recipes, lockfile, command line and reports into one tarball. |
| `scripts/bench-bundle.sh replay` | Deploys the bundle's sha on another host, re-runs the recorded command, and prints which stamp fields match. |
| `.github/workflows/ci.yml` | `cargo check` / `cargo test` / Metal `cargo check`, all `--locked`, on GitHub-hosted runners only. |

Everything that touches a fleet box runs inside a fleetq slot (`/opt/code/fleetq/README.md`). `deploy.sh`, `pack` and
`replay` require `FLEETQ_JOB` and refuse to run without it (`deploy.sh --dry-run` needs none and changes nothing; `show`
is local).

## One-command deploy

```bash
# CUDA on spark1, vLLM stays up, takes no role down:
fleetq run --host spark1 --mem 20 --label "deploy main to spark1" -- scripts/deploy.sh spark1
# a specific commit, a second tree next to the first:
fleetq run --host spark1 --mem 20 --label "..." -- scripts/deploy.sh spark1 --ref 7536a5e6 --tree main-7536a5e6
# HIP on tatooine: the GPU smoke needs the radiance slot, so take it down for the window
fleetq run --host tatooine --takes-down radiance --mem 30 --label "..." -- scripts/deploy.sh tatooine --backend hip
# Metal on this Mac:
fleetq run --host endor --mem 8 --label "..." -- scripts/deploy.sh local --backend metal
# look first:
scripts/deploy.sh spark1 --dry-run
```

`--backend` defaults to what the host has (`nvcc`/`nvidia-smi` is CUDA, Darwin is Metal, `/dev/kfd` is HIP). `--ref`
defaults to `HEAD` and can be any ref; the script resolves it to a full sha and exports **committed content only** with
`git archive`, so an uncommitted edit never sneaks into a stamp (it prints a note when there is one, and a warning when the
sha is on no remote branch, because another host could not replay it).

The steps:

1. **Sync** the export to `~/intarsia-deploy/<tree>/` on the host (`<tree>` defaults to the backend name) with
   `rsync -rlpc --delete`; `target/` survives, so a second deploy of the same sha prints `0 entries changed` and builds
   nothing. Checksums without `-t`: `git archive` gives every file the commit's timestamp, and with `-t` an older ref
   would look older than the last build, so cargo would reuse stale objects.
2. **Build** with `cargo build --release --locked --features <backend> -p intarsia-cli` and `INTARSIA_BUILD_STAMP`
   (`head=<sha7> tree=<hash> backend=<be>`) baked in, so `intarsia qualify` and the accuracy reports carry it.
   CUDA and Metal use the host toolchain. HIP builds in the pinned `intarsia-hip-dev` container through
   `deploy/tatooine/dev.sh` (the script builds the image from `deploy/tatooine/Dockerfile.hip-dev` if the host lacks it).
3. **Smoke**: `intarsia doctor --json --out DEPLOY_DOCTOR.json`, then `intarsia kernels test --quick` (the whole quick
   sweep, about a second on spark1: 51 kernels, 149 cases against the cpu-ref oracle; 10 kernels have no case in the
   quick sweep and are listed as `SKIPPED`/untested, not failed). The deploy fails if `kernels test` fails.
   `--smoke none` skips it; `SMOKE_KERNELS_ARGS="--op matmul"` narrows it.
4. **Stamp**, printed and written to `<tree>/DEPLOY_STAMP.json`:

```
git        70ab09b3bd61e76c3f183be48a282c717e9a1185  (HEAD; tree ea2bfc94d84e)
backend    cuda
rustc      rustc 1.90.0 (1159e78c4 2025-09-14)
cargo.lock b64a06d84a56f46e  (built with --locked)
cuda       Cuda compilation tools, release 13.0, V13.0.88
gpu        NVIDIA GB10  compute 12.1
driver     580.173.02  (cuda toolkit 13.0.3)
binary     target/release/intarsia  sha256 3e6e68bc3cefd17e  37717152 bytes  v0.0.1
smoke      pass
```

(`docs/results/2026-10-21-m5-reproducibility.md` has the full run.) The JSON also carries the container image id and base
digest on HIP, the ROCm version, the host OS/CPU, and the `intarsia doctor` device fingerprint.

`--env-file <file>` (lines of `export INTARSIA_X=...`) is copied to the tree, sourced before the smoke and the command,
and stored in the stamp: the env knobs of a run are part of the run. `-- '<cmd>'` runs a command in the deployed
environment (inside the container on HIP) with `$INTARSIA_BIN` and `$INTARSIA_REPLAY_OUT` set:

```bash
fleetq run --host spark1 --mem 20 --label "..." -- scripts/deploy.sh spark1 --env-file knobs.sh -- \
  '$INTARSIA_BIN bench speed ... --out $INTARSIA_REPLAY_OUT/speed.json'
```

## Bench bundle

```bash
fleetq run --host spark1 --mem 0 --label "pack r7 bundle (reads files)" -- \
  scripts/bench-bundle.sh pack --host spark1 --backend cuda --out r7.tgz \
    --cmd '$INTARSIA_BIN bench speed ... --out $INTARSIA_REPLAY_OUT/speed.json' \
    --report spark1:intarsia-deploy/cuda/replay-out/<run>  --report docs/results/<lane>/localeval \
    --env INTARSIA_CUDA_FP8_GEMM_LT=1 --note "what this is"
scripts/bench-bundle.sh show r7.tgz
fleetq run --host spark2 --mem 20 --label "replay r7" -- scripts/bench-bundle.sh replay r7.tgz spark2
```

`pack` reads what the last `deploy.sh` run left in `<host>:~/intarsia-deploy/<tree>/`, so deploy (and run the
measurement) first. The tarball holds `MANIFEST.json` (git sha, backend, binary sha256, command, env knobs, report list,
the whole stamp), `stamp.json`, `doctor.json`, `env.sh`, `replay.cmd`, `Cargo.lock`, `rust-toolchain.toml`, `recipes/`,
`reports/` (bench-speed JSON, localeval output, logs; each `--report` is a local path or `host:path`, file or
directory) and `SHA256SUMS`. `--with-binary` adds the CUDA/Metal binary (HIP's lives in a docker volume).

`replay` verifies the checksums, deploys the bundle's sha on the target host with the same backend and env knobs,
runs `replay.cmd`, copies `$INTARSIA_REPLAY_OUT` back to `./replay-<sha7>-<time>/` and prints a table of
`match`/`DIFFER` rows: git sha, Cargo.lock, rustc, CUDA, ROCm, container, GPU, driver, binary hash. Expect `DIFFER` on
the GPU and driver rows when the second host is a different machine; that is the point of the table. `--no-deploy` skips
the build and runs the command in the tree already there.

Limits to know:

* **The binary hash is informational.** Two builds of the same sha in different tree paths gave different binary hashes on
  spark1 (3e6e68bc... vs 3f267d9b..., same size class): paths get embedded. The sha, Cargo.lock and toolchain rows pin the
  build. One attempt at `--remap-path-prefix` did not make the hashes equal and was removed; chasing it needs a look at what
  nvcc and `build.rs` embed.
* The bundle records the **command and env knobs you give it**. It cannot know which knobs a run used unless they went
  through `--env-file`/`--env`; if a run was started by hand with extra variables, pass them to `pack --env`.
* It does not capture the model weights or the tuning DB (there is none yet); the report should name the model path and
  revision. `ARCHITECTURE.md` section 11 sketches an `intarsia replay manifest.json` with plan and tuning hashes; this
  bundle is the script-level step toward it.

## What is pinned

| Pin | Where | Value now |
|---|---|---|
| rustc | `rust-toolchain.toml` (rustup reads it) | `1.90.0`, equal to `rust-version` in `Cargo.toml`, spark1's rustup toolchain and the HIP image. Bump all four in one PR. |
| Dependencies | `Cargo.lock`, committed; every deploy build and every CI job uses `--locked` | A stale lockfile fails the build instead of being rewritten. Python tools: `uv.lock`. |
| HIP image | `deploy/tatooine/Dockerfile.hip-dev`: `rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`, rust `1.90.0`, tagged `intarsia-hip-dev:rocm7.2.4-rust1.90.0`. The stamp records the local image id and base digest. | ROCm 7.2.4 |
| CUDA | **No container.** spark1 builds with the host toolchain: nvcc `13.0.88` (toolkit `13.0.3`), driver `580.173.02`, `sm_121`. Every stamp records them, so a drift shows up as a `DIFFER` row. | Host-pinned, not image-pinned |
| Metal | Xcode's Metal toolchain on the Mac; rustc is Homebrew `1.99.0`, which has no rustup and **ignores `rust-toolchain.toml`**. The stamp records the rustc actually used. | Recorded, not enforced |
| CI actions | `actions/checkout` and `Swatinem/rust-cache` pinned by commit sha in the workflow | |

Re-pin the HIP base: `docker pull rocm/dev-ubuntu-24.04:<tag> && docker image inspect --format '{{index .RepoDigests 0}}'
rocm/dev-ubuntu-24.04:<tag>`, edit the `FROM` line, and rebuild the image (a new image id gets a fresh cargo target
volume, see `deploy/tatooine/dev.sh`).

A CUDA container image pinned by digest would close the one gap in the table. It is not built yet: nobody has
chosen a base image whose nvcc/driver pair matches the GB10's. Until then the stamp is the evidence.

## CI

`.github/workflows/ci.yml`, on pull requests and pushes to `main`, GitHub-hosted runners only (it never runs on fleet
hardware and needs no GPU):

* `cargo check --workspace --locked` (ubuntu);
* `cargo test --workspace --locked --exclude intarsia-backend-metal` (ubuntu; the cuda and hip crates build their stub
  kernels without nvcc/hipcc, and their GPU tests skip when no device is present);
* `cargo check --locked -p intarsia-backend-metal` and `-p intarsia-cli --features metal` on the macOS runner.

fmt and clippy are not in CI: `cargo fmt --check` reports about 750 diffs on main today, and this PR does not mass-reformat.
Reformat in a PR of its own, then add the two jobs. The GPU gates (KL21, greedy ids, prefix reuse, goldens, mutants,
admission) remain fleet runs under fleetq; CI does not replace them.

## Replaying a published result

1. Find the sha and knobs in the results doc (or its bundle). 2. `scripts/bench-bundle.sh replay <bundle> <host>` inside a
fleetq slot sized for the run (`--exclusive` for speed numbers; the examples above use a small non-exclusive slot because
they only smoke-test). 3. Compare the printed table, then the numbers. Do not compare a vLLM-up run with a vLLM-down run.
