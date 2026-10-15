# CUDA and HIP check of tonight's Metal work (2026-10-15)

**Outcome.** Both targets build and pass their tests on CUDA (spark1, GB10) and HIP (tatooine, R9700). Nothing failed, so
no bisect against `a296a78` was needed. With `--spec` off (the default), `intarsia plan show` on CUDA is byte-identical
between A and B, and `intarsia kernels test --backend cuda` gives the same 41 kernels / 582 cases, all ADMITTED, on both.
Limits: the HIP GPU tests and `kernels test --backend hip` were not run (radiance holds both cards), and the CUDA plan
check used a small synthetic model, not the real 8B. Details in section 3.

* **A** = `spike/metal` at `c556008`.
* **B** = `metal-spec-ngram` at `44c4a91041d8479f5ec6d63e43b24645d482f426` (committed tip when the run started).
  The `/opt/code/intarsiaml-spec` worktree had uncommitted edits to `intarsia-core/src/{engine,spec}.rs` (the G2/G3 work in
  progress). They are not in B as tested: trees were exported with `git archive` of the commit.

## 0. What changed, from the diffs

* A vs `a296a78` (`git diff a296a78 c556008 --stat -- crates/`): 22 files, all in `intarsia-backend-metal` (build.rs,
  examples, src, tests) except four in `intarsia-cli`: `cmd/qualify.rs`, `qualify/{facts,report,state}.rs` (the `acc-4k`
  stage and 4k golden). No change in `intarsia-core`, `intarsia-models`, `intarsia-backend-{cpu,cuda,hip}` or `kernels/`.
  The only shared code that CUDA and HIP compile and A changes is therefore the CLI qualify code.
* B vs A (`git diff c556008 44c4a91 -- crates`, non-Metal files): `intarsia-core` `engine.rs`, `lib.rs`, `model.rs`,
  `ops.rs` (`OpCall.exact_rows`), `plan.rs` (`Phase::Verify`, verify probes only when `model.verify_rows()` > 1),
  `registry.rs` (`KernelImpl.rows_exact`, `select` skips `rows_exact` kernels for ordinary calls), `spec.rs`,
  `telemetry.rs`; `intarsia-models/src/load.rs` and `tests/spec.rs`; `intarsia-server/src/metrics.rs`; `intarsia-cli`
  `common.rs` (`--spec`, `--spec-k`) and `cmd/generate.rs`. Kernel files: only `kernels/metal/*`. The CUDA and HIP
  backend crates are untouched.

## 1. Results

Build = `cargo build --release -p intarsia-cli --features cuda|hip`. Tests = `cargo test --release -p <crate>`;
numbers are summed over every test binary and doctest of the crate (passed / failed / ignored).

### CUDA (spark1, GB10 sm_121, vLLM left up)

| check | A `c556008` | B `44c4a91` |
|---|---|---|
| build | pass (44.7 s) | pass (44.8 s) |
| test `intarsia-core` | 50 / 0 / 1 | 57 / 0 / 1 |
| test `intarsia-models` | 68 / 0 / 0 | 72 / 0 / 0 |
| test `intarsia-backend-cuda` (`INTARSIA_REQUIRE_GPU=1`) | 32 / 0 / 4 | 32 / 0 / 4 |
| test `intarsia-cli` (`--features cuda`) | 76 / 0 / 0 | 76 / 0 / 0 |
| `kernels test --backend cuda` (8B shapes) | 41 kernels, 582 cases, 0 fail, all ADMITTED | same; output identical to A except the wall-time line |
| `plan show`, spec off | md5 `05d09266...` | md5 `05d09266...` (identical) |
| `plan show --json`, spec off | md5 `26cea075...` | md5 `26cea075...` (identical) |
| `plan show --spec ngram` | n/a | refuses: "--spec Ngram needs the Metal backend ... (backend cuda, tp 1)" |

B adds 7 core tests and 4 models tests (`tests/spec.rs`: identity greedy, sampled fallback, stop inside accepted runs,
min_tokens and EOS inside runs); all pass. The `intarsia-backend-cuda` result, 32 passed / 4 ignored, equals the 2026-10-14 check at `369cf88`
([core-changes check](2026-10-14-core-changes-cuda-hip-check.md)).

### HIP (tatooine, R9700 gfx1201, radiance up, GPUs hidden for tests)

| check | A `c556008` | B `44c4a91` |
|---|---|---|
| build | pass (41.4 s) | pass (41.9 s) |
| `intarsia-backend-hip` test binaries compile (`--no-run`) | pass | pass |
| test `intarsia-core` | 50 / 0 / 1 | 57 / 0 / 1 |
| test `intarsia-models` | 68 / 0 / 0 | 72 / 0 / 0 |
| test `intarsia-cli` (`--features hip`) | 76 / 0 / 0 | 76 / 0 / 0 |
| test `intarsia-backend-hip`, no device visible | 35 / 0 / 6 (see note) | 35 / 0 / 6 (see note) |
| `kernels test --backend hip` | not run: needs radiance down | not run: needs radiance down |
| `plan show` (HIP) | not run: needs radiance down | not run: needs radiance down |
| HIP GPU tests | not run: needs radiance down | not run: needs radiance down |

Note on the HIP crate row: those tests ran with `HIP_VISIBLE_DEVICES=-1`, so every test that needs a card returns early
and the binaries finish in 0.01 s. The 35 passes are not evidence about GPU behaviour. The same run on `a296a78` also gave
35 / 0 / 6, so the 35 (against 37 with the cards visible on 2026-10-14) is a property of running without a device, not of
tonight's change.

Build warning (both trees, not from tonight): `function ik_rt_event_destroy is never used` in `intarsia-backend-hip`.

## 2. Failures

None. No bisect, no comparison run on `a296a78` for CUDA. The only `a296a78` run was the HIP crate tests with no device,
to explain the 35 versus 37 count (see the HIP note). Its log was removed with the temp tree before it was copied; only the
count (35 passed, 6 ignored, exit 0) was read from it.

## 3. Not run, and why

* **HIP GPU tests, `kernels test --backend hip`, HIP `plan show`.** tatooine had 0.1 GB of GPU memory free beside radiance
  (fleetq: 1 GB declared, 0.1 reported), which is less than a HIP context needs. Running them takes radiance down, which
  this task did not allow. The HIP backend crate and kernels are byte-identical in A and B to what ran on `a296a78`
  (no diff in `crates/intarsia-backend-hip` or `kernels/` apart from `kernels/metal`), so what is untested on HIP is the
  core change running against a real card. The core change is exercised on CUDA instead, and its plan on CPU-side tests.
* **`plan show` on the real Qwen3-8B.** Loading 16 GB on spark1 beside vLLM is over the 8 GB limit for this task. The plan
  check used a synthetic model instead: Qwen3-8B's `config.json` and tokenizer with `num_hidden_layers` 2 and
  `vocab_size` 2048, constant bf16 weights (0.81 GB; generator `raw/2026-10-15-cuda-hip-noop/scripts/mk.py`). It keeps the
  8B tensor shapes (hidden 4096, head_dim 128, ffn 12288, 32/8 heads) so kernel selection is the same as for the 8B, but
  the layer count is 2. The plan covers matmul, matmul_pair, attn_prefill, attn_decode, logits, norms and kv_append. It
  was run twice per tree (A, B, A, B): all four text outputs and all four JSON outputs are identical, so the spec-off
  plan is stable and unchanged.
* **Greedy-id and accuracy identity on CUDA.** Needs the real 8B (`intarsia accuracy`, `generate`), so not run. A greedy
  run on the synthetic model gave 128 ids of `0` on both trees (md5 equal, but the model is degenerate), which proves
  nothing, so it is not counted.
* **`--spec ngram` on CUDA or HIP.** B is Metal-only for spec; CUDA refuses it with the message above. HIP not run.
* **qualify pipeline, 35B model, tensor-parallel, fp8/nvfp4 end-to-end.** Out of scope for a no-op check.

## 4. Method and commands

Same method as the [2026-10-14 check](2026-10-14-core-changes-cuda-hip-check.md): export the commit, copy to the box,
build with its own target dir, run cargo tests, run the CLI checks. Differences: `git archive` tarballs of `c556008`
and `44c4a91` instead of a bundle, extracted to `/tmp/cnc/{A,B}` on each box; `a296a78` archived the same way for the one HIP run.

* spark1: `PATH=$HOME/.cargo/bin:/usr/local/cuda/bin`, `CUDA_PATH=/usr/local/cuda`, `CARGO_BUILD_JOBS=5`,
  `CARGO_TARGET_DIR=/tmp/cnc/<T>-target`. A and B ran at the same time. Per tree: `cargo build --release -p intarsia-cli
  --features cuda`; `cargo test --release -p intarsia-core`; `-p intarsia-models`; `INTARSIA_REQUIRE_GPU=1 cargo test
  --release -p intarsia-backend-cuda`; `cargo test --release -p intarsia-cli --features cuda`. Then
  `intarsia kernels test --backend cuda --model <synthetic>` and `intarsia plan show --model <synthetic> --backend cuda
  --max-seq 2048 --mem-budget 4 [--json]`. Peak GPU use of the test processes seen once: 214 to 220 MiB each.
* tatooine: `deploy/tatooine/dev.sh` of each tree (`INTARSIA_REPO=/tmp/cnc/<T>`, `INTARSIA_TARGET_VOL=cnc-target-<T>`),
  `--jobs 8`. Build, then `cargo test --release -p intarsia-backend-hip --no-run`, then the CPU-side tests with
  `HIP_VISIBLE_DEVICES=-1` so no process opened a card.
* Scripts: `raw/2026-10-15-cuda-hip-noop/scripts/{cuda-run.sh,hip-run.sh,plan.sh,mk.py}`. Logs:
  `raw/2026-10-15-cuda-hip-noop/{cuda,hip}/{A,B}/*.log`, plan outputs `cuda/plan-*.{txt,json,err}`,
  kernels reports `cuda/kernels-{A,B}.{out,err}`.

## 5. Versions and slots

* spark1: GB10 sm_121, CUDA 13.0.88 (nvcc), rustc/cargo 1.90.0.
* tatooine: image `intarsia-hip-dev:rocm7.2.4-rust1.90.0` (HIP 7.2.53211, ROCm 7.2.4), rustc 1.90.0.
* fleetq: **#2002** (`--host spark1 --mem 8`, non-exclusive) and **#2003** (`--host tatooine --mem 0`, no
  `--takes-down`). Both started immediately, both released. vLLM and radiance were never touched (radiance `active` at the
  end; vLLM process still holding its 59 GB).
* Cleanup: `/tmp/cnc` removed on both boxes; docker volumes `cnc-target-{A,B,BASE}` removed on tatooine.
