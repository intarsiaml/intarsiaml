# CUDA no-op check of `perf/hip-norm-gemv` (2026-10-17)

**Verdict: PASS.** On spark1 (GB10, sm_121) the CUDA build of `perf/hip-norm-gemv` is bit-for-bit the same as the build of
`origin/main` (`c00af21`) in everything that was compared; decode speed is the same within about 0.3%.

* **A** = `origin/main` `c00af21` (a `git archive` export, tree `~/intarsia-decfuse/ref`).
* **B** = `perf/hip-norm-gemv` at `3ad530e` plus two script commits of this branch (tree `~/intarsia-decfuse/hipnorm`; stamp
  `head=bab3af1 code-tree=c350fc280236 dirty=no`; the extra commits only add `scripts/cuda-decode/{w-noop,build}.sh`).
* What B changes for CUDA: no `.cu` file, no `intarsia-backend-cuda` code. Only the model layer (the block takes the preceding
  norm as a `NormFold`, tries `RmsNormAddMatmulPair` / `GatedRmsnormMatmulFp8` with `try_fused`, and runs the norm itself when
  the registry has no kernel) and the core op tables (two new `Op`s, `fused_ref`, admission cases). `git diff c00af21 3ad530e
  -- crates/intarsia-backend-cuda kernels/cuda` is one change in the table test of `kernels.rs` (the two new ops may be missing from the CUDA table). CUDA registers neither op,
  so every call falls back to the old chain `rmsnorm_add` -> `matmul_pair` / `gated_rmsnorm` -> `matmul_fp8`. The point of the check
  is that this fallback is the same chain in the same order with the same buffers.

## Results

| check | A `c00af21` | B `3ad530e` | verdict |
|---|---|---|---|
| 35B (Qwen3.6-35B-A3B-NVFP4) 256 greedy ids, graph | `raw/gen35b-A-graph.ids` | identical | PASS |
| 35B, `--no-graphs` | identical to A graph | identical to A graph | PASS |
| 8B (Qwen3-8B) 256 greedy ids, graph and `--no-graphs` | identical (4 runs, one id list) | identical | PASS |
| `accuracy` 35b-short vs vLLM golden (sha `b6b51e1b1699c66c`) | prefill top-1 97.99%, KL 6.271e-3; decode 98.12%, KL 1.556e-2; PASS | the same numbers; PASS | PASS |
| self-golden dumps, top-64 log-probs of every prefill and decode row (`cmp_dump.py`) | 60 tensors | **0 of 60 differ**; per-position TSV identical | PASS |
| 8B fp32 goldens, `--gate-class bf16` | not run on A | 8b-short: prefill 100% KL 2.897e-4, decode 100% KL 3.238e-4, PASS; 8b-long: 99.65% KL 4.004e-4, 100% KL 3.332e-4, PASS | PASS |
| 35B decode, `bench speed` gen 128, 5 reps, `--require-graphs`, 2 interleaved rounds | see below | see below | within noise |

Graph runs: `steps=257 graph replays=256 (captures 1)` (35B), `steps=256 graph replays=255` (8B), both for A and B; the
`--no-graphs` runs report `graph replays=0` and 256 eager steps. All eight id lists of a model are byte-identical
(`raw/idproof.txt`), 255 separators = 256 ids each. The 35B 35b-short numbers are also the ones recorded on 2026-10-14
(`2026-10-14-cuda-decode-fusion.md`: 97.99% / 6.271e-3, 98.12% / 1.556e-2).

Decode tok/s, mean of 5 reps per cell, rounds A-B then B-A (`raw/ab/`; all four runs: `eager_steps 0`, replays 1280 for the
two sizes of the measured reps, no capture failure):

| prompt tokens | A round 1 | B round 1 | B round 2 | A round 2 | A mean | B mean | B / A |
|---|---|---|---|---|---|---|---|
| 128 | 80.6 | 80.4 | 80.4 | 80.7 | 80.65 | 80.40 | -0.3% |
| 1024 | 79.9 | 79.8 | 79.8 | 80.0 | 79.95 | 79.80 | -0.2% |

B is lower by 0.15 to 0.25 tok/s in both rounds and both sizes. The rep-to-rep standard deviation inside a run is about 0.1
tok/s (2026-10-14 page), so two rounds do not settle whether 0.2% is a real cost of the model-layer change (the model layer runs only at graph capture
and in eager mode; a replay does not execute it, and in the graph both builds launch the same kernels, so I have no mechanism for a real
cost) or noise; it is far below the 5% band
this repo treats as noise. Decode 4096 and prefill were not part of this check (prompt sizes 128 and 1024 only; prefill figures in
the logs: 1317 / 3933 for A, 1291 / 3930 for B in round 1 and 1309 / 3937 vs 1288 / 3920 in round 2, one prompt-length draw per
cell, not a comparison).

## Not run

* `intarsia qualify --quick-only` (not asked for in this check; Part B runs it on the 35B).
* A on the 8B fp32 goldens (the B figures equal the 2026-10-14 ones: 2.90e-4 / 3.24e-4 and 4.00e-4 / 3.33e-4).
* CUDA backend crate tests: its sources and kernels are unchanged (see above).
* Mac gates (`cargo test --workspace`, `cargo check`) were not re-run here: the branch's own page records them green.

## Method

* Builds: `scripts/cuda-decode/build.sh ref|hipnorm` under `fleetq run --host spark1 --mem 16` (jobs #2332, #2333, 4 cargo jobs each,
  vLLM up). `build.sh` gained `SRC_DIR` / `INTARSIA_STAMP` so that it can build an exported commit.
* Window: `scripts/cuda-decode/window.sh 15` (exclusive lease #2335 `--takes-down guidance`, `docker stop missioncntrl-guidance`,
  trap restore, `/v1/models` wait at 100.120.88.104:8000, every ssh prefixed with `FLEETQ_JOB`), running
  `scripts/cuda-decode/w-noop.sh ref hipnorm`. It runs two lanes in parallel (GPU sharing does not change a bit; no timing is taken from
  them): lane 1 the 35b-short accuracy on A and B with dumps, lane 2 the eight 256-id runs, then the two 8B goldens on B; then,
  alone on the GPU, the two A/B rounds. 256 ids: `generate --chat --no-think --ignore-eos --show-ids -p 'Write a long story about a
  robot who learns to paint.' --max-tokens 256` (35B `--mem-budget 29`, 8B 25, `--max-seq 4096`).
* Logs and dumps' comparison: `docs/results/2026-10-17-cuda-noop-norm-gemv/raw/` (the `.safetensors` dumps stay on spark1,
  `~/intarsia-decfuse/out/noop-hipnorm/`).

## Downtime

vLLM stopped 13:37:18, answered `/v1/models` again 13:52:11 (log: `2026-10-17-cuda-noop-norm-gemv/downtime.log`), 14.8 min, lease
#2335 released. No `--force`, no `--no-queue`, no lease held while coding.
