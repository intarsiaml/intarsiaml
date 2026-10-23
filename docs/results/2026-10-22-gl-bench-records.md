# Lane gl: benchmark records from `intarsia bench` and the composer gap list

Date 2026-10-22 (EDT). CPU-only lane: no GPU job ran, no fleet box other than endor (cargo as a `--load cpu` fleetq slot) was touched.

## What now works

1. `intarsia bench speed --emit-record [PATH]` appends benchmark-record v1 JSON lines after the sweep: one per measured prompt length (it carries prefill and decode tok/s, so both phases of that length) and one per concurrency row. Default PATH `/opt/code/model-benchmarks/data/records/intarsia-bench.jsonl`. Code: `crates/intarsia-bench/src/record.rs` (pure, unit-tested), flags in `crates/intarsia-cli/src/cmd/bench.rs`. Documented in `docs/benchmarking.md` ("Benchmark records"). It reads no `INTARSIA_*` variable, only `FLEETQ_JOB`, so `docs/tuning.md` needs no row (`docs_env_vars` passes).
2. `tools/composer/gap_to_lane.py` writes `docs/composer/gaps.md` (and JSON with `--json`) from model-benchmarks `regimes.json`, `catalog.json` and `extraction-2026-10-22.json`.

## Checks (slot type: shared `--load cpu` lease, fleetq #4346 on endor; no speed claim rests on them)

| Check | Result |
|---|---|
| `cargo test -p intarsia-bench` | 24 passed (12 new in `record::tests`) |
| `cargo test -p intarsia-cli --test docs_env_vars` | 2 passed |
| `cargo check --workspace --locked` | ok |
| `python3 -m unittest discover -s tools/composer -p 'test_*.py'` | 13 passed |
| Real CPU run: `intarsia serve --backend cpu` on Qwen2.5-0.5B (BF16), `bench speed --sizes 128,256 --gen 8 --reps 2 --concurrency 1,2 --emit-record` | 3 records (p128, p256, c2); `uv run --with jsonschema python3 /opt/code/model-benchmarks/schema/validate.py <file>` exit 0, 0 schema errors. Raw: `docs/results/2026-10-22-gl-bench-records/raw/` |

The smoke records are `incomplete` on purpose: Qwen2.5-0.5B is not a family the site names, so `model.family` is `"Unknown"` (the rule from `schema/records.py:29`). They are not in model-benchmarks `data/records/`. The CPU tok/s in them (decode 3 to 20 tok/s depending on what else ran on endor) is not a result.

## Design decisions (LOG.md rows, user-approved 22:29)

* One record per measured cell, not per run, as briefed. `--record-run-id ID` (reusing a localeval run id, per the 22:50 coordinator note) writes the whole run as ONE record, because `validate.py` fails a run_id repeated within a file.
* Lane is `plain` unless `--spec X` (or `--spec X` inside `--serve-cmd`), then `spec:X`. The brief said main is always plain, but main already has `serve --spec ngram`.
* Trust is `measured-here` only with `FLEETQ_JOB`; otherwise `external` (the schema has no unslotted-local tier). `exclusive` is null.
* The bench is a client. Server facts come from `/v1/models` and `/metrics` `intarsia_info` (kv dtype, backend, max_model_len, served path); the run is refused up front unless the server reports itself as IntarsiaML. Engine version and commit are the bench binary's own (commit from `git rev-parse HEAD` of the cwd, `+dirty` when tracked files differ) unless `--engine-version/--engine-commit` are given.
* Hardware, family and quant use the site's strings. Python is ported by literal substring, in the site's order, for the models IntarsiaML serves (`build_data.py:33-62`, `:67-101`, `:106-121`); everything else needs a flag or stays `"Unknown"`.
* `finish()` is ported from `schema/records.py:13-33`. Launch commands and the bench command line are scrubbed with the `SECRET_RE` rule (`build_data.py:29`).

## Composer gaps (live data, rebuilt 23:12 EDT; `gaps.md`)

Same-quant margins are recomputed from `contenders[]` against IntarsiaML's own quant (the cell's `same_quant_*` is relative to the winner's quant: on tatooine it compares llama.cpp MXFP4). Score = margin % x phase weight (decode 1.0, prefill 1.0, concurrent 0.5; a judgement).

| # | Cell (DGX Spark GB10, Qwen3.6-35B-A3B, plain) | Rival (NVFP4) | Margin | Noise | Catalog technique |
|---|---|---|---|---|---|
| 1 | concurrent 16x | vLLM | +40.7 % | 10 % | `sglang-continuous-batching-uncapped-running-requests` (cell winner SGLang; low confidence) |
| 2 | concurrent 4x | vLLM | +40.3 % | 10 % | same |
| 3 | prefill 4k | vLLM | +16.2 % | 10 % | `vllm-gb10-w8a8-fp8-projection-gemm` |
| 4 | prefill 16k | vLLM | +15.9 % | 5 % | `vllm-gb10-w8a8-fp8-projection-gemm` |

Only four same-quant gaps exist. IntarsiaML's quality on that setup is `drop:gsm8k@n1319:0.240<0.281`, so every brief says to settle quality before chasing speed. Also found: 5 ties it could win, 6 cells where the rival is ahead only on a different quant, 13 cells with an empty IntarsiaML quant, 36 measure-first cells. Skipped: R9700 Qwen3.6-35B plain decode 16k (run 20261022-074221-adhoc-tc-off-r1, did not reproduce). The concurrent gaps on CUDA are probably structural: `serve --max-batch` is Metal only.

## Not done

* No `techniques.jsonl`: this lane measured no technique.
* The heuristic file list in each brief is a keyword match, not an analysis.
* No Metal, CUDA or HIP build was run (the change is backend-agnostic client code; CI covers those checks).
