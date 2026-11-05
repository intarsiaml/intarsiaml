# DFlash by default for the dense Qwen3-8B (lane `lane-d8/dflash-8b-default`), 2026-11-05

**Stopped 2026-11-05 18:00 by user scope: fleet models only.** No fleet box serves the 8B. No speed number, greedy-ids gate, fallback run, CUDA `plan show` or Metal no-op was run: the jobs for them (#8588 gates, #8589 and #8590 ABAB, #8597 Metal no-op) were queued and cancelled before they started. **There is no measurement behind the default; do not merge the recipe key as a default.** To keep the default off, delete `dflash_default_draft` from `[tuning.cuda]` in `recipes/qwen3.toml` (the mechanism, tests and scripts are harmless without it).

Authority copied from the brief: on 2026-11-05 at 17:06 EST the user said to ignore what they deferred previously and to build the largest performance wins. Superseded at 18:00 EST (fleet models only).

## Findings that change the brief (from the code, not measured)

* **Metal does not run the 8B's DFlash.** `load.rs` allows a DFlash v1 draft on `cpu` and `cuda` only (v2: `cpu`, `hip`); nothing under `crates/intarsia-backend-metal` mentions DFlash. Endor has no speculation arm. Not ported here (a port, not a gate).
* **The HIP `--spec auto` gate does not port.** It needs an MTP head (`--spec auto` refuses to load without one) and uses MTP's tokens per pass as its covariate; the dense Qwen3-8B has no `[mtp]`. Its choice is DFlash versus plain decode. Existing evidence (docs/results/2026-10-25-dc-dflash-dcoco.md sections 10-12, LOG 2026-10-26 09:40): plain `--spec dflash` with the back-off `SpecGate` measured 2.12x pooled on CUDA, recall 1.86x, code 1.65x, prose 0.98x (6 of 8 prompts at 0.92-0.96x), short prompts only; the `[dflash]` cost gate lifts prose to 1.00x but costs recall/code 11-12%, so it is off. Chat and 4k prompts were never measured.
* **The draft is not loaded automatically.** `--spec dflash` needs `--spec-draft DIR` (the `TD-Labs/Qwen3-8B-DCOCO` checkpoint, 4.6 GB; on spark1 in the hub cache). A default is only safe with a silent fallback, which is what was built.

## What was built (tested on the Mac only)

* `[tuning.<backend>] dflash_default_draft = "Org/Name"` (`crates/intarsia-core/src/recipe.rs`), set for CUDA in `recipes/qwen3.toml`.
* `crates/intarsia-models/src/load.rs`: `dflash_default` (applies with no `--spec`, one local rank, no `--max-batch` above 1, no prefix store, `INTARSIA_SPEC_DEFAULT` not `0`), `find_dflash_draft` (`INTARSIA_DFLASH_DRAFT=DIR`, else the hub cache; never downloads), `dflash_draft_fits` (hidden size, heads, head_dim, vocabulary, tap layers, v1 only), `apply_dflash_default`. The decision is made before `SpecMode::Dflash` is set, so `--max-batch 4` and other backends never reach the hard "dflash with the batching engine" / "cpu and cuda only" errors. A missing, unreadable, mismatching or DFlash2 draft means plain decode with one stderr line. `--spec off` and `INTARSIA_SPEC_DEFAULT=0` opt out; a lone `--spec-k N` sets the draft length.
* Sampled requests: the engine sets `dflash.active = drafting` per request (`engine.rs`, as MTP), so a non-greedy request does not draft; not exercised on a GPU (`scripts/d8/gates.sh fallback` has the check, unrun).
* Docs: `docs/recipes.md` key row, `docs/tuning.md` feature row, `INTARSIA_DFLASH_DRAFT` row, `INTARSIA_SPEC_DEFAULT` row. No LOG.md row was written: no decision was reached.

## Gates

| gate | result |
|---|---|
| New tests `crates/intarsia-models/tests/dflash_default.rs` (6: gating conditions, hub lookup, fit check, shipped pair is exactly qwen3 on CUDA, loader falls back with a missing or mismatching draft with ids equal to off) | pass (endor, fleetq #8565) |
| `cargo test --workspace --no-fail-fast` (endor, fleetq #8584, `raw/ws-test.log`) | 231 test binaries, 1328 passed, 0 failed, including `docs_env_vars` |
| Greedy ids default vs `--spec off` on the 8B, `--max-batch 4` and missing-draft fallback on a server, CUDA `plan show` vs main, Metal no-op (`plan show`, 256 ids, `kernels test`) | **not run** (cancelled) |
| Speed by category (code, math, prose, chat) at 1k and 4k, ABAB on spark1 | **not run** |
| HIP | not run (tatooine busy, as briefed) |

## Left for whoever resumes

`scripts/d8/make_cats.py` (category prompts, `scripts/d8/cats/`), `scripts/d8/abab.sh` (two ABAB windows through localeval), `scripts/d8/gates.sh`, `scripts/d8/spark1.sh`, `scripts/d8/metal_noop.sh`, `tools/d8/table.py` (none ran end to end except the builds). The bar was prose and chat at or above 0.97x of plain at 1k and 4k; the 4k prose/chat cell is the one most likely to miss, because verify attention runs one decode call per row.

spark1 files left (not removed, no slot held): `~/intarsiaml-d8-src`, `-base`, `-target`, `-target-base` (both release CLIs built, stamps `d8-dev-1105174039`, `d8-base-1105174424`), `-out` (empty).
