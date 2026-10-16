# Metal: are 2048-schedule goldens also valid for chunk-256 serving prefill?

Branch `metal-chunk-golden` (off `spike/metal` 28e5c5a, which contains the merged 1-row-tail fix, `2026-10-15-metal-chunk-no1.md`), Qwen3-8B BF16, Metal, endor. No code in `crates/` changed. Raw: `docs/results/raw/2026-10-15-metal-chunk-golden/`. fleetq: #2106 (harness), #2113 (CLI generate + accuracy; first attempt of those steps ran with a binary built without `--features metal`, all exit 1, kept in `failed-run1-nometal/`), both exclusive on endor, released.

Question (continuous-batching-design 8.1): the scheduler prefills at 256 under load, the goldens were recorded at 2048. Is Metal prefill bitwise chunk-invariant on the serving path (`Seq::prefill_step`), so the 2048 goldens cover it?

## Answer

**Yes. On the serving path, prefill at `--prefill-chunk` 256 and 2048 is bitwise identical (greedy ids and all 32 sampled-step logits rows, plus the prefill-final row) for all 33 prompts tested. The existing 2048 goldens cover 256-schedule serving prefill; no new baselines are needed for this reason.** Only 3 of the 33 prompts exceed one chunk at 2048 sizes in a way that tests a 256-vs-2048 difference at the boundary (L > 2048: 2049, 2050, 4064 x2, 4095-4098, 4200, 4557), and the L%256 cases (255, 256, 257, 258, 511-514, 2047-2050, 4095-4098) all pass.

## Method

* Serving path: a standalone harness (`harness/`, not a workspace member, uses `intarsia-cli` as a library) loads the engine at a given chunk and calls `Engine::generate` (greedy, 32 tokens, `ignore_eos`) with `GenRequest.trace_logits` on. `Seq::prefill_step` does the prefill. Every sampling-step logits row (the first one is the prefill-final row, then 31 decode rows) is hashed over its f32 bits (FNV-1a 64); ids compared too. The prefix cache is flushed with an unrelated prompt before each prompt; `cached_tokens` was 0 on all 66 runs. Spec decoding is off by construction with `trace_logits`.
* `Model::forward` path (what accuracy tooling and `--show-logits` use, via `forward_logits_each`): the same prompts, hashing the last row and all L rows.
* Prompts: token prefixes of the repo corpora (`corpus_docs.txt` for even L, `corpus_code.txt` for odd L) for L in 2, 3, 255, 256, 257, 258, 511, 512, 513, 514, 1000, 2047, 2048, 2049, 2050, 4095, 4096, 4097, 4098, 4200, plus the 13 golden prompts (8 short, 2 long, 3 from 4k).
* `--show-logits` was not used for comparison: it goes through `Model::forward` and prints only the top N values at 4 decimals.

## Serving path (Seq::prefill_step) and Model::forward, 256 vs 2048

Full table: `serving/compare.md` (`compare.py`). Serving-path result for every row: ids same, 32/32 logits rows bitwise identical. Summary of the rows that matter:

| prompt | L | L%256 | L%2048 | serving ids | serving logits (bitwise) | Model::forward (last / all rows) |
|---|---|---|---|---|---|---|
| docs2, code3 | 2, 3 | 2, 3 | 2, 3 | same | identical | identical |
| code255 | 255 | 255 | 255 | same | identical | identical |
| docs256 | 256 | 0 | 256 | same | identical | identical |
| code257 | 257 | 1 | 257 | same | identical | **DIFF / DIFF** |
| docs258 | 258 | 2 | 258 | same | identical | identical |
| code511 / docs512 | 511 / 512 | 255 / 0 | | same | identical | identical |
| code513 | 513 | 1 | 513 | same | identical | **DIFF / DIFF** |
| docs514 | 514 | 2 | 514 | same | identical | identical |
| docs1000 | 1000 | 232 | 1000 | same | identical | identical |
| code2047 | 2047 | 255 | 2047 | same | identical | identical |
| docs2048 | 2048 | 0 | 0 | same | identical | identical |
| **code2049** | 2049 | 1 | 1 | same | identical | identical |
| docs2050 | 2050 | 2 | 2 | same | identical | identical |
| code4095 / docs4096 | 4095 / 4096 | 255 / 0 | 2047 / 0 | same | identical | identical |
| code4097 | 4097 | 1 | 1 | same | identical | identical |
| docs4098 | 4098 | 2 | 2 | same | identical | identical |
| docs4200 | 4200 | 104 | 104 | same | identical | identical |
| g4k_0, g4k_1, g4k_2 (golden 4k) | 4064, 4064, 4557 | 224, 224, 205 | 2016, 2016, 461 | same | identical | identical |
| glong_8, glong_9 (golden long) | 300, 700 | 44, 188 | | same | identical | identical |
| gshort_0..7 (golden short) | 18..90 | | | same | identical | identical |

Why `Model::forward` differs only at 257 and 513: it still splits with `ids.chunks(max_tokens)`, so at chunk 256 and L%256==1 it computes a (C,1) tail whose 1-row chunk runs decode kernels, which are not bitwise equal to the multi-row kernels. At 2048 those lengths are one chunk. L=2049 and 4097 (L%2048==1) show no difference here because at chunk 256 the Model::forward split is (256 x 8, 1) and at 2048 it is (2048, 1): both have a 1-row tail, and the tail row is the same computation. The fix for this path is merged separately as `spike/metal` 32a42d8 (`Model::forward` cuts by the engine's prefill rule); this branch and its binary (28e5c5a) do not contain it and I did not rerun for it. It does not affect the serving path.

## intarsia generate (CLI, serving path), 48 tokens, greedy, `--ignore-eos --show-ids`

`generate/`. Prompts from `2026-10-15-metal-chunk-no1/prompts`: L=257 (s1), 2049 (s1, s2), 4097 (s1, s2). Chunk 256 vs 2048: ids identical and stdout text byte-identical in all 5 (exit 0 in all 10 runs).

## intarsia accuracy 256 vs 2048

`accuracy/`; same command lines as the chunk-no1 gates (`--spec off`, `--mode both/all`, decode positions 16/64/64), with `--prefill-chunk` honoured (log line 1 shows it). Confirms the P3 envelope (`2026-10-15-metal-p3-final.md`); not extended.

| Golden | Prefill top-1 / KL | Decode top-1 / KL | 256 vs 2048 numeric rows |
|---|---|---|---|
| short | 100.00% / 2.832e-4 | 100.00% / 3.313e-4 | identical, 16 per-prompt rows |
| long | 99.72% / 3.699e-4 | 99.80% / 3.538e-4 | identical, 20 rows |
| 4k (21-way) | 99.28% / 2.274e-4 | 98.96% / 2.548e-4 | identical, 6 rows |

The summary and per-prompt rows are identical to the printed digit (`diff` only differs in the chunk label and in the prefix-reuse section, which skips or runs different splits depending on the chunk). `golden gate` is PASS everywhere except long at 2048, which exits 1 only on `prefix-reuse verdict: FAIL` because no long-golden prompt (max 700) reaches a 2048 split ("reuse gate: SKIPPED"), the same artifact noted in the chunk-no1 doc; its floor verdict is PASS.

Golden prompts and the Model::forward tail: none hits L%256==1 (short 18..90, long 300/700, 4k 4064/4064/4557; the prefill parts of the teacher-forced runs, L minus decode positions, are not 1 mod 256 either). So accuracy numbers are unaffected by the (C,1) tail on this binary, consistent with the identical numbers above.

## Not done / limits

* Bitwise logits on the serving path were compared for the prefill-final row and 31 following decode rows per prompt (32 sampling steps); full-prompt-position logits are only available on the Model::forward path.
* One model (Qwen3-8B BF16), one machine; two corpora for the synthetic prompts; no hybrid/recurrent model (those use the plain `C` schedule anyway).
* Concurrent/batched prefill (cb) is not covered; this is the solo serving path.
