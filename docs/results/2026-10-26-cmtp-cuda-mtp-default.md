# Lane cmtp: `--spec mtp` as the default for the Qwen3.6-35B-A3B on CUDA (tp=1)

Date 2026-10-26 (EDT). Box spark1 (GB10), model `nvidia/Qwen3.6-35B-A3B-NVFP4`, single-sequence engine, tp=1. Request (coordinator, user-approved
2026-10-26 13:30): decide from measurements whether `--spec mtp` is on by default, ship it if every gate passes. Raw logs: `2026-10-26-cmtp-cuda-mtp-default/raw/`.
Scripts: `scripts/cmtp/`, `tools/cmtp/round_table.py`.

## 0. Result

**Update 2026-10-26 17:33 EDT (user, "do your recommendations"): the CUDA default is left OFF.** The mechanism, tests and docs are merged; `mtp_default_k` is commented out in `recipes/qwen3_5_moe.toml` until the macro prefill pass works with MTP (lane mpf). New rule, for this default and the HIP tp>1 one: a configured prefix store (`--prefix-store`, `INTARSIA_PREFIX_STORE`) skips the default with one stderr line. The measurements below were taken with the key set to 2.

**All five gates pass; the default is shipped as `[tuning.cuda] mtp_default_k = 2` in `recipes/qwen3_5_moe.toml`.** One cost the gates do not
cover is large and is listed first because it decides whether to merge: **prefill is slower with the default** (section 7: 4k -20 %, 16k -37 %, 33k -51 % prefill tok/s;
+0.15 s, +1.5 s, +6.0 s time to first token). Decode is faster in every category and at every length measured.

| gate | result |
| --- | --- |
| 1 ids, 18 prompts (6 short, 6 ~4k, 6 ~16k tokens) | 18/18 identical to `--spec off` with the shipped default (no flag) and 18/18 with explicit `--spec mtp --spec-k 2` on the earlier prompt set (section 2) |
| 2 soak, 40 greedy requests via the server | 40/40 identical, equal to spec-off (2 prompts x 20) (section 3) |
| 3 mutants | 3/3 CUDA-binary mutants caught by the ids gate (`rollback-skip` 5/5 prompts differ, `accept-off-by-one` 5/5, `cuda-perrow-off` 6/6 at ~16k), every mutant binary's sha256 differs from the clean build; CPU tests catch 4 of 5 shared-logic mutants and 3/3 default-mechanism mutants (section 4) |
| 4 speed, localeval, 3 interleaved rounds, 7 categories | every category >= 1.17x of spec-off for k=2 (median and worst round); k=2 chosen over k=3 (section 5) |
| 5 G0 | CUDA 8B and Gemma-4: ids and `plan show` byte-identical to origin/main; 35B `--spec off` and `INTARSIA_SPEC_DEFAULT=0`: ids and plan identical to main; 35B default ids identical to main's plain decode; Metal (endor) 8B and 35B ids and plan byte-identical; HIP: no HIP code or HIP recipe key changed, recipe-pair test (section 6) |

## 1. What changed

- `[tuning.<backend>] mtp_default_k` (new, `BackendTuning::mtp_default_k`, `Recipe::resolve_mtp_default`): the one-rank counterpart of PR #175's
  `mtp_default_tp_k`. Same mechanism (`EngineOptions::spec_auto`, the shared `mtp_default_allowed` gate): applies with no `--spec`, in-process ranks,
  no `--max-batch` above 1, `INTARSIA_SPEC_DEFAULT` not `0`; `world == 1` uses `mtp_default_k`, `world > 1` keeps `mtp_default_tp_k`. A checkpoint
  without `<mtp prefix>.fc.weight` falls back to plain decode with a note (an MoE head is allowed at one rank; under tp it is not sharded, as before).
  A lone `--spec-k N` (origin/main, PR #179) keeps the default and sets k, covered for the new key.
- `recipes/qwen3_5_moe.toml`: `[tuning.cuda] mtp_default_k = 2`. HIP and Metal have no key, so the 35B there is unchanged (recipe-pair test).
- Per-row verify attention on CUDA stays the MTP default (PR #158; loading the head turns it on), as the user decided.
- Tests (`crates/intarsia-models/tests/tp_mtp.rs`): the gating matrix of the new key, the loader on the cpu backend (ids equal off, spec ran, no-head checkpoint
  falls back), lone `--spec-k`, the recipe-pair test (exactly `qwen3_5_moe cuda tp1 2` and `qwen3_5_vl hip tp 2` default to MTP). The `INTARSIA_SPEC_DEFAULT` tests share a lock,
  and `model_dir()` writes the fixture once per process (a rewrite under a mapped file was a SIGBUS in 2 of 8 runs).
- Docs: `docs/recipes.md`, `docs/tuning.md` (`INTARSIA_SPEC_DEFAULT`), `docs/decisions/LOG.md` (row + "Defaults in force").

Gates 1 to 5 ran on a binary built from `b6f682fb6` (md5 `b3fa447331bd`), whose `crates/*/src`, `recipes/` and `kernels/` equal this branch before the merge of origin/main;
the merged head was re-checked in section 8.

## 2. Gate 1: ids, 18 prompts

256 greedy ids (`--ignore-eos`, `--temperature 0`, chat template, thinking off, `--profile-every 0`) per prompt, `generate`, max-seq 4096 / 8192 / 20480.

- `raw/ids-final.log`: `--spec off` vs no flag (the recipe default), prompts of `scripts/cmtp/ids-prompts` sized to the tokenizer (short 34-69 tokens; ~4k: 3.9-4.0k; ~16k: 15.5-16.3k).
  18/18 identical; each auto run's stderr has the `default of recipe qwen3_5_moe on cuda at tp=1` line and a `spec:` stats line (acceptance 50-94 %, 2.0-2.9 tokens per pass).
- `raw/ids-k2.log`: the same with explicit `--spec mtp --spec-k 2` on the first prompt set (`scripts/dfh/ids-prompts`, which came out at 2.6-3.4k and 10.4-12.5k tokens): 18/18.
- The first set was replaced because "16k" meant 11-12k tokens; no prompt exceeds the context (20480).

## 3. Gate 2: soak

`raw/soak/`: one server per arm (`serve --max-batch 1`, `--profile-every 0`), greedy chat, 256 forced tokens, two prompts alternating. The default server (no `--spec`; its log has the default line)
answered 20 + 20 requests: every answer of a prompt has the same sha256 (`2664ec19db205d54`, `973f90d9287d238e`), equal to the 2 + 2 answers of the `--spec off` server.

User path (`raw/userpath.log`): supervised `serve` with no flags but port and model: speculation ran (`intarsia_spec_steps_total` 143, 0 with `--spec off`); its answers to
three requests (a 3.9k-token prompt twice, a short one) equal `--spec off`'s, with and without `--prefix-store on`. Natural stops (no `--ignore-eos`) included.

## 4. Gate 3: mutants

| mutant | where | caught by | result |
| --- | --- | --- | --- |
| `rollback-skip` (recurrent state not restored after a partial accept) | `state.rs` | CPU `tests/mtp.rs` (11 tests); CUDA ids | ids differ on 5/5 prompts (short, 4k, 16k) |
| `accept-off-by-one` (first rejected draft counted as accepted) | `spec.rs` | CPU `tests/mtp.rs` (11 tests); CUDA ids | ids differ on 5/5 |
| `cuda-perrow-off` (CUDA verify attention back to the multi-row call) | `device.rs` | CUDA ids at ~16k (no CPU test can see it) | ids differ on 6/6 |
| `mtp-chain-row`, `mtp-token-pairing` (wrong hidden-state tap / token pairing; drafts only, ids cannot change) | `model/mtp.rs` | CPU `tests/mtp.rs` (drafts-equal-eager, reduced-draft-vocab) | caught |
| `mtp-kv-no-truncate` (drafter cache keeps rejected rows) | `state.rs` | nothing | **survived**: it can change drafts only, never ids; not an exactness bug, no test added |
| `default-optout-ignored`, `default-batch-ignored`, `default-world-any` | `load.rs` | CPU `tests/tp_mtp.rs` | caught (2, 2, 1 tests) |

CUDA binaries (`raw/mutants-cuda-build.log`, clean tree rebuilt in a copy, `target-mut`, never the measured binary): clean `e78662136764d986`, `rollback-skip` `751d4a484ec87f12`,
`accept-off-by-one` `9502e2b14552b074`, `cuda-perrow-off` `7f0c1715478c69f3`. CPU: `raw/mutants-mtp.log`, `raw/mutants-default.log` (sha256 of each test binary differs from the clean one).
A first build of the CUDA mutants reused the previous mutant's objects (rsync keeps old mtimes); `mutant.sh` now touches the sources and the three were rebuilt before any result above was taken.

## 5. Gate 4: speed, k choice

`scripts/cmtp/window.sh` (the 27B lane's method on spark1): one `serve --max-batch 1 --max-seq 8192` per arm, `localeval quality --judge none` over the 7 category files of `scripts/dfh/cats`
(held-out chat/code/prose/recall, rcp code/prose/recall; 36 prompts, 512 forced tokens each), 3 rounds, arm order alternating (off, mtp2, mtp3 / reversed / forward), one fleetq lease, box quiet.
Decode tok/s, tokens / sum of decode times per category and round (`raw/s1/round-table.txt`); ratio of the median to off's median, and the worst single round:

| category | off | mtp k=2 | x off | worst round | tokens/pass | mtp k=3 | x off | worst round | tokens/pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ho-chat | 83.9 | 110.9 | 1.32 | 1.313 | 2.35 | 107.2 | 1.28 | 1.269 | 2.68 |
| ho-code | 83.8 | 128.6 | 1.54 | 1.525 | 2.74 | 138.7 | 1.66 | 1.621 | 3.48 |
| ho-prose | 83.8 | 98.6 | 1.18 | 1.168 | 2.08 | 91.2 | 1.09 | 1.074 | 2.26 |
| ho-recall | 83.8 | 116.3 | 1.39 | 1.377 | 2.47 | 117.4 | 1.40 | 1.386 | 2.95 |
| rcp-code | 83.8 | 127.8 | 1.53 | 1.514 | 2.72 | 133.7 | 1.60 | 1.579 | 3.37 |
| rcp-prose | 83.8 | 100.2 | 1.20 | 1.188 | 2.12 | 94.0 | 1.12 | 1.115 | 2.34 |
| rcp-recall | 83.7 | 119.5 | 1.43 | 1.418 | 2.53 | 119.4 | 1.43 | 1.414 | 2.99 |

Every category is far above 0.99x. **k=2**: the geometric mean over the categories is 1.36 (k=2) vs 1.35 (k=3), k=3 wins code by 5-8 % and loses chat and prose by 3-8 % (its worst category is 1.07x
against 1.17x), k=2 loads less, and it is the MTP default elsewhere. Chat decode here is a short-prompt measure; for long contexts see section 7.

## 6. Gate 5: G0

- CUDA (`raw/cuda-g0.log`, `scripts/cmtp/cuda-g0.sh`, dev vs origin/main `1f979fad9`, 256 ids after a short and a ~3.5k-token prompt, `plan show` JSON and text): Qwen3-8B and Gemma-4-26B-A4B with no flag:
  ids and plan byte-identical (8B md5 `b43773b5` / `8aca363b`, as in lane dc's G0). 35B: main's no-flag ids (plain decode) equal dev `--spec off`, dev `INTARSIA_SPEC_DEFAULT=0` and dev no flag (the default), plan byte-identical for the two opt-outs.
  `plan show` of the 35B with the default differs (22 text lines): it plans the MTP load (kernel set adds the `_rows` variants, scratch 336 MB -> 84 MB because the macro prefill workspace is not planned; see section 7).
- Metal, endor (`raw/metal-g0.log`, `scripts/cmtp/metal-noop.sh`): 8B and 35B, ids (short, 4k) and `plan show` byte-identical to main.
- HIP: no HIP code, kernel or HIP recipe key changed. The recipe-pair test shows `qwen3_5_moe` on hip/metal/cpu and every other recipe unchanged. No GPU run on tatooine (not taken down).

## 7. Not a gate, but decisive: prefill and time to first token

A loaded MTP head disables the macro prefill pass (`model.rs`: "a speculative sequence never takes a macro pass") and the head also runs over the prompt rows.
`localeval speed` (`raw/longctx/`, 2 interleaved rounds x 3 reps, `scripts/cmtp/longctx.sh`, 128 forced tokens), off vs the default:

| prompt tokens | prefill tok/s off | default | TTFT off | default | decode off | default | output tokens that repay the extra TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1k | 6894-6930 | 5490-5516 (-20 %) | 0.59 s | 0.74 s (+0.15 s) | 80.5 | 97-101 (+21-25 %) | ~65 |
| 16.7k | 6653-6663 | 4161-4197 (-37 %) | 2.51 s | 3.97-4.00 s (+1.5 s) | 73.2 | 84.1-84.3 (+15 %) | ~825 |
| 33.9k | 5894-5911 | 2886-2901 (-51 %) | 5.74 s | 11.70-11.75 s (+6.0 s) | 66.1 | 68.0-70.1 (+3-6 %) | ~9400 |

So for long prompts with short answers (summaries, retrieval) the default is slower end to end; for chat and code generation it is faster. The localeval decode sweeps are filler text (acceptance not
representative). A fix is outside this lane (MTP prefill overhead and re-enabling the macro pass under MTP); the default can be limited by dropping the recipe key.

## 8. Other findings

- **Prefix store**: `--spec off --prefix-store on` gives different answers from `--spec off` without a store (all three requests, including the first, `raw/userpath.log`, `raw/diag.log`). The store changes prefill numerics on its own, with
  or without MTP. With the default (MTP loaded) the answers with the store equal spec-off without it. A user with `--prefix-store` therefore sees their 35B answers change with this default (to the no-store ones).
- Subcommands that load a model without `--spec` inherit the default on the 35B CUDA (`generate`, `serve` single-sequence, `plan show`, `accuracy`); `serve --max-batch N` (N > 1) keeps plain decode.
- Merged head (origin/main with PR #179, binary `5a8ff79bfe47` built from the merge; `raw/recheck-merged.log`): 6/6 prompts (short, ~4k, ~16k; two each) identical to `--spec off` with the default; a lone `--spec-k 3` runs the default with k=3 (`(k from --spec-k)` in the load line, 3.27 tokens per pass) and its 256 ids equal off's.
- Tests on the merged head: `cargo test --release -p intarsia-core -p intarsia-models` 443 passed, 0 failed (`raw/cargo-test-core-models.log`); `intarsia-cli` tests and `scripts/prepush.sh` (docs-sync) pass.

## 9. Reproduce

`scripts/cmtp/spark1.sh build` (fleetq `--load cpu`), `ids.sh`, `window.sh` (speed: `W=s1`; soak: `ARMS="off auto" CATS=none SOAK=40`), `mutant.sh`, `cuda-g0.sh`, `userpath.sh`, `longctx.sh`, `metal-noop.sh`, CPU: `TEST=mtp scripts/cmtp/mutants.sh`,
`TEST=tp_mtp scripts/cmtp/mutants.sh default-optout-ignored ...`. Tables: `python3 -I tools/cmtp/round_table.py raw/s1`, `tools/dfh/cat_table.py raw/s1`.
Binaries: dev `b3fa447331bd` (md5), main `057c74a37ebd`. All GPU work through fleetq on spark1 (`cmtp:` labels); no fleet role server was touched.
