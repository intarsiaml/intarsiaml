asp: `--spec auto`, per-pass choice between the MTP head and the DFlash2 drafter, Qwen3.8-27B MXFP4 tp=2 (HIP), 2026-10-28

Lane `lane-asp/spec-auto`, rebased on `origin/main` `56da97eb4` (after VAT #218). tatooine, 2x R9700, `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, `serve --tp 2 --devices 0,1 --max-batch 1`, localeval `--judge none`, 512 forced tokens, temperature 0, the seven category files of `scripts/dfh/cats/`, driver `scripts/asp/cat-window.sh` (copy of `scripts/vrc/`). Raw: `docs/results/2026-10-28-asp-spec-auto/raw/`. Tables: `tools/dfh/cat_table.py`, `tools/dfh/vrc_spread.py`, `tools/dfh/text_compare.py`. Status: PR, `--spec auto` opt-in; the 27B default stays `--spec mtp --spec-k 2`.

## Headline

`--spec auto --spec-draft DIR` loads the MTP head and the DFlash2 draft and picks the drafter per pass. Greedy output equals `--spec off` (ids 18/18, forced switches, category texts, GPU tests). Speed: it takes most of DFlash2's win on code (2.41x / 2.37x of off, +27.8% / +24.3% over MTP k2) but the policy is too noisy on prose: 6% under MTP k2 on ho-prose and rcp-prose, 1-3% under the per-category best elsewhere. That misses the brief's bar (>= MTP k2 in every category, within about 3% of the best), so auto is not the default. The machinery (state handling, lazy DFlash stash, MTP catch-up) is free: forcing auto to the MTP head costs nothing (section 4).

## 1. Design

Both drafters are lossless under greedy verify: a draft only proposes, the row-exact verify pass decides every emitted token. So switching drafters per pass cannot change the output as long as the target state (KV, GDN rows, rollback) is untouched, which it is: the choice only picks who proposes. What must stay valid for a switch is each drafter's own context:

- MTP head (`MtpState`): queues the hidden row of every pass (verify passes queue without catch-up, rollback keeps the accepted rows); `mtp_draft` catches up over any number of queued rows in one run. Under DFlash-driven passes the queue only grows, and `mtp_queue` overflow would run a collective inside a peer thread; so before a DFlash pass the engine runs a lockstep `Model::mtp_catch_up` when the queue is near `cap_rows`.
- DFlash2 (`DflashState`): ingests the features of every pass. Under MTP passes that would cost a 1.4 ms flush per pass (5% of a 20.8 ms pass), so in auto mode `DflashState::lazy` makes every pass stash its feature rows (a prefix of the stash is exactly the accepted rows after rollback, `DflashState::cut`) and the rows are ingested in one batch before the next DFlash draft.

Policy (`spec::AutoPolicy`, pure host logic, deterministic, unit-tested): per drafter an EWMA (weight 0.2, averaging evenly over the first samples) of tokens per pass and of measured ms per pass (draft + verify + rollback, excluding the first pass after a switch, which pays the catch-up); score = tokens / ms; the better score wins with a 3% hysteresis; each drafter gets 2 seeding passes at request start (MTP first); the idle drafter is probed for one pass every 48 passes (96 when the winner leads by 30% or more; a probe that alone beats the leader earns a second one). `INTARSIA_SPEC_AUTO=mtp|dflash|alt:N|log` (comma-separated) forces a drafter, switches every N passes (tests, A/B) or logs one summary line per request. k: the MTP head drafts 2, DFlash `--spec-k` (default 7); verify passes are 8 rows at most. No graph steps in auto (eager passes, as tp=2 has no captured speculative step).

Memory: the DFlash2 bf16 draft (3.85 GB) is on rank 0 next to the MTP head; `--spec auto` loads and serves 16k-token prompts at `--max-seq 20480 --mem-budget 29` (weights 17.02 GB in the load line, `raw/speed1/ids/gen-tp2-16k-*-auto.err`), all six 16k ids prompts pass.

## 2. Gates

| gate | result |
|---|---|
| ids, `--spec auto` greedy vs `--spec off`, tp=2, 18 prompts (6 short, 6 ~4k, 6 ~16k), 256 tokens | **18 / 18** twice (window ids1 on the pre-VAT tree, `raw/ids1/`; window speed1 on rebased main, `raw/speed1/ids/`); window speed2 again 18 / 18 (`raw/speed2/ids/`) |
| forced switches on HIP tp=2, `INTARSIA_SPEC_AUTO=alt:N` (MTP and DFlash alternating every N passes) | alt:2 **18 / 18** (ids1, speed1); alt:1 and alt:3 17 / 18 each (`raw/gates1/`): both misses are `short-0` against the `--spec off` run of that window; that off run differs from the `off` run of window ids1 at token 63, and alt:1, alt:2, alt:3 and auto all equal the ids1 off run (md5 `e8b55aee...`). Plain `--spec off` run-to-run nondeterminism at tp=2 (coordinator: tracked separately), not auto |
| forced switches on CPU, `spec_auto_greedy_ids_equal_spec_off_with_forced_switches_and_the_policy` (`tests/dflash2_cpu.rs`): the hybrid tiny model with the head and the DFlash2 draft, modes mtp, dflash, alt:1/2/5, scored policy, own drafts and scripted drafts hitting every acceptance pattern, prefill chunks 8 and 64 (small chunk = the MTP queue overflows without the catch-up), max_tokens 1..120, follow-up requests | pass |
| category texts vs `--spec off` (7 categories, `tools/dfh/text_compare.py`) | speed1: mtp2 108 / 108, dfl7 108 / 108, auto **71 / 72** (`raw/speed1/text-compare.txt`); speed2 (no off arm): auto 108 / 108 and automtp 108 / 108 equal to mtp2 (`raw/speed2/text-compare-vs-mtp2.txt`). The one miss: speed1 round 1, `ho-prose` `prose-bakery`, diverges at char 41; off, mtp2, dfl7 and auto round 2 of the same case are identical. Not reproduced: a soak of the same prompt (16 requests, 256 forced tokens, `raw/speed2/soak.txt`) gives one answer for auto, for auto forced to MTP, and equal to mtp2. Treated as the tracked tp=2 nondeterminism (coordinator's rule); not chased |
| GPU tests (`INTARSIA_BATCH_TEST_BACKEND=hip`, `raw/gates1/`) | batch 6, batch_recurrent 14, mtp 21, hybrid 11, fusion_p2 2, graphs 7, hybrid_mxfp4 1, spec 4 on card 0; tp_hybrid 8, tp_mtp 7, tp_batch 14, tp 5 on both cards: all pass (pre-VAT tree; no model-level code of the lane changed afterwards except the policy constants) |
| `--spec off`, `mtp`, `dflash` unchanged | `off` 56.9 tok/s, mtp2 and dfl7 within round noise of the VAT-era numbers, texts equal to off (above). Code: the DFlash `lazy` flag is false outside auto, `forward`/`forward_verify` untouched, the MTP/DFlash draft calls untouched |
| `plan show` for configs without auto | not run against a base binary (no second build). The planner is untouched: no file under `plan.rs`, `memplan.rs`, `ops.rs` or a recipe changed; `--spec auto` adds no plan op |
| Mac `cargo test --workspace --release --no-fail-fast` | green, 1079 passed, 0 failed (final tree, includes the CPU forced-switch test) |

## 3. Decode, tp=2, interleaved rounds (window speed1, rebased main, `raw/speed1/`)

Pooled decode tok/s (tokens / sum of decode times), 3 rounds for off / mtp2 / dfl7, **2 rounds for auto** (the window's cap cut round 3 after one category); round-to-round spread 0.1-1.1% (auto up to 3.3% on ho-prose), `raw/speed1/cat-table.txt`, spread by `tools/dfh/vrc_spread.py`.

| category | off | mtp k2 | dflash k7 | auto | auto vs best | auto vs mtp k2 | DFlash share of auto passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| ho-chat | 56.9 | **90.3** | 79.8 | 89.5 | -0.9% | -0.9% | 17% |
| ho-code | 56.9 | 107.5 | **139.7** | 137.4 | -1.6% | +27.8% | 66% |
| ho-prose | 56.9 | **80.4** | 65.6 | 75.4 | -6.2% | -6.2% | 31% |
| ho-recall | 56.9 | 95.7 | **95.9** | 94.8 | -1.1% | -0.9% | 11% |
| rcp-code | 56.9 | 108.6 | **139.1** | 135.0 | -2.9% | +24.3% | 63% |
| rcp-prose | 56.9 | **83.3** | 68.5 | 77.9 | -6.5% | -6.5% | 29% |
| rcp-recall | 56.9 | 98.8 | **109.4** | 106.9 | -2.3% | +8.2% | 36% |

Tokens per pass (auto / dfl7 / mtp2): ho-code 4.64 / 5.32 / 2.74, ho-prose 2.23 / 2.50 / 2.04, rcp-recall 3.20 / 4.17 / 2.50. Switches per request 5-12 (`raw/speed1/serve-auto-r*.log`, `INTARSIA_SPEC_AUTO=log`).

Reading: on prose the right answer is MTP alone (rate 0.079 vs DFlash 0.063 tokens/ms); auto spends 30% of its passes on DFlash there. Cause: acceptance in prose is a few tokens per pass with large pass-to-pass variance, the idle drafter's estimate rests on 2 seeding passes (the first untimed) plus rare probes, and the first-token acceptance at the start of a response is low, so the early MTP estimate is pessimistic and the EWMA (weight 0.2) takes dozens of passes to forget it. On code the policy reaches DFlash after the seeds and loses 2-3% to the seeds and probes.

## 4. Overhead of the machinery, and one tuning attempt (window speed2, `raw/speed2/`)

`automtp` = `--spec auto` with `INTARSIA_SPEC_AUTO=mtp` (both drafters loaded, lazy DFlash stash running, MTP always drafting) against `mtp2`, 3 rounds: ho-chat 89.9 / 90.5, ho-code 107.7 / 108.4, ho-prose 80.3 / 80.5, ho-recall 95.1 / 95.5, rcp-code 108.6 / 108.3, rcp-prose 83.3 / 83.1, rcp-recall 98.5 / 98.8: within 0.7%, i.e. holding the DFlash context lazily is free for the MTP passes.

A steadier policy (EWMA weight 0.08, 12% hysteresis, probes every 64 / 192) was tried in the same window with 3 rounds: it was worse (ho-chat 84.4, ho-prose 69.4, rcp-code 129.8, rcp-prose 77.3; better only on ho-recall 97.5). The slow EWMA keeps a bad early estimate longer. Reverted; the shipped constants are the speed1 ones.

## 5. Not done / next

- Prose: the policy needs a better estimator, not smoothing. Ideas, untested: priors for tokens and ms per pass per drafter (the pass costs are stable: MTP k2 about 25.7 ms, DFlash k7 about 39.7 ms) so the decision is a token-ratio threshold (DFlash needs about 1.54x the MTP tokens per pass); a longer idle-drafter memory across requests; discounting the first 16 tokens of a response.
- Default: not changed. `--spec auto` is opt-in; the recipe has no way to name the DFlash2 draft checkpoint (only `--spec-draft`), and the numbers above do not meet the default bar.
- Speed rounds: auto has 2 full rounds in speed1 (3 in speed2 with the rejected constants).
- Batching (`--max-batch > 1`), `--prefix-store` bypass and tensor-parallel groups across processes are refused as for `--spec mtp`.

## 6. Fleet

Windows (all `radiance` down, restored and `/v1/models` answering after each; `raw/downtime.log`): #6425 (empty, wrong stages name, 14 s), #6427 ids1 (8 min), #6429 gates1 (11 min), #6433 speed1 (64 min), window 5 speed2 (about 51 min, the coordinator's approved fifth). CPU-slot builds #6423, #6432, #6434. Cleanup #6440: `~/intarsiaml-asp`, volumes `intarsia-cargo-asp-dev` and `intarsia-cargo-asp-dev-hiptest` removed; `.worktrees/asp` removed after the merge.
